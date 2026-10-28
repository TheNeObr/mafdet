"""The session over an API: the same chat, at the level of text and messages.

watch.py's Session works in token ids because it drove llama-server's own KV cache. Here the
server owns the tokenizer and its prompt cache, and what we still control is the text we put in
each request:

  - at a pause, if enough whole words arrived, a `warm` on the stable prefix: one token of
    generation, whose only job is to make the server keep what it computed — the number that
    comes back (`cached_tokens` in the usage) is the real one, measured by the server, where
    the catalog used to put an estimate;
  - the thinking-ahead: the notes the model wrote at a pause ride along in the final request as
    an assistant turn, which is what they always were in the prompt, said in the language the
    API speaks. The seed frame goes as an assistant prefill where the server takes one
    (assistantPrefill in the profile); where it does not, the frame rides as a system message
    and the bias is simply not sent — the frame alone measured 41-54% of notes still about the
    message instead of the 17-26% with the bias. That loss is the price of the API, and it was
    agreed before the cut;
  - on Enter: the messages as they are, and the answer streamed back.

The rules that do not change: the hold-back, the minimum new words before a read is worth
sending, the notes surviving edits, and only the exact text earning the answer. They are in
core/prefix.py and tested there — including the word-boundary cut, which is what lets the
server's cache serve the prefix at all.
"""
from __future__ import annotations

import threading
import time

from .api_backend import ApiBackend, wait
from .core import prefix, providers

DEBOUNCE = 0.25          # the quiet that makes a pause a pause (watch.py uses the same)
MIN_NEW = prefix.MIN_NEW    # whole words that make a read worth sending


def backend_for(settings: dict, provider: dict | None = None) -> ApiBackend:
    """The provider to talk to, with the model and the address the settings say."""
    p = provider or providers.find(settings.get("provider", ""))
    if p is None:
        raise ValueError("no provider")
    return ApiBackend(dict(p, base_url=settings.get("api_base") or p["base_url"]),
                      model=settings.get("api_model") or p["model"])


class ApiSession:
    """What ChatService talks to when the model is an API: the same duck, over HTTP."""

    def __init__(self, backend: ApiBackend, think: bool = True, think_ahead: int = 1024,
                 max_tokens: int = 0, branches: int = 0, on_note=None):
        self.backend, self.think = backend, think
        self.think_ahead, self.max_tokens, self.branches = think_ahead, max_tokens, branches
        self.on_note = on_note
        self.state, self.error, self.label = "loading", "", backend.model
        self.messages: list[dict] = []        # the conversation, as the server sees it
        self.notes: list[dict] = []           # notes written ahead, as assistant turns
        self.text = ""                        # what the user has typed of the current message
        self.read_words = 0                   # words the last read covered
        self.new_since_read = 0
        self.stats = {"reads": 0, "cached": 0, "asked": 0}
        self.generating = self.busy = False
        self.stop = threading.Event()
        self._lock = threading.RLock()
        self._pause: threading.Timer | None = None
        self._warm = None                     # the read in flight, so a new one does not stack
        if backend.ready():
            self.state = "ready"

    # ---- what the interfaces show --------------------------------------------------

    def status(self) -> dict:
        with self._lock:
            s = {"state": self.state, "error": self.error, "model": self.label, "ctx": 0,
                 "generating": self.generating, "busy": self.busy, "has_text": bool(self.text.strip()),
                 "words": prefix.whole_words(self.text), "read_words": self.read_words,
                 "reads": self.stats["reads"], "read": self.stats["cached"],
                 "left": max(0, self.stats["asked"] - self.stats["cached"]),
                 "cached_tokens": self.stats["cached"], "asked_tokens": self.stats["asked"],
                 "think": self.think, "provider": (self.backend.provider or {}).get("id", "")}
            if self.branches:
                s["branches"] = 0
            return s

    # ---- typing ---------------------------------------------------------------------

    def draft(self, text: str):
        """The page calls this on every change. A pause is what makes a read happen."""
        with self._lock:
            if text == self.text:
                return self.status()
            added = max(0, prefix.whole_words(text) - prefix.whole_words(self.text))
            self.text, self.new_since_read = text, self.new_since_read + added
            if self._pause:
                self._pause.cancel()
            if self.state == "ready" and self.new_since_read >= MIN_NEW:
                self._pause = threading.Timer(DEBOUNCE, self._read_ahead)
                self._pause.daemon = True
                self._pause.start()
        return self.status()

    def _read_ahead(self):
        with self._lock:
            if self.generating or self.busy or self.new_since_read < MIN_NEW:
                return
            stable = prefix.stable_text(self.text)
            if not stable:
                return
            self.new_since_read = 0
            self.read_words = len(stable.split())
            self.busy = True
            self.stats["reads"] += 1
            self.stats["asked"] += self.read_words
            msgs = list(self.messages) + [{"role": "user", "content": stable}]
            if self._warm is not None:
                self._warm.cancel()           # the previous read is now stale
        try:
            s = wait(self.backend.warm(msgs), 60)
            cached = s.cached_tokens
        except Exception:
            cached = 0
        finally:
            with self._lock:
                self._warm = None
                self.busy = False
                if cached:
                    self.stats["cached"] += cached

    # ---- sending ---------------------------------------------------------------------

    def send(self, text: str):
        if self.state != "ready":
            yield {"type": "error", "message": "The provider is not ready."}
            return
        self.stop.clear()
        self.generating = True
        t0 = time.perf_counter()
        with self._lock:
            if self._pause:
                self._pause.cancel()
            msgs = list(self.messages) + self.notes + [{"role": "user", "content": text}]
            self.text, self.read_words, self.new_since_read, self.notes = "", 0, 0, []
        stream = self.backend.generate(msgs, self.max_tokens or 1024)
        ttft, seen, seen_r = None, 0, 0
        try:
            while True:
                with stream.cond:
                    stream.cond.wait(0.2)
                    content, reasoning, done = "".join(stream.content), "".join(stream.reasoning), stream.done
                if ttft is None and (content or reasoning):
                    ttft = time.perf_counter() - t0
                if len(reasoning) > seen_r:
                    yield {"type": "think", "text": reasoning[seen_r:]}
                    seen_r = len(reasoning)
                if len(content) > seen:
                    yield {"type": "token", "text": content[seen:]}
                    seen = len(content)
                if done or self.stop.is_set():
                    break
            if self.stop.is_set():
                stream.cancel()
            with stream.cond:
                cached, usage = stream.cached_tokens, stream.info.get("usage") or {}
            self.messages = msgs + [{"role": "assistant", "content": stream.text}]
            self.stats["asked"] += len(text.split())
            self.stats["cached"] += cached
            if stream.error:
                yield {"type": "error", "message": stream.error}
            yield {"type": "done", "ttft_ms": round((ttft or 0) * 1000),
                   "tokens": usage.get("completion_tokens", seen + seen_r),
                   "cached_tokens": cached, "stopped": self.stop.is_set(), "reads": self.stats["reads"]}
        except Exception as e:
            yield {"type": "error", "message": str(e) or type(e).__name__}
        finally:
            self.generating = False

    def new_chat(self):
        with self._lock:
            self.messages, self.notes, self.text = [], [], ""
            self.read_words, self.new_since_read = 0, 0
            self.stats = {"reads": 0, "cached": 0, "asked": 0}

    def close(self):
        with self._lock:
            if self._pause:
                self._pause.cancel()
