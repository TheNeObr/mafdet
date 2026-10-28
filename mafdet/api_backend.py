"""The API backend: an OpenAI-compatible server, reached over HTTP with nothing installed.

The engine speaks `messages` now, not token ids: the server owns the chat template and the
tokenizer, and what we control is which text we put in the request. The reading ahead is a
request like any other — a `warm`, one token of generation over the stable prefix, whose only
job is to make the server keep that prefix in its prompt cache; on Enter the request is the
same text plus what was typed since, and the server serves the part it has seen before. That
is the whole mechanism, and it is the server's, not ours: we cannot trim a cache we do not
own, only ask for the same prefix in the same order.

Stdlib only (http.client), as the rest of the app: the app must not need a library to talk to
a server that is already speaking HTTP.

The response's `usage` is the honest part: `prompt_tokens_details.cached_tokens` says how much
of the prefix the server actually served from cache — a measured number where the catalog used
to put an estimate. Servers that do not report it answer with nothing, and the panel says so
instead of making it up.
"""
from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.parse

from .core import providers
from .core.tasks import Cancelled

TIMEOUT = 600
USER_AGENT = "mafdet"


class ApiStream:
    """One streamed generation, run in its own thread and read by whoever needs it.

    The text arrives in pieces on `content` and `reasoning`; `done` says it ended on its own,
    `cancelled` that we stopped it. `info` keeps the last event (timings, usage) when the
    server sends one.
    """

    def __init__(self, backend: "ApiBackend", messages: list, max_tokens: int, sampling: dict | None = None,
                 warm: bool = False):
        self.backend, self.messages, self.max_tokens = backend, messages, max_tokens
        self.sampling, self.warm = sampling, warm
        self.content, self.reasoning = [], []
        self.info: dict = {}
        self.done = self.cancelled = False
        self.error = ""
        self.cond = threading.Condition()
        self._conn: http.client.HTTPConnection | None = None
        threading.Thread(target=self._run, daemon=True).start()

    @property
    def text(self) -> str:
        return "".join(self.content)

    @property
    def reasoning_text(self) -> str:
        return "".join(self.reasoning)

    @property
    def cached_tokens(self) -> int:
        """What the server says it served from its cache for this request (0 if it does not say)."""
        u = self.info.get("usage") or {}
        d = u.get("prompt_tokens_details") or {}
        return int(d.get("cached_tokens") or u.get("cache_read_input_tokens") or 0)

    def cancel(self):
        with self.cond:
            if self.done or self.cancelled:
                return
            self.cancelled = True
            self.cond.notify_all()
        if self._conn is not None:              # closing the socket is how the server stops too
            try:
                self._conn.close()
            except OSError:
                pass

    def _run(self):
        body = {"messages": self.messages, "max_tokens": self.max_tokens, "stream": True,
                "stream_options": {"include_usage": True}, **(self.sampling or self.backend.sampling)}
        try:
            self.backend._post_stream(body, self)
        except Cancelled:
            pass
        except Exception as e:
            self.error = str(e)
        finally:
            with self.cond:
                self.done = True
                self.cond.notify_all()


class ApiBackend:
    """A provider profile plus the model and key to use on it."""

    # The sampling the reference model's authors recommend, kept as the default for servers that
    # take it; a provider profile can carry its own.
    SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0}

    def __init__(self, provider: dict, model: str = "", api_key: str | None = None, sampling: dict | None = None):
        self.provider = provider
        self.base = (provider.get("base_url") or "").rstrip("/")
        self.model = model or provider.get("model") or ""
        self.key = api_key if api_key is not None else providers.api_key(provider)
        self.sampling = sampling or self.SAMPLING
        if not self.base:
            raise ValueError("the provider has no base_url: nothing to talk to")

    # ---- the calls ------------------------------------------------------------

    def ready(self) -> bool:
        """The server answers and the model is there: what `load` needs to be true."""
        try:
            models = self.models()
        except Exception:
            return False
        return not self.model or self.model in models

    def models(self) -> list[str]:
        data = self._get("/models")
        return [m.get("id") or m.get("name") for m in (data.get("data") or []) if (m.get("id") or m.get("name"))]

    def warm(self, messages: list, sampling: dict | None = None) -> ApiStream:
        """Read the prefix in: one token of generation, so the server caches what it computed.

        A warm on a server without a prompt cache is a plain cost: it still returns the number
        (0), and the panel shows the difference instead of hiding it.
        """
        return ApiStream(self, messages, 1, sampling, warm=True)

    def generate(self, messages: list, max_tokens: int, sampling: dict | None = None) -> ApiStream:
        return ApiStream(self, messages, max_tokens, sampling)

    def tokenize(self, text: str) -> list[int] | None:
        """Token ids, when the server exposes /tokenize: the only way logit_bias comes back."""
        if not self.provider.get("tokenize"):
            return None
        data = self._post("/tokenize", {"text": text})
        return data.get("tokens") or None

    def logit_bias(self, words: list[str], bias: float) -> dict | None:
        """The words made less likely, in the form this server takes: ids, or token strings."""
        if not self.provider.get("logit_bias") or bias == 0:
            return None
        out = {}
        for w in words:
            if self.provider.get("logit_bias_by_id"):
                ids = self.tokenize(w)
                if ids and len(ids) == 1:
                    out[str(ids[0])] = bias
            else:
                out[w] = bias
        return out or None

    # ---- HTTP ------------------------------------------------------------------

    def _conn(self) -> http.client.HTTPConnection:
        u = urllib.parse.urlparse(self.base)
        if u.scheme not in ("http", "https"):
            raise ValueError(f"{self.base}: an http(s) address is needed")
        cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        return cls(u.netloc, timeout=TIMEOUT)

    def _headers(self) -> dict:
        h = {"User-Agent": USER_AGENT, "Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        return h

    def _path(self, p: str) -> str:
        root = urllib.parse.urlparse(self.base).path.rstrip("/")
        return f"{root}{p}" if p.startswith("/") else p

    def _get(self, p: str) -> dict:
        return self._request("GET", p)

    def _post(self, p: str, body: dict) -> dict:
        return self._request("POST", p, body)

    def _request(self, method: str, p: str, body: dict | None = None) -> dict:
        conn = self._conn()
        try:
            conn.request(method, self._path(p), json.dumps(body) if body is not None else None, self._headers())
            data = conn.getresponse().read()
        finally:
            conn.close()
        return json.loads(data) if data else {}

    def _post_stream(self, body: dict, stream: ApiStream) -> None:
        """One SSE response, feeding `stream` as the pieces arrive."""
        conn = self._conn()
        stream._conn = conn
        try:
            conn.request("POST", self._path("/chat/completions"), json.dumps(body), self._headers())
            resp = conn.getresponse()
            if resp.status >= 400:
                raise RuntimeError(f"{self.base}/chat/completions answered {resp.status}: "
                                   f"{resp.read()[:300].decode('utf-8', 'replace')}")
            for raw in resp:
                if stream.cancelled:
                    return
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                with stream.cond:
                    if event.get("usage"):
                        stream.info = {"usage": event["usage"], **{k: v for k, v in event.items() if k != "usage"}}
                    for ch in event.get("choices") or []:
                        delta = ch.get("delta") or {}
                        field = self.provider.get("reasoning_field") or "reasoning_content"
                        if delta.get(field):
                            stream.reasoning.append(delta[field])
                            stream.cond.notify_all()
                        if delta.get("content"):
                            stream.content.append(delta["content"])
                            stream.cond.notify_all()
                        if ch.get("finish_reason"):
                            stream.done = True
                if stream.done:
                    break
        finally:
            stream._conn = None
            conn.close()


def wait(stream: ApiStream, timeout: float = 0) -> ApiStream:
    """Block until the stream ends (a warm is waited on only when its number is wanted)."""
    deadline = time.perf_counter() + timeout if timeout else None
    with stream.cond:
        while not stream.done:
            left = None if deadline is None else deadline - time.perf_counter()
            if left is not None and left <= 0:
                break
            stream.cond.wait(left if left is not None else 1.0)
    return stream
