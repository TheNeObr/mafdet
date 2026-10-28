"""The API backend against a mock OpenAI-compatible server with a prefix cache.

The cache here is what the real servers do — keep what they computed, serve the longest prefix
of a new request that they have already seen — and it is the only part of the mechanism the
backend can see. Words stand for tokens in the mock; making the two agree is `prefix.py`'s
whole job, tested there against the real tokenizer.

What these tests hold the backend to: a warm at a pause makes the next request cheaper, a
request that changed the text mid-way only gets back what it still shares, and a provider that
does not claim /tokenize does not get to use logit_bias.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mafdet.api_backend import ApiBackend, wait
from mafdet.core import prefix, providers

CACHE = []                      # the word-lists the mock has computed: what a prefix cache holds
LOCK = threading.Lock()


def tok(s):
    return [w for w in s.replace("\n", " ").split() if w]


def common(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def handle_one_request(self):
        # A cancelled stream closes the socket mid-answer: the server notices, and that is the
        # client doing the right thing, not an error to print.
        try:
            super().handle_one_request()
        except (ConnectionError, BrokenPipeError):
            self.close_connection = True

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))

    def do_GET(self):
        if self.path.endswith("/models"):
            return self._json({"object": "list", "data": [{"id": "test-model"}]})
        self._json({"error": "nope"}, 404)

    def do_POST(self):
        if self.path.endswith("/tokenize"):
            return self._json({"tokens": list(range(1, len(tok(self._body()["text"])) + 1))})
        body = self._body()
        p = tok(" ".join(m["content"] for m in body["messages"]))
        with LOCK:
            served = max([common(p, seen) for seen in CACHE], default=0)
            CACHE.append(p)
        self._usage = {"prompt_tokens": len(p), "completion_tokens": 1,
                       "prompt_tokens_details": {"cached_tokens": served}}
        if not body.get("stream"):
            return self._json({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                               "usage": self._usage})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n')
        self.wfile.flush()
        time.sleep(0.02)
        if body["max_tokens"] > 1:
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"the answer"}}]}\n\n')
            self.wfile.flush()
            time.sleep(0.02)
        self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}],
                                                "usage": self._usage}) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@pytest.fixture
def server():
    with LOCK:
        CACHE.clear()
    s = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_address[1]}/v1"
    s.shutdown()


def backend(url, **cap):
    return ApiBackend(dict(providers.find("generic"), base_url=url, **cap), model="test-model")


def test_the_server_answers_and_the_model_is_there(server):
    b = backend(server)
    assert b.models() == ["test-model"] and b.ready()


def test_a_warm_makes_the_next_request_cheaper(server):
    b = backend(server)
    msgs = [{"role": "user", "content": "quanto pesa la luna"}]
    assert wait(b.warm(msgs)).cached_tokens == 0          # nothing computed yet: the first read pays
    answer = wait(b.generate(msgs, 64))
    assert answer.cached_tokens == 4                      # four words, all of them from the cache
    assert answer.reasoning_text == "thinking"            # a warm stops there: it is not an answer
    assert answer.text == "the answer"


def test_text_added_after_the_pause_is_paid_for_and_the_rest_is_not(server):
    b = backend(server)
    wait(b.warm([{"role": "user", "content": "quanto pesa la"}]))
    assert wait(b.generate([{"role": "user", "content": "quanto pesa la luna rispetto"}],
                            64)).cached_tokens == 3               # three words cached, six asked
    # A word that changed is not the same token: the cache serves what is still shared, which
    # here is the one word before the change.
    assert wait(b.generate([{"role": "user", "content": "quanto costa la luna"}],
                            64)).cached_tokens == 1


def test_the_engines_stable_text_is_what_the_cache_serves(server):
    # The two must agree exactly: what prefix.py says cannot change any more is what comes back
    # from the cache, and the words after it are paid for again.
    b = backend(server)
    stable = prefix.stable_text("quanto pesa la", holdback=0)
    wait(b.warm([{"role": "user", "content": stable}]))
    text = "quanto pesa la luna rispetto alla terra"
    assert wait(b.generate([{"role": "user", "content": text}], 64)).cached_tokens == len(tok(stable))


def test_logit_bias_needs_a_server_that_can_tokenize(server):
    # A provider that does not claim /tokenize cannot name the tokens, so the bias is not sent:
    # the profile is the promise, and the backend keeps it even when the server would answer.
    plain = backend(server)
    assert plain.tokenize("user typing") is None and plain.logit_bias(["user"], -4.0) is None
    can = backend(server, tokenize=True, logit_bias=True, logit_bias_by_id=True)
    assert can.tokenize("user typing") == [1, 2]
    assert can.logit_bias(["user"], -4.0) == {"1": -4.0}


def test_a_cancelled_stream_stops(server):
    b = backend(server)
    s = b.generate([{"role": "user", "content": "oi"}, {"role": "user", "content": "oi"}], 64)
    s.cancel()
    wait(s, 2)
    assert s.cancelled
