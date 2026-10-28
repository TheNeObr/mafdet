"""Provider profiles for the API backend: how to reach an OpenAI-compatible server.

One entry per server family, with the few things the engine cannot guess:

  base_url            the server's /v1 root. Local servers are the common case here.
  key_env             environment variable holding the key ("" = no key: a local server)
  prefix_cache        does the server keep a prompt cache across requests? The read-ahead
                      (the warm request at a pause) is what feeds it, so on a server without
                      it the warm is a plain cost, not a saving: leave it on and it is the
                      cached_tokens column that tells you the truth.
  cache_min_tokens    below this the server does not cache at all (OpenAI: 1024). A message
                      shorter than this gets no benefit from typing ahead.
  tokenize            the server answers /tokenize: the engine can then count tokens exactly
                      and use logit_bias, which the frames of watch.py rely on.
  logit_bias          the endpoint takes logit_bias. logit_bias_by_id: keyed by token id
                      (needs tokenize); otherwise by token string.
  reasoning_field     where the streamed reasoning arrives, if anywhere ("" = the answer
                      comes as content and the model does not think out loud)

Adding a provider is one entry here; nothing in the engine is provider-specific.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from importlib.resources import files

FIELDS = ("base_url", "key_env", "model", "prefix_cache", "cache_min_tokens", "tokenize",
          "logit_bias", "logit_bias_by_id", "reasoning_field")


@lru_cache(maxsize=1)
def _providers() -> dict:
    return {p["id"]: p for p in json.loads(files("mafdet").joinpath("providers.json").read_text(encoding="utf-8"))["providers"]}


def _fill(p: dict) -> dict:
    out = {f: p.get(f) for f in FIELDS}
    out.update(id=p["id"], name=p.get("name", p["id"]))
    for f in FIELDS:                                  # the JSON carries the defaults; this keeps them in one place
        if out[f] is None:
            out[f] = {"base_url": "", "key_env": "", "model": "", "prefix_cache": False,
                      "cache_min_tokens": 0, "tokenize": False, "logit_bias": False,
                      "logit_bias_by_id": False, "reasoning_field": ""}[f]
    return out


def providers() -> list[dict]:
    """Every profile, in file order (the UI lists them in this order)."""
    return [_fill(p) for p in _providers().values()]


def find(provider_id: str) -> dict | None:
    p = _providers().get(provider_id)
    return _fill(p) if p else None


def api_key(p: dict) -> str:
    """The key, from the environment. Never stored in config.json: the sync would carry it away."""
    return os.environ.get(p["key_env"], "") if p["key_env"] else ""
