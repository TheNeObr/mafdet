"""The prefix rules at text level: what is safe to have read already, over an API.

The engine used to work in token ids, because it drove llama-server's own KV cache.
Against an OpenAI-compatible server we send messages and the server tokenizes them:
what we control is which *text* we put in the request, and the server's prompt cache
does the rest. The invariant is the same one the token version rests on, restated here:

    a cut just after a whole word — letters or digits, with a separator (or the end)
    next — is a cut at a token boundary, for every text that continues from it.

Checked against the tokenizer of the reference model (research/prefix_boundary.py,
Qwen/Qwen3-4B): of 621 word-end cuts on the samples and on typed messages, 621 are token
boundaries under this rule. The two cuts it is not are the tempting ones:

  - after the space: the space belongs to the NEXT token ("word " ends with its own space
    token, "word more" puts it inside " more"), so the prefix stops being a prefix — 11 of
    621 were boundaries;
  - after punctuation: a BPE merge can take "." into the token together with the newline
    after it ("duplicati.\n\n" is one token), so the cut falls inside a token.

Everything downstream (hold-back, min-new, the debounce) only decides how much work is
wasted, never the result, exactly as before.

Units are words here rather than tokens, because words are what this layer can see:
HOLDBACK is the words at the end that may still be typed, MIN_NEW the whole words that
make a read worth sending. The token numbers they replace (3 and 16 on the reference
tokenizer, 1.7 tokens per word) are the same thresholds seen through the coarser unit.
"""
from __future__ import annotations

HOLDBACK = 2                # whole words at the end kept out of the stable text
MIN_NEW = 10                # whole words added since the last read that make a read worth it

_SPACE = " \t\n\r\v\f"


def is_word_end(text: str, n: int) -> bool:
    """True when a whole word ends exactly at n: letters or digits before, a separator after.

    Punctuation does not end a word for this purpose: it can merge with what follows into one
    token, and a cut there would not be a boundary (see the module docstring).
    """
    if n <= 0 or not text[n - 1].isalnum():
        return False
    return n >= len(text) or text[n] in _SPACE


def word_ends(text: str) -> list[int]:
    """The offsets just past each whole word of `text`, in order."""
    return [n for n in range(1, len(text) + 1) if is_word_end(text, n)]


def common_prefix(a: str, b: str) -> str:
    """The longest text both share, cut where both agree a word ends: what a cache can serve.

    A character-by-character common prefix can stop inside a word, and the two texts would
    then tokenize it differently. The cut goes back to the last position that is a word end
    in both.
    """
    n = 0
    limit = min(len(a), len(b))
    while n < limit and a[n] == b[n]:
        n += 1
    while n > 0 and not (is_word_end(a, n) and is_word_end(b, n)):
        n -= 1
    return a[:n]


def stable_text(text: str, holdback: int = HOLDBACK) -> str:
    """The text up to the end of the last whole word that cannot change any more.

    The words being typed now are left out — `holdback` of them — because they can still
    gain letters and change how the text is split. Trailing whitespace is not part of a
    word and changes nothing: `stable_text("a b ")` == `stable_text("a b")`.
    """
    ends = word_ends(text)
    n = len(ends) - holdback
    return text[:ends[n - 1]] if n > 0 else ""


def whole_words(text: str) -> int:
    """How many words of `text` are already whole (the last one may still be typed)."""
    if text and text[-1] not in _SPACE:
        return max(0, len(word_ends(text)) - 1)
    return len(word_ends(text))


def reuses(cached: str, target: str) -> int:
    """Characters of `target` a cache holding `cached` can serve."""
    return len(common_prefix(cached, target))
