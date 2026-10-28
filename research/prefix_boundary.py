"""Is a space-aligned cut really a token boundary? Checked against a real tokenizer.

core/prefix.py cuts the common prefix at the end of a whole word, and the whole read-ahead
rests on that being a token boundary for every text that continues from it. Checked here on
the tokenizer of the reference model (Qwen/Qwen3-4B), on the bundled samples and on texts
typed the way the chat gets them:

  - a cut at the end of a word (a separator follows) is a token boundary: the tokens of the
    shorter text are a prefix of the tokens of the longer one;
  - a cut after the space is NOT: the space belongs to the next token, so "word " ends with
    its own space token while "word more" puts it inside " more" — the prefix stops being a
    prefix, and a cache keyed on it silently never matches;
  - a cut inside a word is not either, which is why the hold-back exists.

Run: python research/prefix_boundary.py            (needs transformers; downloads the tokenizer)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mafdet.core import prefix                                        # noqa: E402
from mafdet.text import read_sample                                   # noqa: E402

MODEL = "Qwen/Qwen3-4B"


def ids(tok, s):
    return tok(s, add_special_tokens=False).input_ids


def boundary_ok(tok, short, long):
    """`short` is a prefix of `long`: does it tokenize as a prefix of long's tokens?"""
    a, b = ids(tok, short), ids(tok, long)
    return b[:len(a)] == a


def cuts(text):
    """Every space-aligned cut of `text`, as prefix.py would make them (and the trap next to them)."""
    for n in prefix.word_ends(text):
        yield n, "end of word"
        if n < len(text) and text[n] in " \t\n":
            yield n + 1, "after the space"


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)

    texts = [read_sample(n).strip() for n in ("messaggio_medio.txt", "documento.txt", "domanda.txt")]
    texts += ["Quanto pesa la Luna rispetto alla Terra",
              "Luca ha 25 euro, compra 3 quaderni da 4 euro e una penna da 2 euro. Quanti euro gli restano?",
              "Mi puoi spiegare come funziona la cache dei transformer, per favore"]

    bad = {"end of word": 0, "after the space": 0}
    checked = {"end of word": 0, "after the space": 0}
    for text in texts:
        for k in range(1, len(text)):
            short, long = text[:k], text
            for n, kind in cuts(long):
                if n != k or not long.startswith(short):
                    continue
                checked[kind] += 1
                if not boundary_ok(tok, short, long):
                    bad[kind] += 1
                    if kind == "end of word" and bad[kind] <= 3:
                        print(f"  not a boundary: {short!r}")

    # The hold-back rule, checked the same way: what stable_text keeps must always be a boundary.
    fails = 0
    for text in texts:
        for k in range(len(text) + 1):
            stable = prefix.stable_text(text[:k])
            if stable and not boundary_ok(tok, stable, text[:k]):
                fails += 1
                if fails <= 3:
                    print(f"  stable_text is not a boundary: {stable!r} in {text[:k]!r}")

    print(f"\ncuts at the end of a word: {checked['end of word'] - bad['end of word']} / "
          f"{checked['end of word']} are token boundaries")
    print(f"cuts after the space:      {checked['after the space'] - bad['after the space']} / "
          f"{checked['after the space']} are token boundaries")
    print(f"stable_text over prefixes: {fails} failures")
    print("(the second line is the trap the module warns about; on the reference tokenizer it "
          "does fail, which is why the cut is before the space.)")
    return 0 if fails == 0 and checked["end of word"] and bad["end of word"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
