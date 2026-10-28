"""The prefix rules: the word-boundary cut carries the whole invariant, so it is tested as such.

The rule is whitespace only. Punctuation is not a token boundary — a BPE merge can swallow
it ("36, e" can come out as one token) — so a cut before punctuation is not trusted, and
the tests say so. What is trusted is checked against a real tokenizer in
`research/prefix_boundary.py`.
"""
import pytest

from mafdet.core import prefix


def test_a_cut_at_the_end_of_a_whole_word_is_a_boundary():
    # The whole first text is a word-aligned prefix of the second: the cut stays where it is.
    assert prefix.common_prefix("Quanto pesa la Luna", "Quanto pesa la Luna rispetto") == "Quanto pesa la Luna"
    # Here the characters agree up to inside "L", which could still grow ("Luna", "Lucca"): the cut
    # goes back to the last position that ends a word in both texts.
    assert prefix.common_prefix("Quanto pesa la L", "Quanto pesa la Luna") == "Quanto pesa la"


def test_the_cut_survives_the_word_that_made_it_change():
    # A read cut at "la" must still be a prefix of the final text, or the cache cannot serve it.
    cached = prefix.common_prefix("Quanto pesa la L", "Quanto pesa la Luna")
    assert prefix.common_prefix("Quanto pesa la Luna", "Quanto pesa la Luna rispetto alla").startswith(cached)


def test_only_a_space_aligned_end_is_a_boundary():
    # A cut is trusted only after letters or digits, with a separator next. A cut after
    # punctuation is not: a BPE merge can take the punctuation into the token with what follows.
    assert not prefix.is_word_end("ciao.come", 4)
    assert not prefix.is_word_end("36, e", 3)
    assert prefix.is_word_end("36 e", 2)
    # Both texts end a word at the same place: the cut stays where it is.
    assert prefix.common_prefix("3 x 12 = 36, e", "3 x 12 = 36, e poi") == "3 x 12 = 36, e"
    # Here "e" could still grow into "ed": not a boundary in both. The cut goes back to the last
    # word end that is one in both texts.
    assert prefix.common_prefix("3 x 12 = 36, e", "3 x 12 = 36, ed poi") == "3 x 12"


def test_stable_text_leaves_the_words_that_may_still_be_typed():
    text = "one two three four five"
    assert prefix.stable_text(text, holdback=2) == "one two three"
    assert prefix.stable_text("one two three four", holdback=0) == "one two three four"
    assert prefix.stable_text("one", holdback=2) == ""
    # Trailing space is not a word: it changes nothing, and must not sneak into the stable text.
    assert prefix.stable_text("one two ", holdback=0) == "one two"
    assert prefix.stable_text("one two three ", holdback=2) == "one"


def test_whole_words_do_not_count_the_one_being_typed():
    assert prefix.whole_words("one two thre") == 2
    assert prefix.whole_words("one two three ") == 3
    assert prefix.whole_words("one") == 0                    # might still be "ones"
    assert prefix.whole_words("") == 0


def test_reuses_counts_what_the_cache_can_serve():
    assert prefix.reuses("one two three", "one two four") == len("one two")
    assert prefix.reuses("", "one two") == 0


@pytest.mark.parametrize("a,b", [
    ("alpha beta gamma", "alpha beta delta gamma"),
    ("a bb ccc dddd", "a bb ccc dddd eeeee"),
    ("x y", "x y z"),
    ("o gato comeu", "o gato comeu ração"),
])
def test_prefix_is_a_word_aligned_prefix_of_both(a, b):
    p = prefix.common_prefix(a, b)
    assert a.startswith(p) and b.startswith(p)
    if p:
        assert prefix.is_word_end(a, len(p)) and prefix.is_word_end(b, len(p))
