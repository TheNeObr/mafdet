"""Does the read-ahead survive over an API? Measured on a real server, not on the mock.

The mechanism on this backend is the server's prompt cache: at a pause we send what is stable
with one token of generation (the `warm`), the server computes and keeps it, and the request
made on Enter is served from there — only the tail is computed again. On the mock that is an
agreement between two pieces of my own code. Here it is the server saying so.

Fair by construction, and the first version of this script was not: every trial gets a message
the server has never seen, because a message sent twice is served from cache in both arms and
the comparison measures nothing. The two arms get different messages of the same length, and
what is read is the server's own `timings` — the OpenAI endpoint does not report cached_tokens,
and a field that is absent is not a zero.

Run: python research/api_spike.py [base_url]        (default http://127.0.0.1:8080/v1)
"""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mafdet.api_backend import ApiBackend, wait                                        # noqa: E402
from mafdet.core import prefix, providers                                               # noqa: E402

PAUSE_AFTER = 3              # whole words typed before the pause where the read happens
TRIALS = 6

# Sentences of the same number of words, one per trial, so no trial inherits another's cache.
SENTENCES = [
    "A lua gira em torno da terra enquanto o sol queima gases ha bilhões de anos",
    "O oceano cobre a maior parte do planeta e ainda guarda criaturas que ninguém viu",
    "Uma vela acesa ilumina mais que mil palavras ditas sobre a mesma escuridão fria",
    "O trem das seis chega atrasado toda manhã e ninguém parece disposto a resolver isso",
    "Meu avô plantou uma árvore que hoje dá sombra para três casas além da nossa cerca",
    "A impressora funciona apenas quando ninguém precisa dela e isso tem uma explicação",
    "Cada degrau da escada gasto pela metade conta uma história que ninguém escreveu",
    "O farol acesso à noite guia barcos que nunca passam por aqui de dia",
    "A fila do café anda mais devagar quando estou atrasado para a reunião",
    "Um gato dorme dezesseis horas e acorda exatamente quando alguém abre um pacote",
]


def cold(b, message):
    """No typing ahead: the whole message arrives at once, on Enter."""
    t = time.perf_counter()
    data = b._post("/chat/completions", {"messages": [{"role": "user", "content": message}], "max_tokens": 1})
    return time.perf_counter() - t, data


def warm_then_enter(b, message):
    """Type, pause, read ahead, then Enter: what the app will do."""
    stable = prefix.stable_text(" ".join(message.split()[:PAUSE_AFTER]), holdback=0)
    b._post("/chat/completions", {"messages": [{"role": "user", "content": stable}], "max_tokens": 1})
    t = time.perf_counter()
    data = b._post("/chat/completions", {"messages": [{"role": "user", "content": message}], "max_tokens": 1})
    return time.perf_counter() - t, data


def pasted(b, doc, question, warm_first):
    """A document, a pause, then the question. Both arms do the same work at Enter and generate
    the same number of tokens: the only difference is whether the document was already read.

    A fresh document per trial (a marker line on top), or the second arm would inherit the
    first one's cache and measure nothing.
    """
    if warm_first:
        b._post("/chat/completions", {"messages": [{"role": "user", "content": doc}], "max_tokens": 1})
    t = time.perf_counter()
    data = b._post("/chat/completions",
                   {"messages": [{"role": "user", "content": doc + "\n\n" + question}], "max_tokens": 32})
    return time.perf_counter() - t, data


def prefill_ms(data):
    t = data.get("timings") or {}
    return t.get("pre_fill_ms") or t.get("prompt_ms")


def usage(data, key):
    return (data.get("usage") or {}).get(key)


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080/v1"
    b = ApiBackend(dict(providers.find("generic"), base_url=base))
    if not b.ready():
        print(f"nothing answering on {base}")
        return 1
    b.model = b.models()[0]
    print(f"server: {base}\nmodel:  {b.model}")
    print(f"pause after {PAUSE_AFTER} words, {TRIALS} trials, one fresh sentence per trial\n")

    for label, run in (("no read-ahead (cold)", cold), ("read-ahead at the pause", warm_then_enter)):
        sentences = SENTENCES[:TRIALS] if run is cold else SENTENCES[TRIALS:2 * TRIALS]
        secs, pre = [], []
        for s in sentences:
            dt, data = run(b, s)
            secs.append(dt * 1000)
            p = prefill_ms(data)
            pre.append(p if p is not None else float("nan"))
            time.sleep(0.2)
        ok = [x for x in pre if x == x]
        print(f"{label:24s}  Enter -> first token: {statistics.mean(secs):6.0f} ms"
              f"   prefill at Enter: {statistics.mean(ok) if ok else float('nan'):6.0f} ms")

    # The case where prefill is not free: a document pasted, a pause, then the question. The
    # reading ahead has work to save here, and the answer is what the user waits for.
    doc = (Path(__file__).resolve().parents[1] / "mafdet" / "samples" / "documento.txt").read_text(encoding="utf-8").strip()
    question = "In quanti articoli si articola il documento? Rispondi solo con il numero."
    print(f"\na document of {len(doc.split())} words pasted, a pause, then the question "
          f"(both arms generate 32 tokens at Enter):")
    for n, (label, warm_first) in enumerate((("no read-ahead (cold)", False), ("read-ahead at the pause", True))):
        secs, pre = [], []
        for k in range(TRIALS):
            fresh = f"[{n}-{k}] " + doc        # a document this trial's cache has not seen
            dt, data = pasted(b, fresh, question, warm_first)
            secs.append(dt * 1000)
            p = prefill_ms(data)
            pre.append(p if p is not None else float("nan"))
            time.sleep(0.2)
        ok = [x for x in pre if x == x]
        print(f"{label:24s}  Enter -> first token: {statistics.mean(secs):6.0f} ms"
              f"   prefill at Enter: {statistics.mean(ok) if ok else float('nan'):6.0f} ms")

    print("\n(the prefill column is the one the read-ahead is supposed to move; if it did not,")
    print(" the server is not keeping the prefix, and the mechanism does not survive the API.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
