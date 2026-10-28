"""Add the API settings keys to the four languages, in the file's own flat style."""
import json
import sys
from pathlib import Path

I18N = Path(__file__).resolve().parents[1] / "mafdet" / "ui" / "i18n"

KEYS = {
    "en": {
        "settings.section.api": "API",
        "settings.api.provider": "Provider",
        "settings.api.provider.hint": "Where the chat goes. The reading ahead is the server's prompt cache: it only pays where the server has one, and above its minimum.",
        "settings.api.model": "Model",
        "settings.api.model.hint": "The name the server knows this model by.",
        "settings.api.base": "Server address",
        "settings.api.base.hint": "The /v1 address. The key comes from the environment variable the provider names, never from this file.",
    },
    "it": {
        "settings.section.api": "API",
        "settings.api.provider": "Fornitore",
        "settings.api.provider.hint": "Dove va la chat. La lettura anticipata è la cache di prompt del server: paga solo dove il server ce l'ha, e sopra il suo minimo.",
        "settings.api.model": "Modello",
        "settings.api.model.hint": "Il nome con cui il server conosce questo modello.",
        "settings.api.base": "Indirizzo del server",
        "settings.api.base.hint": "L'indirizzo /v1. La chiave viene dalla variabile d'ambiente che il fornitore indica, mai da questo file.",
    },
    "fr": {
        "settings.section.api": "API",
        "settings.api.provider": "Fournisseur",
        "settings.api.provider.hint": "Où va la discussion. La lecture anticipée, c'est le cache de prompt du serveur : elle ne paie que là où le serveur en a un, et au-dessus de son minimum.",
        "settings.api.model": "Modèle",
        "settings.api.model.hint": "Le nom sous lequel le serveur connaît ce modèle.",
        "settings.api.base": "Adresse du serveur",
        "settings.api.base.hint": "L'adresse /v1. La clé vient de la variable d'environnement que le fournisseur indique, jamais de ce fichier.",
    },
    "de": {
        "settings.section.api": "API",
        "settings.api.provider": "Anbieter",
        "settings.api.provider.hint": "Wohin der Chat geht. Das Vorauslesen ist der Prompt-Cache des Servers: er lohnt nur, wo der Server einen hat, und oberhalb seines Minimums.",
        "settings.api.model": "Modell",
        "settings.api.model.hint": "Der Name, unter dem der Server dieses Modell kennt.",
        "settings.api.base": "Serveradresse",
        "settings.api.base.hint": "Die /v1-Adresse. Der Schlussel kommt aus der Umgebungsvariablen, die der Anbieter nennt, nie aus dieser Datei.",
    },
}

ADD = {
    "en": {"settings.api.none": "No provider (a model of our own)", "settings.api.no_key": "no key"},
    "it": {"settings.api.none": "Nessun fornitore (un modello nostro)", "settings.api.no_key": "senza chiave"},
    "fr": {"settings.api.none": "Aucun fournisseur (un modele a nous)", "settings.api.no_key": "sans cle"},
    "de": {"settings.api.none": "Kein Anbieter (ein eigenes Modell)", "settings.api.no_key": "ohne Schluessel"},
}
for lang, pairs in ADD.items():
    KEYS[lang].update(pairs)

for lang, pairs in KEYS.items():
    path = I18N / f"{lang}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for k, v in pairs.items():
        if k in data and not data[k].startswith("PLACEHOLDER"):
            continue
        data[k] = v
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(f"{lang}: {len(data)} keys")
