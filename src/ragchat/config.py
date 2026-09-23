"""Gestione del file di configurazione config.properties.

Il file viene cercato nella stessa cartella dell'eseguibile (o del file
``__main__.py`` quando si lancia con ``python -m ragchat``).
Se non esiste viene creato automaticamente con i valori di default.

Convenzione per le liste di modelli
------------------------------------
I valori che rappresentano liste di stringhe vengono serializzati come
sequenze separate dal carattere ``|`` (pipe), senza spazi attorno al
separatore.  Esempio::

    embedding_models = intfloat/multilingual-e5-large|BAAI/bge-m3

Le funzioni :func:`models_to_str` / :func:`str_to_models` gestiscono
la conversione in entrambe le direzioni.
"""

import configparser
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Separatore usato per serializzare le liste di modelli nel file .properties
_LIST_SEP = "|"

# ---------------------------------------------------------------------------
# Helpers lista modelli  (devono precedere DEFAULTS che li usa)
# ---------------------------------------------------------------------------


def models_to_str(models: list[str]) -> str:
    """Converte una lista di nomi modello nella stringa da salvare su disco."""
    return _LIST_SEP.join(m.strip() for m in models if m.strip())


def str_to_models(value: str) -> list[str]:
    """Converte la stringa letta da disco in lista di nomi modello."""
    return [m.strip() for m in value.split(_LIST_SEP) if m.strip()]


# ---------------------------------------------------------------------------
# Valori di default
# ---------------------------------------------------------------------------

# Modelli di embedding disponibili di default (il primo è quello attivo)
_DEFAULT_EMBEDDING_MODELS = [
    "intfloat/multilingual-e5-large",
]

# Modelli LLM disponibili di default (il primo è quello attivo)
_DEFAULT_QUERY_MODELS = [
    "bartowski/gemma-2-2b-it-GGUF",
]

DEFAULTS: dict[str, str] = {
    "embedding_models": models_to_str(_DEFAULT_EMBEDDING_MODELS),
    "embedding_model": _DEFAULT_EMBEDDING_MODELS[0],
    "query_models": models_to_str(_DEFAULT_QUERY_MODELS),
    "query_model": _DEFAULT_QUERY_MODELS[0],
    "top_k": "4",
    "log_level": "ERROR",
    "default_db_path": "",
    "prompt_template": (
        "Sei un assistente utile. Rispondi alla domanda basandoti "
        "esclusivamente sul contesto fornito.\n"
        "Se il contesto non contiene informazioni sufficienti per "
        "rispondere, dillo chiaramente.\n\n"
        "Contesto:\n{context}\n\n"
        "Domanda: {question}\n\n"
        "Risposta:"
    ),
}

SECTION = "ragchat"

# ---------------------------------------------------------------------------
# Percorso del file
# ---------------------------------------------------------------------------


def _config_path() -> Path:
    """Restituisce il percorso di config.properties accanto all'applicazione."""
    if getattr(sys, "frozen", False):
        # Bundle PyInstaller
        base = Path(sys.executable).parent
    else:
        # Sviluppo: directory del package ragchat (src/ragchat)
        base = Path(__file__).parent
    return base / "config.properties"


# ---------------------------------------------------------------------------
# API pubblica
# ---------------------------------------------------------------------------


def load() -> dict[str, str]:
    """Carica la configurazione da disco; crea il file con i default se assente.

    Returns:
        Dizionario chiave->valore con tutte le impostazioni.
    """
    path = _config_path()
    parser = configparser.ConfigParser()

    if not path.exists():
        _write_defaults(path, parser)
        logger.info("config.properties creato con valori di default in %s", path)
    else:
        parser.read(path, encoding="utf-8")
        logger.debug("Configurazione caricata da %s", path)

    result: dict[str, str] = {}
    for key, default in DEFAULTS.items():
        result[key] = parser.get(SECTION, key, fallback=default)

    # Integrita': se il modello attivo non e' nella lista, aggiungilo in testa
    for kind in ("embedding", "query"):
        active = result[f"{kind}_model"]
        lst = str_to_models(result[f"{kind}_models"])
        if active not in lst:
            lst.insert(0, active)
            result[f"{kind}_models"] = models_to_str(lst)

    return result


def save(values: dict[str, str]) -> None:
    """Salva *values* nel file config.properties.

    Args:
        values: Dizionario chiave->valore da persistere.
    """
    path = _config_path()
    parser = configparser.ConfigParser()
    # Legge l'esistente per non perdere chiavi non gestite dall'UI
    if path.exists():
        parser.read(path, encoding="utf-8")
    if not parser.has_section(SECTION):
        parser.add_section(SECTION)
    for key, value in values.items():
        parser.set(SECTION, key, value)
    with path.open("w", encoding="utf-8") as fh:
        parser.write(fh)
    logger.info("Configurazione salvata in %s", path)


# ---------------------------------------------------------------------------
# Helpers interni
# ---------------------------------------------------------------------------


def _write_defaults(path: Path, parser: configparser.ConfigParser) -> None:
    parser.add_section(SECTION)
    for key, value in DEFAULTS.items():
        parser.set(SECTION, key, value)
    with path.open("w", encoding="utf-8") as fh:
        parser.write(fh)
