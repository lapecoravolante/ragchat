"""Gestione del file di configurazione config.json.

Il file viene cercato nella stessa cartella del package ``ragchat``
(sia ``src/ragchat/`` in sviluppo che la directory dell'eseguibile
quando si lancia con PyInstaller).
Se non esiste viene creato automaticamente con i valori di default.

Formato JSON nativo con tipi forti.

Le liste di modelli sono memorizzate come array JSON di dizionari con le chiavi:
- ``id``: identificativo stringa del modello (repo-id o URL)
- ``url``: URL di download del modello
- ``format``: formato del file (``"gguf"``, ``"safetensors"``)
- ``tags``: lista di stringhe da mostrare nel tooltip della UI

I modelli attivi (``embedding_model`` e ``query_model``) sono stringhe
con l'ID del modello selezionato.  ``top_k`` e' un intero.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CONFIG_FILENAME = "config.json"

# Modelli di embedding disponibili di default
_DEFAULT_EMBEDDING_MODELS: list[dict[str, Any]] = [
    {
        "id": "intfloat/multilingual-e5-large",
        "url": "https://huggingface.co/intfloat/multilingual-e5-large",
        "format": "safetensors",
        "tags": ["feature-extraction", "multilingual", "sentence-similarity"],
    },
]

# Modelli LLM disponibili di default
_DEFAULT_QUERY_MODELS: list[dict[str, Any]] = [
    {
        "id": "bartowski/gemma-2-2b-it-GGUF",
        "url": "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
        "format": "gguf",
        "tags": ["text-generation", "gemma", "instruction-tuned", "Q4_K_M"],
    },
]

_DEFAULT_DATA: dict = {
    "embedding_models": list(_DEFAULT_EMBEDDING_MODELS),
    "embedding_model": _DEFAULT_EMBEDDING_MODELS[0]["id"],
    "query_models": list(_DEFAULT_QUERY_MODELS),
    "query_model": _DEFAULT_QUERY_MODELS[0]["id"],
    "top_k": 4,
    "hf_max_models": 500,
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


def model_entry_id(entry: dict) -> str:
    """Restituisce l'ID di un'entrata modello."""
    return entry.get("id", "")


def model_entry_url(entry: dict) -> str:
    """Restituisce l'URL di download di un'entrata modello."""
    return entry.get("url", entry.get("id", ""))


def model_entry_format(entry: dict) -> str:
    """Restituisce il formato di un'entrata modello (``"gguf"`` o ``"safetensors"``)."""
    return entry.get("format", "safetensors")


def model_entry_tags(entry: dict) -> list[str]:
    """Restituisce i tag di un'entrata modello."""
    return list(entry.get("tags", []))


class Config:
    """Gestione della configurazione dell'applicazione in formato JSON."""

    @classmethod
    def path(cls) -> Path:
        """Restituisce il percorso di config.json accanto al package ragchat."""
        if getattr(sys, "frozen", False):
            base = Path(sys.executable).parent
        else:
            base = Path(__file__).parent.parent
        return base / _CONFIG_FILENAME

    @classmethod
    def get_defaults(cls) -> dict:
        """Restituisce una copia dei valori di default."""
        return json.loads(json.dumps(_DEFAULT_DATA))

    @classmethod
    def load(cls) -> dict:
        """Carica la configurazione da disco; crea il file con i default se assente.

        Returns:
            Dizionario chiave->valore con tutte le impostazioni e tipi nativi
            JSON (``list[dict]`` per le liste modelli, ``int`` per ``top_k``).
        """
        path = cls.path()

        if not path.exists():
            values = cls.get_defaults()
            cls.save(values)
            logger.info("config.json creato con valori di default in %s", path)
        else:
            try:
                with path.open(encoding="utf-8") as fh:
                    values = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                logger.error("Errore lettura %s (%s): ripristino default", path, exc)
                values = cls.get_defaults()
                cls.save(values)
            logger.debug("Configurazione caricata da %s", path)

        cls._ensure_consistency(values)
        return values

    @classmethod
    def save(cls, values: dict) -> None:
        """Salva *values* nel file config.json.

        Args:
            values: Dizionario chiave->valore da persistere.
        """
        path = cls.path()
        with path.open("w", encoding="utf-8") as fh:
            json.dump(values, fh, indent=2, ensure_ascii=False)
        logger.info("Configurazione salvata in %s", path)

    @classmethod
    def _ensure_consistency(cls, values: dict) -> None:
        """Garantisce coerenza della configurazione caricata.

        Riempie le chiavi mancanti con i valori di default, normalizza
        ``top_k`` a intero e verifica che il modello attivo sia presente
        nella lista corrispondente.
        """
        defaults = cls.get_defaults()

        for key, default_val in defaults.items():
            if key not in values:
                values[key] = (
                    list(default_val)
                    if isinstance(default_val, list)
                    else default_val
                )

        if "top_k" in values:
            try:
                values["top_k"] = int(values["top_k"])
            except (TypeError, ValueError):
                values["top_k"] = defaults["top_k"]

        if "hf_max_models" in values:
            try:
                values["hf_max_models"] = int(values["hf_max_models"])
            except (TypeError, ValueError):
                values["hf_max_models"] = defaults["hf_max_models"]
