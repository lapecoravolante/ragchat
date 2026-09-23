"""Singleton lazy-loaded HuggingFace embedding model.

Il nome del modello attivo viene letto da ``ragchat.config`` ogni volta che
viene richiesta l'istanza.  Se il modello configurato e' cambiato rispetto a
quello attualmente caricato, il vecchio singleton viene scartato e il nuovo
modello sara' caricato al prossimo utilizzo (lazy reload).
"""

import logging
from pathlib import Path
from typing import Optional

import httpx
from huggingface_hub import set_client_factory
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Singleton corrente e nome del modello con cui e' stato creato
_embeddings_instance: Optional[HuggingFaceEmbeddings] = None
_loaded_model_name: Optional[str] = None


def _get_model_name() -> str:
    """Legge il nome del modello di embedding attivo dalla configurazione."""
    try:
        import ragchat.config as cfg
        return cfg.load()["embedding_model"]
    except Exception:  # noqa: BLE001
        return "intfloat/multilingual-e5-large"


# Disabilita la verifica dei certificati TLS per le connessioni HuggingFace.
def _insecure_client_factory() -> httpx.Client:
    return httpx.Client(verify=False, follow_redirects=True)


set_client_factory(_insecure_client_factory)


def _resolve_model_path(model_name: str) -> tuple[str, bool]:
    """Restituisce (model_name_or_path, local_files_only).

    Se il modello e' gia' in cache HuggingFace restituisce il percorso locale
    della snapshot directory e ``local_files_only=True``, cosi'
    sentence_transformers non effettua alcuna connessione di rete.
    Se il modello non e' in cache restituisce il nome remoto e
    ``local_files_only=False`` per permettere il download.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(model_name, "config.json")
        if cached is not None and isinstance(cached, str):
            snap_dir = str(Path(cached).parent)
            logger.debug("Modello embedding trovato in cache: %s", snap_dir)
            return snap_dir, True
    except Exception:  # noqa: BLE001
        pass
    logger.debug("Modello embedding non in cache, verra' scaricato da HuggingFace.")
    return model_name, False


def get_embeddings() -> HuggingFaceEmbeddings:
    """Restituisce il singleton HuggingFaceEmbeddings per il modello attivo.

    Se il modello configurato e' cambiato rispetto a quello caricato, il
    singleton precedente viene invalidato e il nuovo sara' creato al prossimo
    accesso (lazy reload).  Il caricamento effettivo avviene qui, alla prima
    chiamata con il nuovo modello.
    """
    global _embeddings_instance, _loaded_model_name

    model_name = _get_model_name()

    if _embeddings_instance is not None and _loaded_model_name != model_name:
        logger.info(
            "Modello embedding cambiato (%s -> %s): il vecchio singleton viene scartato.",
            _loaded_model_name,
            model_name,
        )
        _embeddings_instance = None
        _loaded_model_name = None

    if _embeddings_instance is None:
        model_path, local_only = _resolve_model_path(model_name)
        logger.info(
            "Caricamento modello embedding '%s' (locale=%s)...", model_name, local_only
        )
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": "cpu", "local_files_only": local_only},
            encode_kwargs={"normalize_embeddings": True},
        )
        _loaded_model_name = model_name
        logger.info("Modello embedding '%s' caricato.", model_name)

    return _embeddings_instance


def format_query(text: str) -> str:
    """Prepend the 'query: ' prefix required by multilingual-e5 for user queries."""
    return "query: " + text


def format_passage(text: str) -> str:
    """Prepend the 'passage: ' prefix required by multilingual-e5 for document chunks."""
    return "passage: " + text
