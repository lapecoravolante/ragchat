"""Singleton lazy-loaded HuggingFace embedding model (intfloat/multilingual-e5-large)."""

import logging
from pathlib import Path
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

_MODEL_NAME = "intfloat/multilingual-e5-large"

_embeddings_instance: Optional[HuggingFaceEmbeddings] = None


def _resolve_model_name() -> tuple[str, bool]:
    """Restituisce (model_name_or_path, local_files_only).

    Se il modello è già in cache HuggingFace restituisce il percorso locale
    della snapshot directory e ``local_files_only=True``, così
    sentence_transformers non effettua alcuna connessione di rete.
    Se il modello non è in cache restituisce il nome remoto e
    ``local_files_only=False`` per permettere il download.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(_MODEL_NAME, "config.json")
        if cached is not None and isinstance(cached, str):
            snap_dir = str(Path(cached).parent)
            logger.debug("Modello embedding trovato in cache: %s", snap_dir)
            return snap_dir, True
    except Exception:  # noqa: BLE001
        pass
    logger.debug("Modello embedding non in cache, verrà scaricato da HuggingFace.")
    return _MODEL_NAME, False


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the singleton HuggingFaceEmbeddings instance.

    Se il modello è già presente nella cache HuggingFace locale viene caricato
    direttamente senza alcuna connessione di rete (``local_files_only=True``).
    Solo al primo utilizzo, se il modello non è in cache, viene scaricato.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        model_name, local_only = _resolve_model_name()
        logger.info("Caricamento modello embedding (locale=%s)...", local_only)
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu", "local_files_only": local_only},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Modello embedding caricato.")
    return _embeddings_instance


def format_query(text: str) -> str:
    """Prepend the 'query: ' prefix required by multilingual-e5 for user queries."""
    return "query: " + text


def format_passage(text: str) -> str:
    """Prepend the 'passage: ' prefix required by multilingual-e5 for document chunks."""
    return "passage: " + text
