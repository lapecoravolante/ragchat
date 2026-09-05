"""Singleton lazy-loaded HuggingFace embedding model (intfloat/multilingual-e5-large)."""

import logging
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

_MODEL_NAME = "intfloat/multilingual-e5-large"

_embeddings_instance: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the singleton HuggingFaceEmbeddings instance.

    On the first call the model is loaded (downloaded from HuggingFace if not
    already cached). Subsequent calls return the cached instance.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        logger.info("Caricamento modello embedding '%s'...", _MODEL_NAME)
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=_MODEL_NAME,
            model_kwargs={"device": "cpu"},
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
