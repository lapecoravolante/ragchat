"""Singleton caricato pigramente per il modello di embedding HuggingFace.

Il nome del modello attivo viene letto da :mod:`ragchat.utils.config`
ogni volta che viene richiesta l'istanza.  Se il modello configurato
è cambiato rispetto a quello attualmente caricato, il vecchio singleton
viene scartato e il nuovo sarà caricato al prossimo accesso (lazy reload).

Funzioni pubbliche:
    :func:`get_embeddings` — restituisce il singleton corrente.
    :func:`invalidate_embedding_cache` — forza il reload al prossimo accesso.
    :func:`format_query` — aggiunge il prefisso ``query:`` (richiesto da multilingual-e5).
    :func:`format_passage` — aggiunge il prefisso ``passage:`` (richiesto da multilingual-e5).
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
        from ragchat.utils.config import Config
        return Config.load()["embedding_model"]
    except Exception:  # noqa: BLE001
        return "intfloat/multilingual-e5-large"


def _insecure_client_factory() -> httpx.Client:
    """Crea un client ``httpx`` senza verifica dei certificati TLS.

    Usato da ``huggingface_hub`` tramite :func:`set_client_factory` per
    evitare errori di certificato in ambienti aziendali con proxy HTTPS.
    """
    return httpx.Client(verify=False, follow_redirects=True)


set_client_factory(_insecure_client_factory)


def _resolve_model_path(model_name: str) -> tuple[str, bool]:
    """Restituisce la coppia (percorso o nome del modello, solo file locali).

    Se il modello è già nella cache HuggingFace, restituisce il percorso locale
    della directory snapshot e ``local_files_only=True``, così
    sentence_transformers non effettua alcuna connessione di rete.
    Se il modello non è nella cache, restituisce il nome remoto e
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


def invalidate_embedding_cache() -> None:
    """Invalida il singleton degli embedding e ne forza il ricaricamento al prossimo get_embeddings()."""
    global _embeddings_instance, _loaded_model_name
    _embeddings_instance = None
    _loaded_model_name = None
    logger.info("Singleton embeddings invalidato.")


def format_query(text: str) -> str:
    """Aggiunge ``query: `` come prefisso per il modello di embedding configurato.

    Il prefisso è richiesto da multilingual-e5 e viene aggiunto sempre;
    il modello configurato deve quindi supportare questo formato di input.
    """
    return "query: " + text


def format_passage(text: str) -> str:
    """Aggiunge ``passage: `` come prefisso per il modello di embedding configurato.

    Il prefisso è richiesto da multilingual-e5 e viene aggiunto sempre;
    il modello configurato deve quindi supportare questo formato di input.
    """
    return "passage: " + text
