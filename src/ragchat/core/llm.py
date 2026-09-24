"""
Wrapper per il modello LLM locale GGUF tramite llama-cpp-python.

Gestisce:
- Download automatico del file GGUF da HuggingFace al primo utilizzo
- Istanziazione singleton lazy-loaded di LlamaCpp (langchain-community)
- Invalidazione e reload del singleton quando il modello attivo cambia in config
- Graceful degradation se llama-cpp-python non e' installato
- Caricamento bundled delle VC++ runtime DLL (Windows) senza installazione di sistema
- Risoluzione del modello da URL HuggingFace (hf_hub_download + LlamaCpp)
  o repo-id (snapshot_download + transformers via langchain)
"""

import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Regex per estrarre repo_id e filename da una URL HuggingFace
# Formato: https://huggingface.co/{repo_id}/(blob|resolve)/{revision}/{filename}
_HF_URL_RE = re.compile(
    r"^https?://huggingface\.co/([^/]+/[^/]+)/(?:blob|resolve)/([^/]+)/(.+)$"
)

# Cartella vendor con le VC++ runtime DLL incluse nel progetto
_VENDOR_RUNTIME_DIR = Path(__file__).parent.parent / "vendor" / "win_runtime"


def _get_model_spec() -> str:
    """Legge lo specificatore del modello LLM attivo dalla configurazione.

    Il valore puo' essere:
    - una URL HuggingFace a un file GGUF specifico
    - un semplice repo-id (es. ``bartowski/gemma-2-2b-it-GGUF``)
    """
    try:
        from ragchat.utils.config import Config
        return Config.load()["query_model"]
    except Exception:  # noqa: BLE001
        return Config.get_defaults()["query_model"]


def _is_hf_url(spec: str) -> bool:
    """Verifica se *spec* e' una URL HuggingFace a un file GGUF."""
    return _HF_URL_RE.match(spec) is not None


def _parse_hf_url(url: str) -> Tuple[str, str]:
    """Estrae ``(repo_id, filename)`` da una URL HuggingFace.

    Raises:
        ValueError: se l'URL non e' valida.
    """
    match = _HF_URL_RE.match(url)
    if not match:
        raise ValueError(f"URL HuggingFace non valida: {url}")
    repo_id = match.group(1)
    filename = match.group(3)
    return repo_id, filename


def _resolve_model_spec(spec: str) -> Tuple[str, Optional[str]]:
    """Risolve lo specificatore modello in ``(repo_id, filename)``.

    Se *spec* e' una URL HuggingFace, *filename* e' estratto dall'URL.
    Se *spec* e' un repo-id, *filename* e' ``None`` (verra' determinato
    al momento del download tramite ``snapshot_download``).
    """
    if _is_hf_url(spec):
        return _parse_hf_url(spec)
    return spec, None


def _register_windows_dll_dirs() -> None:
    """
    Su Windows, registra tramite os.add_dll_directory:
    1. La cartella vendor/win_runtime del progetto (MSVCP140, VCRUNTIME140, ...)
       -- permette di usare l'app senza installare il VC++ Redistributable di sistema.
    2. La cartella lib/ del pacchetto llama_cpp (ggml-*.dll, llama.dll, ...).

    L'ordine e' importante: le VC++ runtime vengono registrate per prime cosi'
    le DLL di llama_cpp le trovano gia' disponibili nel loader di Windows.
    """
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return

    # 1. VC++ runtime bundled nel progetto
    if _VENDOR_RUNTIME_DIR.is_dir():
        os.add_dll_directory(str(_VENDOR_RUNTIME_DIR))
        logger.debug("VC++ runtime bundled registrate: %s", _VENDOR_RUNTIME_DIR)
    else:
        logger.debug(
            "Cartella vendor/win_runtime non trovata (%s). "
            "Verra' usata l'installazione di sistema se disponibile.",
            _VENDOR_RUNTIME_DIR,
        )

    # 2. DLL native di llama_cpp (ggml-base, ggml, ggml-cpu, llama, mtmd)
    try:
        import importlib.util as _ilu
        ilu_spec = _ilu.find_spec("llama_cpp")
        if ilu_spec and ilu_spec.origin:
            _lib_dir = Path(ilu_spec.origin).parent / "lib"
            if _lib_dir.is_dir():
                os.add_dll_directory(str(_lib_dir))
                logger.debug("llama_cpp lib/ registrata: %s", _lib_dir)
    except Exception:  # noqa: BLE001
        pass


# Verifica disponibilita' llama-cpp-python
try:
    _register_windows_dll_dirs()
    import llama_cpp  # noqa: F401
    _LLAMA_AVAILABLE = True
    logger.debug("llama-cpp-python disponibile.")
except (ImportError, RuntimeError, OSError):
    _LLAMA_AVAILABLE = False
    logger.warning(
        "llama-cpp-python non e' installato o non caricabile. "
        "Il LLM locale non sara' disponibile. "
        "Per installarlo: pip install llama-cpp-python"
    )

# Singleton corrente e specificatore con cui e' stato creato
_llm_instance = None
_loaded_spec: Optional[str] = None


def is_llm_available() -> bool:
    """
    Restituisce True se llama-cpp-python (per URL GGUF) oppure transformers
    (per repo-id) e' disponibile, False altrimenti.
    """
    if _LLAMA_AVAILABLE:
        return True
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def get_model_path() -> Optional[str]:
    """
    Restituisce il percorso locale del modello attivo se gia' presente in
    cache, None altrimenti (senza scaricare nulla).

    Se il modello e' specificato come URL HuggingFace, restituisce il percorso
    del file GGUF in cache. Se e' un repo-id, restituisce il percorso della
    directory snapshot in cache (usato da transformers).
    """
    spec = _get_model_spec()
    repo_id, filename = _resolve_model_spec(spec)

    try:
        from huggingface_hub import try_to_load_from_cache, snapshot_download
    except ImportError:
        logger.warning("huggingface_hub non e' installato.")
        return None

    if filename is not None:
        # URL HuggingFace: controlla la cache per quel file specifico
        cached = try_to_load_from_cache(repo_id=repo_id, filename=filename)
        if cached is not None and isinstance(cached, str):
            logger.debug("Modello trovato in cache: %s", cached)
            return cached
        return None

    # Repo-id: verifica che lo snapshot sia gia' presente in cache
    try:
        snapshot_path = snapshot_download(repo_id, local_files_only=True)
    except Exception:  # noqa: BLE001
        return None

    logger.debug("Modello trovato in cache: %s", snapshot_path)
    return snapshot_path


def _download_model(spec: str) -> str:
    """
    Scarica il modello da HuggingFace Hub e restituisce il percorso locale.

    Se *spec* e' una URL HuggingFace, usa ``hf_hub_download`` per scaricare
    il file GGUF specifico. Se *spec* e' un repo-id, usa ``snapshot_download``
    per scaricare l'intero repository.

    Raises:
        RuntimeError: se il download fallisce o huggingface_hub non e' installato.
    """
    repo_id, filename = _resolve_model_spec(spec)

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub non e' installato. "
            "Installarlo con: pip install huggingface-hub"
        ) from exc

    if filename is not None:
        # URL HuggingFace: scarica il file GGUF specifico
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )
        logger.info("Modello GGUF disponibile in: %s", path)
        return path

    # Repo-id: scarica l'intero snapshot
    logger.info("Download modello da: %s ...", repo_id)
    path = snapshot_download(repo_id)
    logger.info("Modello disponibile in: %s", path)
    return path


def _create_llama_cpp(model_path: str):
    """Istanzia ``LlamaCpp`` da un file GGUF scaricato.

    Raises:
        ImportError: se langchain-community o llama-cpp-python non sono installati.
    """
    if not _LLAMA_AVAILABLE:
        raise ImportError(
            "llama-cpp-python non e' installato. "
            "Installarlo con: pip install llama-cpp-python"
        )

    try:
        from langchain_community.llms import LlamaCpp
    except ImportError as exc:
        raise ImportError(
            "langchain-community non e' installato. "
            "Installarlo con: pip install langchain-community"
        ) from exc

    logger.info("Caricamento modello LLM GGUF da: %s", model_path)
    return LlamaCpp(
        model_path=model_path,
        n_ctx=4096,
        temperature=0.7,
        max_tokens=512,
        n_threads=4,
        verbose=False,
    )


def _create_hf_llm(repo_path: str):
    """Istanzia un modello HuggingFace tramite langchain.

    Usa ``transformers`` per caricare il modello e ``HuggingFacePipeline``
    di langchain-community per l'interfaccia.

    Raises:
        ImportError: se transformers, torch o langchain-community
            non sono installati.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except ImportError as exc:
        raise ImportError(
            "transformers non e' installato. "
            "Installarlo con: pip install transformers"
        ) from exc

    try:
        from langchain_community.llms import HuggingFacePipeline
    except ImportError as exc:
        raise ImportError(
            "langchain-community non e' installato. "
            "Installarlo con: pip install langchain-community"
        ) from exc

    logger.info("Caricamento modello LLM transformers da: %s", repo_path)
    tokenizer = AutoTokenizer.from_pretrained(repo_path)
    model = AutoModelForCausalLM.from_pretrained(repo_path)
    hf_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
    )
    return HuggingFacePipeline(pipeline=hf_pipe)


def get_llm():
    """
    Restituisce il singleton LLM per il modello attivo.

    Se lo specificatore configurato e' cambiato rispetto a quello caricato, il
    singleton precedente viene invalidato e il nuovo verra' creato al prossimo
    accesso (lazy reload).

    Al primo accesso con uno specificatore dato:
    1. Cerca il modello in cache; se assente lo scarica da HuggingFace.
    2. Se *spec* e' una URL HuggingFace, crea un'istanza ``LlamaCpp``
       (llama-cpp-python). Se *spec* e' un repo-id, carica il modello con
       ``transformers`` tramite ``HuggingFacePipeline`` (langchain-community).

    Returns:
        Istanza LLM di langchain-community (``LlamaCpp`` o ``HuggingFacePipeline``).

    Raises:
        ImportError: se le dipendenze necessarie non sono installate.
        RuntimeError: se il download del modello fallisce.
    """
    global _llm_instance, _loaded_spec

    spec = _get_model_spec()

    if _llm_instance is not None and _loaded_spec != spec:
        logger.info(
            "Modello LLM cambiato (%s -> %s): il vecchio singleton viene scartato.",
            _loaded_spec,
            spec,
        )
        _llm_instance = None
        _loaded_spec = None

    if _llm_instance is not None:
        return _llm_instance

    # Cerca prima in cache, altrimenti scarica
    model_path = get_model_path()
    if model_path is None:
        model_path = _download_model(spec)

    if _is_hf_url(spec):
        _llm_instance = _create_llama_cpp(model_path)
    else:
        _llm_instance = _create_hf_llm(model_path)

    _loaded_spec = spec
    logger.info("LLM '%s' pronto.", spec)
    return _llm_instance
