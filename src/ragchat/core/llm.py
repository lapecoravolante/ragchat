"""
Wrapper per il modello LLM locale gemma2-2b GGUF tramite llama-cpp-python.

Gestisce:
- Download automatico del file GGUF da HuggingFace al primo utilizzo
- Istanziazione singleton lazy-loaded di LlamaCpp (langchain-community)
- Graceful degradation se llama-cpp-python non è installato
- Caricamento bundled delle VC++ runtime DLL (Windows) senza installazione di sistema
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Costanti HuggingFace
REPO_ID = "bartowski/gemma-2-2b-it-GGUF"
FILENAMES = ["gemma-2-2b-it-Q4_K_M.gguf", "gemma-2-2b-it-Q8_0.gguf"]

# Cartella vendor con le VC++ runtime DLL incluse nel progetto
_VENDOR_RUNTIME_DIR = Path(__file__).parent.parent / "vendor" / "win_runtime"


def _register_windows_dll_dirs() -> None:
    """
    Su Windows, registra tramite os.add_dll_directory:
    1. La cartella vendor/win_runtime del progetto (MSVCP140, VCRUNTIME140, …)
       — permette di usare l'app senza installare il VC++ Redistributable di sistema.
    2. La cartella lib/ del pacchetto llama_cpp (ggml-*.dll, llama.dll, …).

    L'ordine è importante: le VC++ runtime vengono registrate per prime così
    le DLL di llama_cpp le trovano già disponibili nel loader di Windows.
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
            "Verrà usata l'installazione di sistema se disponibile.",
            _VENDOR_RUNTIME_DIR,
        )

    # 2. DLL native di llama_cpp (ggml-base, ggml, ggml-cpu, llama, mtmd)
    try:
        import importlib.util as _ilu
        spec = _ilu.find_spec("llama_cpp")
        if spec and spec.origin:
            _lib_dir = Path(spec.origin).parent / "lib"
            if _lib_dir.is_dir():
                os.add_dll_directory(str(_lib_dir))
                logger.debug("llama_cpp lib/ registrata: %s", _lib_dir)
    except Exception:  # noqa: BLE001
        pass


# Verifica disponibilità llama-cpp-python
try:
    _register_windows_dll_dirs()
    import llama_cpp  # noqa: F401
    _LLAMA_AVAILABLE = True
    logger.debug("llama-cpp-python disponibile.")
except (ImportError, RuntimeError, OSError):
    _LLAMA_AVAILABLE = False
    logger.warning(
        "llama-cpp-python non è installato o non caricabile. "
        "Il LLM locale non sarà disponibile. "
        "Per installarlo: pip install llama-cpp-python"
    )

# Singleton
_llm_instance = None


def is_llm_available() -> bool:
    """
    Restituisce True se llama-cpp-python è importabile, False altrimenti.

    Usato dalla GUI per decidere se mostrare messaggi di errore o avvertimenti
    relativi all'assenza del runtime LLM.
    """
    return _LLAMA_AVAILABLE


def get_model_path() -> Optional[str]:
    """
    Restituisce il percorso locale del file GGUF se già presente in cache,
    None altrimenti (senza scaricare nulla).
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        logger.warning("huggingface_hub non è installato.")
        return None

    for filename in FILENAMES:
        cached = try_to_load_from_cache(repo_id=REPO_ID, filename=filename)
        # try_to_load_from_cache restituisce None o un sentinel se non trovato
        if cached is not None and isinstance(cached, str):
            logger.debug("Modello trovato in cache: %s", cached)
            return cached

    return None


def _download_model() -> str:
    """
    Scarica il file GGUF da HuggingFace Hub e restituisce il percorso locale.
    Prova prima Q4_K_M, poi Q8_0 come fallback.

    Raises:
        RuntimeError: se nessun file può essere scaricato.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub non è installato. "
            "Installarlo con: pip install huggingface-hub"
        ) from exc

    last_error: Optional[Exception] = None
    for filename in FILENAMES:
        try:
            logger.info("Download modello GGUF: %s/%s …", REPO_ID, filename)
            path = hf_hub_download(repo_id=REPO_ID, filename=filename)
            logger.info("Modello scaricato in: %s", path)
            return path
        except Exception as exc:  # noqa: BLE001
            logger.warning("Download fallito per %s: %s", filename, exc)
            last_error = exc

    raise RuntimeError(
        f"Impossibile scaricare il modello GGUF da {REPO_ID}. "
        f"Ultimo errore: {last_error}"
    )


def get_llm():
    """
    Restituisce l'istanza singleton del LLM (LlamaCpp).

    Al primo invocazione:
    1. Verifica che llama-cpp-python sia installato.
    2. Cerca il file GGUF in cache; se assente lo scarica.
    3. Istanzia LlamaCpp con i parametri di default.

    Returns:
        Istanza ``LlamaCpp`` di langchain-community.

    Raises:
        ImportError: se llama-cpp-python non è installato.
        RuntimeError: se il download del modello fallisce.
    """
    global _llm_instance

    if _llm_instance is not None:
        return _llm_instance

    if not _LLAMA_AVAILABLE:
        raise ImportError(
            "llama-cpp-python non è installato. "
            "Installarlo con: pip install llama-cpp-python"
        )

    # Cerca prima in cache, altrimenti scarica
    model_path = get_model_path()
    if model_path is None:
        model_path = _download_model()

    try:
        from langchain_community.llms import LlamaCpp
    except ImportError as exc:
        raise ImportError(
            "langchain-community non è installato. "
            "Installarlo con: pip install langchain-community"
        ) from exc

    logger.info("Caricamento modello da: %s", model_path)
    _llm_instance = LlamaCpp(
        model_path=model_path,
        n_ctx=4096,
        temperature=0.7,
        max_tokens=512,
        n_threads=4,
        verbose=False,
    )
    logger.info("LLM pronto.")
    return _llm_instance
