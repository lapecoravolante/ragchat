"""
Wrapper per il modello LLM locale GGUF tramite llama-cpp-python.

Gestisce:
- Download automatico del file GGUF da HuggingFace al primo utilizzo
- Istanziazione singleton lazy-loaded di LlamaCpp (langchain-community)
- Invalidazione e reload del singleton quando il modello attivo cambia in config
- Graceful degradation se llama-cpp-python non e' installato
- Caricamento bundled delle VC++ runtime DLL (Windows) senza installazione di sistema
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Repo-id di default; viene sovrascritto dal valore in config.json
_DEFAULT_REPO_ID = "bartowski/gemma-2-2b-it-GGUF"

# File GGUF da cercare nel repo (in ordine di preferenza)
FILENAMES = ["gemma-2-2b-it-Q4_K_M.gguf", "gemma-2-2b-it-Q8_0.gguf"]

# Manteniamo REPO_ID come simbolo di modulo per retrocompatibilita'
REPO_ID = _DEFAULT_REPO_ID

# Cartella vendor con le VC++ runtime DLL incluse nel progetto
_VENDOR_RUNTIME_DIR = Path(__file__).parent.parent / "vendor" / "win_runtime"


def _get_repo_id() -> str:
    """Legge il repo-id del modello LLM attivo dalla configurazione."""
    try:
        from ragchat.utils.config import Config
        return Config.load()["query_model"]
    except Exception:  # noqa: BLE001
        return _DEFAULT_REPO_ID


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
        spec = _ilu.find_spec("llama_cpp")
        if spec and spec.origin:
            _lib_dir = Path(spec.origin).parent / "lib"
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

# Singleton corrente e repo-id con cui e' stato creato
_llm_instance = None
_loaded_repo_id: Optional[str] = None


def is_llm_available() -> bool:
    """
    Restituisce True se llama-cpp-python e' importabile, False altrimenti.
    """
    return _LLAMA_AVAILABLE


def get_model_path() -> Optional[str]:
    """
    Restituisce il percorso locale del file GGUF del repo attivo se gia'
    presente in cache, None altrimenti (senza scaricare nulla).
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        logger.warning("huggingface_hub non e' installato.")
        return None

    repo_id = _get_repo_id()
    for filename in FILENAMES:
        cached = try_to_load_from_cache(repo_id=repo_id, filename=filename)
        if cached is not None and isinstance(cached, str):
            logger.debug("Modello trovato in cache: %s", cached)
            return cached

    return None


def _download_model(repo_id: str) -> str:
    """
    Scarica il file GGUF da HuggingFace Hub e restituisce il percorso locale.
    Prova prima Q4_K_M, poi Q8_0 come fallback.

    Raises:
        RuntimeError: se nessun file puo' essere scaricato.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub non e' installato. "
            "Installarlo con: pip install huggingface-hub"
        ) from exc

    last_error: Optional[Exception] = None
    for filename in FILENAMES:
        try:
            # Prima tenta il caricamento locale (nessuna rete);
            # se il file non e' in cache passa al download effettivo.
            for local_only in (True, False):
                try:
                    if local_only:
                        logger.debug(
                            "Tentativo caricamento locale: %s/%s", repo_id, filename
                        )
                    else:
                        logger.info(
                            "Download modello GGUF: %s/%s ...", repo_id, filename
                        )
                    path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_files_only=local_only,
                    )
                    logger.info("Modello disponibile in: %s", path)
                    return path
                except Exception as inner_exc:  # noqa: BLE001
                    if local_only:
                        logger.debug(
                            "Non trovato in cache locale (%s): %s", filename, inner_exc
                        )
                        continue
                    raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("Download fallito per %s: %s", filename, exc)
            last_error = exc

    raise RuntimeError(
        f"Impossibile scaricare il modello GGUF da {repo_id}. "
        f"Ultimo errore: {last_error}"
    )


def get_llm():
    """
    Restituisce il singleton LlamaCpp per il modello LLM attivo.

    Se il repo-id configurato e' cambiato rispetto a quello caricato, il
    singleton precedente viene invalidato e il nuovo verra' creato al prossimo
    accesso (lazy reload).

    Al primo accesso con un dato repo-id:
    1. Verifica che llama-cpp-python sia installato.
    2. Cerca il file GGUF in cache; se assente lo scarica.
    3. Istanzia LlamaCpp.

    Returns:
        Istanza ``LlamaCpp`` di langchain-community.

    Raises:
        ImportError: se llama-cpp-python non e' installato.
        RuntimeError: se il download del modello fallisce.
    """
    global _llm_instance, _loaded_repo_id

    repo_id = _get_repo_id()

    if _llm_instance is not None and _loaded_repo_id != repo_id:
        logger.info(
            "Modello LLM cambiato (%s -> %s): il vecchio singleton viene scartato.",
            _loaded_repo_id,
            repo_id,
        )
        _llm_instance = None
        _loaded_repo_id = None

    if _llm_instance is not None:
        return _llm_instance

    if not _LLAMA_AVAILABLE:
        raise ImportError(
            "llama-cpp-python non e' installato. "
            "Installarlo con: pip install llama-cpp-python"
        )

    # Cerca prima in cache, altrimenti scarica
    model_path = get_model_path()
    if model_path is None:
        model_path = _download_model(repo_id)

    try:
        from langchain_community.llms import LlamaCpp
    except ImportError as exc:
        raise ImportError(
            "langchain-community non e' installato. "
            "Installarlo con: pip install langchain-community"
        ) from exc

    logger.info("Caricamento modello LLM '%s' da: %s", repo_id, model_path)
    _llm_instance = LlamaCpp(
        model_path=model_path,
        n_ctx=4096,
        temperature=0.7,
        max_tokens=512,
        n_threads=4,
        verbose=False,
    )
    _loaded_repo_id = repo_id
    logger.info("LLM '%s' pronto.", repo_id)
    return _llm_instance
