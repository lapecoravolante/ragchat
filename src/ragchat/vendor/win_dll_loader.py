"""Caricamento automatico delle DLL Windows necessarie all'avvio.

Registra tramite ``os.add_dll_directory``:
1. La cartella ``vendor/win_runtime`` bundled nel progetto (MSVCP140, VCRUNTIME140, ...)
   -- permette di usare l'app senza installare il VC++ Redistributable di sistema.
2. La cartella ``lib/`` del pacchetto llama_cpp (ggml-*.dll, llama.dll, ...).

L'ordine e' importante: le VC++ runtime vengono registrate per prime cosi'
le DLL di llama_cpp le trovano gia' disponibili nel loader di Windows.

La funzione e' idempotente: puo' essere chiamata piu' volte senza effetti
collaterali (usa un flag di modulo per evitare registrazioni doppie).
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Percorso della cartella win_runtime bundled accanto a questo file
_VENDOR_RUNTIME_DIR = Path(__file__).parent / "win_runtime"

# Flag per evitare registrazioni duplicate in caso di import multipli
_dll_dirs_registered = False


def register_windows_dll_dirs() -> None:
    """Registra le cartelle DLL necessarie su Windows.

    Su sistemi non Windows (Linux, macOS) la funzione non fa nulla.
    L'operazione e' idempotente: ulteriori chiamate dopo la prima sono no-op.
    """
    global _dll_dirs_registered

    if _dll_dirs_registered:
        return

    _dll_dirs_registered = True  # imposta prima, cosi' eventuali errori non causano loop

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
