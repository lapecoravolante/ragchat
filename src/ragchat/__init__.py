"""Package principale dell'applicazione RAG Chat.

All'importazione viene eseguito il preload delle DLL VC++ incluse nel
vendor (solo Windows), in modo che ``llama_cpp`` possa trovare le
dipendenze native prima di essere importato.

Entry point:
    :func:`main` — avvia la finestra principale Tkinter.
"""

import ctypes
import logging
import os
import sys


def _preload_vendor_dlls() -> None:
    """Pre-carica le DLL del runtime VC++ incluse nel vendor.

    Carica esplicitamente tramite ``ctypes.CDLL`` le DLL
    ``VCRUNTIME140.dll``, ``VCRUNTIME140_1.dll`` e ``MSVCP140.dll``
    incluse in ``vendor/win_runtime/``, in modo che siano già
    disponibili nel loader di Windows prima che ``llama_cpp`` provi
    a caricare le sue DLL native (che ne dipendono).

    Non fa nulla su piattaforme non Windows o se la cartella
    ``vendor/win_runtime/`` non esiste.
    """
    if sys.platform != "win32":
        return
    vendor_dir = os.path.join(os.path.dirname(__file__), "vendor", "win_runtime")
    if not os.path.isdir(vendor_dir):
        return
    for dll in ("VCRUNTIME140.dll", "VCRUNTIME140_1.dll", "MSVCP140.dll"):
        dll_path = os.path.join(vendor_dir, dll)
        if os.path.isfile(dll_path):
            try:
                ctypes.CDLL(dll_path)
            except OSError:
                pass  # già caricata o non necessaria su questa macchina


_preload_vendor_dlls()

from ragchat.app import MainApp  # noqa: E402  (deve venire dopo il preload)


def main() -> None:
    """Entry point dell'applicazione.

    Configura il logging di base, crea l'istanza di
    :class:`~ragchat.app.MainApp` e avvia il loop eventi Tkinter.
    Viene invocato dallo script ``ragchat`` definito in ``pyproject.toml``.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Avvio applicazione RAG Chat")

    app = MainApp()
    app.mainloop()
