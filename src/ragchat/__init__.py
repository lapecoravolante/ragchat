import ctypes
import logging
import os
import sys


def _preload_vendor_dlls() -> None:
    """Pre-carica le DLL del runtime VC++ incluse nel vendor per sistemi
    che non hanno il Visual C++ Redistributable installato."""
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Avvio applicazione RAG Chat")

    app = MainApp()
    app.mainloop()
