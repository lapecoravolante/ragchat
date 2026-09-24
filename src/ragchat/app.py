import logging
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from ragchat.utils.config import Config
from ragchat.ui.db_panel import DBPanel
from ragchat.ui.chat_panel import ChatPanel
from ragchat.ui.log_panel import LogPanel

logger = logging.getLogger(__name__)


def _setup_logging(level_name: str) -> None:
    """Configura il livello di logging radice in base al valore da config."""
    level = getattr(logging, level_name.upper(), logging.ERROR)
    logging.getLogger().setLevel(level)


class MainApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        # ── Carica configurazione e imposta il logging ───────────────────
        self._config = Config.load()
        _setup_logging(self._config["log_level"])

        self.title("RAG Chat")
        self.geometry("1200x700")
        self.resizable(True, True)
        self.minsize(800, 500)

        # Tema nativo della piattaforma
        style = ttk.Style(self)
        try:
            style.theme_use("vista")  # Windows
        except tk.TclError:
            try:
                style.theme_use("aqua")  # macOS
            except tk.TclError:
                style.theme_use("clam")  # Linux/fallback

        # ── Pannello log (singleton) ─────────────────────────────────────
        self._log_panel = LogPanel(self)
        self._log_panel.install()

        # Layout principale: pannello sinistro (DB) + pannello destro (Chat)
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)

        self._db_panel = DBPanel(
            paned,
            on_db_changed=self._on_db_changed,
            log_panel=self._log_panel,
        )
        self._chat_panel = ChatPanel(paned)

        paned.add(self._db_panel, weight=1)
        paned.add(self._chat_panel, weight=3)
        paned.pack(fill=tk.BOTH, expand=True)

        # Carica DB di default se specificato in config, altrimenti fallback storico
        self._load_default_db()

        # Handler chiusura finestra
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_db_changed(self, faiss_store) -> None:
        from ragchat.core.rag import RAGChain

        rag_chain = RAGChain(faiss_store)
        self._chat_panel.set_rag_chain(rag_chain)
        self._db_panel.set_store(faiss_store)
        logger.info("DB aggiornato: RAGChain ricreata.")

    def _load_default_db(self) -> None:
        # Legge il path dalla configurazione; se vuoto usa il fallback storico
        config_path = self._config["default_db_path"].strip()
        if config_path:
            default_path = Path(config_path)
        else:
            default_path = Path.home() / "faiss_db"

        if default_path.exists():
            try:
                from ragchat.core.vectorstore import FAISSStore

                # Verifica modello embedding prima di caricare
                if not self._db_panel._verify_embedding_model(str(default_path)):
                    return

                store = FAISSStore.load(str(default_path))
                self._on_db_changed(store)
                logger.info("DB di default caricato da %s", default_path)
            except Exception as exc:
                logger.warning(
                    "Impossibile caricare il DB di default '%s': %s",
                    default_path,
                    exc,
                )

    def _on_close(self) -> None:
        self.destroy()
