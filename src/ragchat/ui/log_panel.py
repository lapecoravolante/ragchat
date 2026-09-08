"""Pannello log — finestra Toplevel che mostra i messaggi di logging in tempo reale.

Utilizzo
--------
1. Creare un'istanza ``LogPanel(master)`` una volta sola (preferibilmente in ``MainApp``).
2. Chiamare ``LogPanel.install()`` per collegare il handler al root logger.
3. Collegare ``LogPanel.toggle()`` al click sulla label di stato desiderata.

Il pannello è singleton (una sola finestra alla volta): se viene chiuso e
poi riaperto conserva tutto il testo già accumulato.
"""

import logging
import queue
import tkinter as tk
from tkinter import scrolledtext


class _QueueHandler(logging.Handler):
    """Handler che deposita ogni LogRecord in una queue.Queue thread-safe."""

    def __init__(self, log_queue: queue.Queue) -> None:
        super().__init__()
        self._queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        self._queue.put_nowait(self.format(record))


class LogPanel:
    """Finestra Toplevel che mostra i log dell'applicazione in tempo reale.

    La finestra può essere mostrata/nascosta tramite :meth:`toggle`.
    Il testo accumulato non viene perso quando la finestra è nascosta.
    """

    _POLL_MS = 100  # intervallo polling coda log (ms)

    def __init__(self, master: tk.Misc) -> None:
        self._master = master
        self._queue: queue.Queue[str] = queue.Queue()
        self._handler = _QueueHandler(self._queue)
        self._handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        self._win: tk.Toplevel | None = None
        self._text: scrolledtext.ScrolledText | None = None
        # Buffer dei messaggi arrivati prima che la finestra fosse creata
        self._buffer: list[str] = []

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def install(self) -> None:
        """Collega il handler al root logger."""
        logging.getLogger().addHandler(self._handler)

    def toggle(self) -> None:
        """Mostra la finestra se è nascosta/inesistente, la chiude se è aperta."""
        if self._win is None or not self._win.winfo_exists():
            self._open()
        else:
            self._win.destroy()
            self._win = None
            self._text = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _open(self) -> None:
        """Crea e mostra la finestra Toplevel."""
        self._win = tk.Toplevel(self._master)
        self._win.title("Log applicazione")
        self._win.geometry("800x400")
        self._win.resizable(True, True)
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

        # Barra pulsanti
        btn_bar = tk.Frame(self._win)
        btn_bar.pack(fill=tk.X, padx=4, pady=(4, 0))

        tk.Button(
            btn_bar,
            text="Pulisci",
            command=self._clear,
        ).pack(side=tk.LEFT, padx=(0, 4))

        # Area testo
        self._text = scrolledtext.ScrolledText(
            self._win,
            state=tk.DISABLED,
            wrap=tk.NONE,
            font=("Courier New", 9),
            background="#1e1e1e",
            foreground="#d4d4d4",
            insertbackground="#d4d4d4",
        )
        self._text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Tag colori per livello
        self._text.tag_config("ERROR",   foreground="#f44747")
        self._text.tag_config("WARNING", foreground="#ce9178")
        self._text.tag_config("INFO",    foreground="#9cdcfe")
        self._text.tag_config("DEBUG",   foreground="#6a9955")

        # Scarica i messaggi bufferizzati
        for msg in self._buffer:
            self._append(msg)

        # Avvia il polling della coda
        self._poll()

    def _on_close(self) -> None:
        if self._win is not None:
            self._win.destroy()
        self._win = None
        self._text = None

    def _clear(self) -> None:
        if self._text is not None:
            self._text.config(state=tk.NORMAL)
            self._text.delete("1.0", tk.END)
            self._text.config(state=tk.DISABLED)
        self._buffer.clear()

    def _poll(self) -> None:
        """Drena la coda e aggiorna il widget testo; si ripianifica da sola."""
        if self._win is None or not self._win.winfo_exists():
            return
        try:
            while True:
                msg = self._queue.get_nowait()
                self._buffer.append(msg)
                self._append(msg)
        except queue.Empty:
            pass
        self._win.after(self._POLL_MS, self._poll)

    def _append(self, msg: str) -> None:
        if self._text is None:
            return
        # Determina il tag dal livello nel messaggio formattato
        tag = ""
        for level in ("ERROR", "WARNING", "INFO", "DEBUG"):
            if f"[{level}]" in msg:
                tag = level
                break
        self._text.config(state=tk.NORMAL)
        if tag:
            self._text.insert(tk.END, msg + "\n", tag)
        else:
            self._text.insert(tk.END, msg + "\n")
        self._text.config(state=tk.DISABLED)
        self._text.see(tk.END)
