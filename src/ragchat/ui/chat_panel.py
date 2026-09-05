import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import logging

logger = logging.getLogger(__name__)


class ChatPanel(tk.Frame):
    """Pannello destro: area di chat con il LLM tramite RAG."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._rag_chain = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Titolo
        title = tk.Label(self, text="💬 Chat", font=(None, 12, "bold"))
        title.grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))

        # Area cronologia
        self._history = scrolledtext.ScrolledText(
            self,
            state=tk.DISABLED,
            wrap=tk.WORD,
            background="#f5f5f5",
            relief=tk.FLAT,
            font=(None, 11),
        )
        self._history.tag_config("user", foreground="#1a3a5c", font=(None, 11, "bold"))
        self._history.tag_config("assistant", foreground="#1a4a2a", font=(None, 11))
        self._history.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)

        # Label stato (nascosta di default)
        self._status_var = tk.StringVar(value="")
        self._status_label = tk.Label(
            self, textvariable=self._status_var, anchor="w", foreground="#888888"
        )
        self._status_label.grid(row=2, column=0, sticky="ew", padx=8)
        self._status_label.grid_remove()

        # Frame input
        input_frame = tk.Frame(self)
        input_frame.grid(row=3, column=0, sticky="ew", padx=8, pady=(4, 8))
        input_frame.columnconfigure(0, weight=1)

        self._input_var = tk.StringVar()
        self._entry = ttk.Entry(input_frame, textvariable=self._input_var)
        self._entry.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._entry.bind("<Return>", lambda _e: self._on_send())

        self._send_btn = ttk.Button(input_frame, text="Invia", command=self._on_send)
        self._send_btn.grid(row=0, column=1)

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def set_rag_chain(self, chain):
        """Aggiorna il riferimento alla RAGChain (chiamato quando cambia il DB)."""
        self._rag_chain = chain

    # ------------------------------------------------------------------
    # Invio messaggio
    # ------------------------------------------------------------------

    def _on_send(self):
        question = self._input_var.get().strip()
        if not question:
            return

        self._input_var.set("")

        if self._rag_chain is None:
            self._append(
                "Nessun DB caricato. Apri un DB nel pannello a sinistra.\n\n",
                tag="assistant",
            )
            return

        self._append(f"Tu: {question}\n", tag="user")
        self._set_busy(True)

        threading.Thread(target=self._run_llm, args=(question,), daemon=True).start()

    def _run_llm(self, question: str):
        try:
            answer = self._rag_chain.ask(question)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Errore durante la chiamata LLM")
            answer = f"[Errore] {exc}"
        self.after(0, self._on_llm_done, answer)

    def _on_llm_done(self, answer: str):
        self._append(f"Assistente: {answer}\n\n", tag="assistant")
        self._set_busy(False)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _append(self, text: str, tag: str = ""):
        self._history.config(state=tk.NORMAL)
        if tag:
            self._history.insert(tk.END, text, tag)
        else:
            self._history.insert(tk.END, text)
        self._history.config(state=tk.DISABLED)
        self._history.see(tk.END)

    def _set_busy(self, busy: bool):
        if busy:
            self._status_var.set("⏳ In elaborazione...")
            self._status_label.grid()
            self._send_btn.config(state=tk.DISABLED)
            self._entry.config(state=tk.DISABLED)
        else:
            self._status_label.grid_remove()
            self._status_var.set("")
            self._send_btn.config(state=tk.NORMAL)
            self._entry.config(state=tk.NORMAL)
