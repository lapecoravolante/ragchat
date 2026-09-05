"""Pannello sinistro — gestione DB FAISS (apertura, creazione, ingestione)."""

import logging
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

logger = logging.getLogger(__name__)


class DBPanel(tk.Frame):
    """Widget Tkinter per la gestione del database vettoriale FAISS.

    Pannello verticale con:
    - Sezione percorso DB + pulsanti Apri / Nuovo
    - Listbox documenti indicizzati con scrollbar
    - Pulsanti Aggiungi / Rimuovi
    - Label di stato in basso
    """

    def __init__(self, parent, on_db_changed: callable = None, **kwargs):
        """Costruisce il pannello.

        Args:
            parent: Widget Tkinter padre.
            on_db_changed: Callback ``on_db_changed(faiss_store: FAISSStore)``
                chiamato ogni volta che il DB attivo cambia.
        """
        super().__init__(parent, **kwargs)
        self._on_db_changed = on_db_changed
        self._store = None  # FAISSStore corrente

        self._build_ui()

    # ------------------------------------------------------------------
    # Costruzione UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Costruisce l'intera interfaccia del pannello."""
        self.columnconfigure(0, weight=1)

        # ── Titolo ───────────────────────────────────────────────────
        title_lbl = tk.Label(
            self,
            text="🗄 Database Vettoriale",
            font=("", 11, "bold"),
            anchor="w",
        )
        title_lbl.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        ttk.Separator(self, orient="horizontal").grid(
            row=1, column=0, sticky="ew", padx=4
        )

        # ── Sezione percorso DB ──────────────────────────────────────
        path_frame = tk.Frame(self)
        path_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(6, 2))
        path_frame.columnconfigure(0, weight=1)

        tk.Label(path_frame, text="Percorso DB:", anchor="w").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )

        self._path_var = tk.StringVar(value="(nessun DB aperto)")
        self._path_lbl = tk.Label(
            path_frame,
            textvariable=self._path_var,
            anchor="w",
            wraplength=200,
            justify="left",
            foreground="#57606a",
        )
        self._path_lbl.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 4))

        btn_frame_db = tk.Frame(path_frame)
        btn_frame_db.grid(row=2, column=0, columnspan=2, sticky="w")

        self._btn_open = ttk.Button(
            btn_frame_db, text="Apri DB", command=self._on_open_db
        )
        self._btn_open.pack(side="left", padx=(0, 4))

        self._btn_new = ttk.Button(
            btn_frame_db, text="Nuovo DB", command=self._on_new_db
        )
        self._btn_new.pack(side="left")

        ttk.Separator(self, orient="horizontal").grid(
            row=3, column=0, sticky="ew", padx=4, pady=(6, 0)
        )

        # ── Sezione documenti ────────────────────────────────────────
        docs_frame = tk.Frame(self)
        docs_frame.grid(row=4, column=0, sticky="nsew", padx=8, pady=(4, 2))
        docs_frame.columnconfigure(0, weight=1)
        docs_frame.rowconfigure(1, weight=1)
        self.rowconfigure(4, weight=1)

        self._docs_count_var = tk.StringVar(value="Documenti (0):")
        tk.Label(docs_frame, textvariable=self._docs_count_var, anchor="w").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )

        self._listbox = tk.Listbox(docs_frame, selectmode="single", activestyle="none")
        self._listbox.grid(row=1, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            docs_frame, orient="vertical", command=self._listbox.yview
        )
        scrollbar.grid(row=1, column=1, sticky="ns")
        self._listbox.configure(yscrollcommand=scrollbar.set)

        btn_frame_docs = tk.Frame(docs_frame)
        btn_frame_docs.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self._btn_add = ttk.Button(
            btn_frame_docs, text="Aggiungi", command=self._on_add_documents
        )
        self._btn_add.pack(side="left", padx=(0, 4))

        self._btn_remove = ttk.Button(
            btn_frame_docs, text="Rimuovi sel.", command=self._on_remove_document
        )
        self._btn_remove.pack(side="left")

        ttk.Separator(self, orient="horizontal").grid(
            row=5, column=0, sticky="ew", padx=4, pady=(4, 0)
        )

        # ── Label di stato ───────────────────────────────────────────
        self._status_var = tk.StringVar(value="● Pronto")
        self._status_lbl = tk.Label(
            self,
            textvariable=self._status_var,
            anchor="w",
            foreground="#57606a",
        )
        self._status_lbl.grid(row=6, column=0, sticky="ew", padx=8, pady=(4, 8))

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def set_store(self, store) -> None:
        """Aggiorna il pannello con un nuovo FAISSStore.

        Args:
            store: Istanza ``FAISSStore`` da rendere attiva.
        """
        self._store = store
        self._path_var.set(store.db_path if store else "(nessun DB aperto)")
        self.refresh_documents()

    def refresh_documents(self) -> None:
        """Aggiorna la Listbox con i documenti del store corrente."""
        self._listbox.delete(0, tk.END)
        if self._store is None:
            self._docs_count_var.set("Documenti (0):")
            return
        docs = self._store.list_documents()
        self._docs_count_var.set(f"Documenti ({len(docs)}):")
        for doc in docs:
            self._listbox.insert(tk.END, doc["filename"])

    # ------------------------------------------------------------------
    # Handler pulsanti
    # ------------------------------------------------------------------

    def _on_open_db(self) -> None:
        """Apre un DB FAISS esistente scelto dall'utente."""
        path = filedialog.askdirectory(title="Seleziona cartella DB FAISS")
        if not path:
            return

        self._set_buttons_state("disabled")
        self._status_var.set("⏳ Caricamento DB...")

        def _worker():
            try:
                from ragchat.core.vectorstore import FAISSStore

                store = FAISSStore.load(path)
                self.after(0, lambda: self._finish_db_change(store))
            except Exception as exc:
                logger.exception("Errore apertura DB: %s", exc)
                self.after(
                    0,
                    lambda: self._set_status(f"✗ Errore: {exc}", restore_buttons=True),
                )

        threading.Thread(target=_worker, daemon=True).start()

    def _on_new_db(self) -> None:
        """Crea un nuovo DB FAISS nella cartella scelta dall'utente."""
        path = filedialog.askdirectory(title="Seleziona cartella per nuovo DB")
        if not path:
            return

        self._set_buttons_state("disabled")
        self._status_var.set("⏳ Creazione DB...")

        def _worker():
            try:
                from ragchat.core.vectorstore import FAISSStore

                store = FAISSStore.create(path)
                self.after(0, lambda: self._finish_db_change(store))
            except Exception as exc:
                logger.exception("Errore creazione DB: %s", exc)
                self.after(
                    0,
                    lambda: self._set_status(f"✗ Errore: {exc}", restore_buttons=True),
                )

        threading.Thread(target=_worker, daemon=True).start()

    def _on_add_documents(self) -> None:
        """Aggiunge uno o più documenti al DB corrente."""
        if self._store is None:
            messagebox.showwarning("Nessun DB", "Aprire o creare un DB prima.")
            return

        from ragchat.core.ingestor import get_supported_extensions

        exts = get_supported_extensions()
        pattern = " ".join(f"*{e}" for e in exts)
        filetypes = [("Documenti", pattern), ("Tutti i file", "*.*")]

        paths = filedialog.askopenfilenames(
            title="Seleziona documenti da indicizzare", filetypes=filetypes
        )
        if not paths:
            return

        self._set_buttons_state("disabled")
        self._status_var.set("⏳ Indicizzazione in corso...")

        def _worker():
            from ragchat.core.ingestor import ingest_document

            total = 0
            errors = 0
            for file_path in paths:
                try:
                    n = ingest_document(file_path, self._store)
                    total += n
                except Exception as exc:
                    logger.exception("Errore ingestione '%s': %s", file_path, exc)
                    errors += 1

            msg = f"✔ {total} chunk indicizzati"
            if errors:
                msg += f" ({errors} errori)"
            self.after(0, lambda: self._finish_ingest(msg))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_remove_document(self) -> None:
        """Rimuove il documento selezionato dalla Listbox."""
        if self._store is None:
            messagebox.showwarning("Nessun DB", "Aprire o creare un DB prima.")
            return

        selection = self._listbox.curselection()
        if not selection:
            messagebox.showinfo("Nessuna selezione", "Selezionare un documento.")
            return

        filename = self._listbox.get(selection[0])
        if not messagebox.askyesno(
            "Conferma", f"Rimuovere il documento '{filename}'?"
        ):
            return

        self._set_buttons_state("disabled")
        self._status_var.set(f"⏳ Rimozione '{filename}'...")

        def _worker():
            try:
                self._store.remove_document(filename)
                self.after(0, lambda: self._finish_remove(filename))
            except Exception as exc:
                logger.exception("Errore rimozione '%s': %s", filename, exc)
                self.after(
                    0,
                    lambda: self._set_status(f"✗ Errore: {exc}", restore_buttons=True),
                )

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Helpers interni
    # ------------------------------------------------------------------

    def _finish_db_change(self, store) -> None:
        """Chiamato nel thread GUI dopo apertura/creazione DB."""
        self._store = store
        self._path_var.set(store.db_path)
        self.refresh_documents()
        self._set_status("● Pronto", restore_buttons=True)
        if self._on_db_changed:
            try:
                self._on_db_changed(store)
            except Exception as exc:
                logger.exception("Errore in on_db_changed: %s", exc)

    def _finish_ingest(self, msg: str) -> None:
        """Chiamato nel thread GUI al termine dell'ingestione."""
        self.refresh_documents()
        self._set_status(msg, restore_buttons=True)

    def _finish_remove(self, filename: str) -> None:
        """Chiamato nel thread GUI dopo la rimozione di un documento."""
        self.refresh_documents()
        self._set_status(f"✔ '{filename}' rimosso", restore_buttons=True)

    def _set_status(self, msg: str, *, restore_buttons: bool = False) -> None:
        """Aggiorna la label di stato."""
        self._status_var.set(msg)
        if restore_buttons:
            self._set_buttons_state("normal")

    def _set_buttons_state(self, state: str) -> None:
        """Abilita o disabilita tutti i pulsanti del pannello."""
        for btn in (self._btn_open, self._btn_new, self._btn_add, self._btn_remove):
            btn.configure(state=state)
