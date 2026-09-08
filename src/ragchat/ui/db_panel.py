"""Pannello sinistro — gestione DB FAISS (apertura, creazione, ingestione)."""

import logging
import queue
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

    I lavori potenzialmente bloccanti vengono eseguiti in thread
    separati. I risultati vengono trasferiti al thread GUI tramite
    queue.Queue, evitando chiamate Tkinter dai thread worker.
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

        # Coda utilizzata dai thread worker per comunicare con
        # il thread principale Tkinter.
        self._ui_queue = queue.Queue()

        self._build_ui()

        # Il polling della coda viene eseguito esclusivamente
        # dal thread GUI.
        self.after(50, self._process_ui_queue)

        logger.info("DBPanel inizializzato")

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
        path_frame.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=8,
            pady=(6, 2),
        )
        path_frame.columnconfigure(0, weight=1)

        tk.Label(path_frame, text="Percorso DB:", anchor="w",).grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="w",
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
        self._path_lbl.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(0, 4),
        )

        btn_frame_db = tk.Frame(path_frame)
        btn_frame_db.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="w",
        )

        self._btn_open = ttk.Button(
            btn_frame_db,
            text="Apri DB",
            command=self._on_open_db,
        )
        self._btn_open.pack(side="left", padx=(0, 4))

        self._btn_new = ttk.Button(
            btn_frame_db,
            text="Nuovo DB",
            command=self._on_new_db,
        )
        self._btn_new.pack(side="left")

        ttk.Separator(self, orient="horizontal").grid(
            row=3,
            column=0,
            sticky="ew",
            padx=4,
            pady=(6, 0),
        )

        # ── Sezione documenti ────────────────────────────────────────
        docs_frame = tk.Frame(self)
        docs_frame.grid(
            row=4,
            column=0,
            sticky="nsew",
            padx=8,
            pady=(4, 2),
        )

        docs_frame.columnconfigure(0, weight=1)
        docs_frame.rowconfigure(1, weight=1)

        self.rowconfigure(4, weight=1)

        self._docs_count_var = tk.StringVar(value="Documenti (0):")

        tk.Label(docs_frame, textvariable=self._docs_count_var, anchor="w",).grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="w",
        )

        self._listbox = tk.Listbox(
            docs_frame,
            selectmode="single",
            activestyle="none",
        )
        self._listbox.grid(
            row=1,
            column=0,
            sticky="nsew",
        )

        scrollbar = ttk.Scrollbar(
            docs_frame,
            orient="vertical",
            command=self._listbox.yview,
        )
        scrollbar.grid(
            row=1,
            column=1,
            sticky="ns",
        )

        self._listbox.configure(yscrollcommand=scrollbar.set)

        btn_frame_docs = tk.Frame(docs_frame)
        btn_frame_docs.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(4, 0),
        )

        self._btn_add = ttk.Button(
            btn_frame_docs,
            text="Aggiungi",
            command=self._on_add_documents,
        )
        self._btn_add.pack(side="left", padx=(0, 4))

        self._btn_remove = ttk.Button(
            btn_frame_docs,
            text="Rimuovi sel.",
            command=self._on_remove_document,
        )
        self._btn_remove.pack(side="left")

        ttk.Separator(self, orient="horizontal").grid(
            row=5,
            column=0,
            sticky="ew",
            padx=4,
            pady=(4, 0),
        )

        # ── Label di stato ───────────────────────────────────────────
        self._status_var = tk.StringVar(value="● Pronto")

        self._status_lbl = tk.Label(
            self,
            textvariable=self._status_var,
            anchor="w",
            foreground="#57606a",
        )
        self._status_lbl.grid(
            row=6,
            column=0,
            sticky="ew",
            padx=8,
            pady=(4, 8),
        )

    # ------------------------------------------------------------------
    # Gestione coda thread -> GUI
    # ------------------------------------------------------------------

    def _process_ui_queue(self) -> None:
        """Esegue nel thread GUI le callback prodotte dai worker."""
        try:
            while True:
                callback = self._ui_queue.get_nowait()

                try:
                    callback()
                except Exception:
                    logger.exception("Errore nell'esecuzione di una callback GUI")

        except queue.Empty:
            pass

        # Continua il polling della coda.
        try:
            self.after(50, self._process_ui_queue)
        except tk.TclError:
            # La finestra potrebbe essere già stata distrutta.
            pass

    def _run_on_ui(self, callback) -> None:
        """Accoda una callback da eseguire nel thread GUI."""
        self._ui_queue.put(callback)

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
            self._listbox.insert(
                tk.END,
                doc["filename"],
            )

    # ------------------------------------------------------------------
    # Handler pulsanti
    # ------------------------------------------------------------------

    def _on_open_db(self) -> None:
        """Apre un DB FAISS esistente scelto dall'utente."""
        path = filedialog.askdirectory(title="Seleziona cartella DB FAISS")

        if not path:
            return

        logger.info(
            "Richiesta apertura DB: %s",
            path,
        )

        self._set_buttons_state("disabled")
        self._status_var.set("⏳ Caricamento DB...")

        def _worker():
            logger.info(
                "Worker apertura DB avviato: %s",
                path,
            )

            try:
                from ragchat.core.vectorstore import FAISSStore

                logger.info(
                    "Chiamata FAISSStore.load(%s)",
                    path,
                )

                store = FAISSStore.load(path)

                logger.info(
                    "FAISSStore.load() terminato: %r",
                    store,
                )

                self._run_on_ui(lambda: self._finish_db_change(store))

                logger.info("Risultato apertura DB accodato alla GUI")

            except Exception as exc:
                logger.exception(
                    "Errore apertura DB '%s'",
                    path,
                )

                self._run_on_ui(
                    lambda exc=exc: self._set_status(
                        f"✗ Errore: {exc}",
                        restore_buttons=True,
                    )
                )

        threading.Thread(
            target=_worker,
            daemon=True,
            name="load-faiss-db",
        ).start()

    def _on_new_db(self) -> None:
        """Crea un nuovo DB FAISS nella cartella scelta dall'utente."""
        path = filedialog.askdirectory(title="Seleziona cartella per nuovo DB")

        if not path:
            return

        logger.info(
            "Richiesta creazione nuovo DB: %s",
            path,
        )

        self._set_buttons_state("disabled")
        self._status_var.set("⏳ Creazione DB...")

        def _worker():
            logger.info(
                "Worker creazione DB avviato: %s",
                path,
            )

            try:
                logger.info("Importazione FAISSStore...")

                from ragchat.core.vectorstore import FAISSStore

                logger.info(
                    "FAISSStore importato. " "Chiamata FAISSStore.create(%s)",
                    path,
                )

                store = FAISSStore.create(path)

                logger.info(
                    "FAISSStore.create() terminato: %r",
                    store,
                )

                self._run_on_ui(lambda: self._finish_db_change(store))

                logger.info("Risultato creazione DB accodato alla GUI")

            except Exception as exc:
                logger.exception(
                    "Errore creazione DB '%s'",
                    path,
                )

                self._run_on_ui(
                    lambda exc=exc: self._set_status(
                        f"✗ Errore: {exc}",
                        restore_buttons=True,
                    )
                )

        threading.Thread(
            target=_worker,
            daemon=True,
            name="create-faiss-db",
        ).start()

    def _on_add_documents(self) -> None:
        """Aggiunge uno o più documenti al DB corrente."""
        if self._store is None:
            messagebox.showwarning(
                "Nessun DB",
                "Aprire o creare un DB prima.",
            )
            return

        from ragchat.core.ingestor import (
            get_supported_extensions,
        )

        exts = get_supported_extensions()
        
        # 1. Genera l'elenco flat di tutti i pattern per il filtro cumulativo
        all_patterns = []
        category_filters = []
        
        for category_name, extensions in exts.items():
            # Pulisce le estensioni rimuovendo il punto se già presente e aggiungendo '*.'
            patterns = [f"*.{ext}" for ext in extensions]
            all_patterns.extend(patterns)
            
            # Crea il filtro specifico per questa categoria (es. ("Formati PDF", "*.pdf"))
            patterns_string = " ".join(patterns)
            category_filters.append((f"{category_name}", patterns_string))

        # 2. Popola la lista filetypes combinando i filtri creati
        filetypes = [
            ("Tutti i file", "*.*"),
            *category_filters,
            ("Tutti i file supportati da Docling", " ".join(all_patterns))
        ]

        paths = filedialog.askopenfilenames(
            title="Seleziona documenti da indicizzare",
            filetypes=filetypes,
        )

        if not paths:
            return

        logger.info(
            "Richiesta ingestione di %d documento/i",
            len(paths),
        )

        self._set_buttons_state("disabled")
        self._status_var.set("⏳ Indicizzazione in corso...")

        # Catturiamo il riferimento al DB corrente prima
        # di avviare il thread.
        store = self._store

        def _worker():
            from ragchat.core.ingestor import ingest_document

            logger.info("Worker ingestione avviato")

            total = 0
            errors = 0

            for file_path in paths:
                try:
                    logger.info(
                        "Indicizzazione documento: %s",
                        file_path,
                    )

                    n = ingest_document(
                        file_path,
                        store,
                    )

                    total += n

                    logger.info(
                        "Documento indicizzato: %s (%d chunk)",
                        file_path,
                        n,
                    )

                except Exception as exc:
                    logger.exception(
                        "Errore ingestione '%s': %s",
                        file_path,
                        exc,
                    )
                    errors += 1

            msg = f"✔ {total} chunk indicizzati"

            if errors:
                msg += f" ({errors} errori)"

            logger.info(
                "Worker ingestione terminato: %s",
                msg,
            )

            self._run_on_ui(lambda msg=msg: self._finish_ingest(msg))

        threading.Thread(
            target=_worker,
            daemon=True,
            name="ingest-documents",
        ).start()

    def _on_remove_document(self) -> None:
        """Rimuove il documento selezionato dalla Listbox."""
        if self._store is None:
            messagebox.showwarning(
                "Nessun DB",
                "Aprire o creare un DB prima.",
            )
            return

        selection = self._listbox.curselection()

        if not selection:
            messagebox.showinfo(
                "Nessuna selezione",
                "Selezionare un documento.",
            )
            return

        filename = self._listbox.get(selection[0])

        if not messagebox.askyesno(
            "Conferma",
            f"Rimuovere il documento '{filename}'?",
        ):
            return

        logger.info(
            "Richiesta rimozione documento: %s",
            filename,
        )

        self._set_buttons_state("disabled")
        self._status_var.set(f"⏳ Rimozione '{filename}'...")

        store = self._store

        def _worker():
            logger.info(
                "Worker rimozione avviato: %s",
                filename,
            )

            try:
                store.remove_document(filename)

                logger.info(
                    "Documento rimosso: %s",
                    filename,
                )

                self._run_on_ui(lambda: self._finish_remove(filename))

            except Exception as exc:
                logger.exception(
                    "Errore rimozione '%s': %s",
                    filename,
                    exc,
                )

                self._run_on_ui(
                    lambda exc=exc: self._set_status(
                        f"✗ Errore: {exc}",
                        restore_buttons=True,
                    )
                )

        threading.Thread(
            target=_worker,
            daemon=True,
            name="remove-document",
        ).start()

    # ------------------------------------------------------------------
    # Helpers interni
    # ------------------------------------------------------------------

    def _finish_db_change(self, store) -> None:
        """Chiamato esclusivamente nel thread GUI dopo apertura/creazione DB."""
        logger.info(">>> _finish_db_change() INIZIO")

        self._store = store

        logger.info(">>> _store aggiornato")

        self._path_var.set(store.db_path)

        logger.info(
            ">>> path aggiornato: %s",
            store.db_path,
        )

        self.refresh_documents()

        logger.info(">>> refresh_documents() terminato")

        self._set_status(
            "● Pronto",
            restore_buttons=True,
        )

        logger.info(">>> stato aggiornato e pulsanti riabilitati")

        if self._on_db_changed:
            try:
                logger.info(">>> chiamata on_db_changed()")

                self._on_db_changed(store)

                logger.info(">>> on_db_changed() terminata")

            except Exception as exc:
                logger.exception(
                    "Errore in on_db_changed: %s",
                    exc,
                )

        logger.info(">>> _finish_db_change() FINE")

    def _finish_ingest(self, msg: str) -> None:
        """Chiamato nel thread GUI al termine dell'ingestione."""
        logger.info("Completamento ingestione nel thread GUI")

        self.refresh_documents()
        self._set_status(
            msg,
            restore_buttons=True,
        )

    def _finish_remove(self, filename: str) -> None:
        """Chiamato nel thread GUI dopo la rimozione di un documento."""
        logger.info(
            "Completamento rimozione nel thread GUI: %s",
            filename,
        )

        self.refresh_documents()

        self._set_status(
            f"✔ '{filename}' rimosso",
            restore_buttons=True,
        )

    def _set_status(
        self,
        msg: str,
        *,
        restore_buttons: bool = False,
    ) -> None:
        """Aggiorna la label di stato."""
        self._status_var.set(msg)

        if restore_buttons:
            self._set_buttons_state("normal")

    def _set_buttons_state(self, state: str) -> None:
        """Abilita o disabilita tutti i pulsanti del pannello."""
        for btn in (
            self._btn_open,
            self._btn_new,
            self._btn_add,
            self._btn_remove,
        ):
            btn.configure(state=state)
