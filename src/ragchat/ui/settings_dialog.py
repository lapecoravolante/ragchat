"""Finestra di dialogo per la modifica delle impostazioni dell'applicazione.

Apre una toplevel modale con un form che rispecchia le chiavi del file
``config.json``.  Le modifiche vengono persistite su disco solo al
clic di "Salva".
"""

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Optional

from ragchat.utils.config import Config, model_entry_id, model_entry_tags

logger = logging.getLogger(__name__)

# Livelli di log disponibili in Python
_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Etichette human-friendly per le chiavi di configurazione
_LABELS: dict[str, str] = {
    "embedding_model": "Modello Embedding attivo",
    "embedding_models": "Modelli Embedding disponibili",
    "query_model": "Modello LLM attivo",
    "query_models": "Modelli LLM disponibili",
    "top_k": "Top-K (numero risultati)",
    "log_level": "Livello di logging",
    "default_db_path": "Percorso DB di default",
    "prompt_template": "Testo del prompt",
}


class _Tooltip:
    """Tooltip che compare al passaggio del mouse su un widget."""

    def __init__(self, widget: tk.Widget, text_func) -> None:
        """
        Args:
            widget: il widget a cui associare il tooltip.
            text_func: callable che restituisce il testo corrente del tooltip.
        """
        self._widget = widget
        self._text_func = text_func
        self._tip_window: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)
        widget.bind("<Motion>", self._on_motion)

    def _show(self, event: tk.Event) -> None:
        """Mostra la finestra tooltip vicino al cursore."""
        text = self._text_func()
        if not text or self._tip_window:
            return
        x = event.x_root + 12
        y = event.y_root + 12
        self._tip_window = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(
            tw,
            text=text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=("", 9),
            wraplength=320,
        )
        lbl.pack(ipadx=4, ipady=2)

    def _hide(self, _event: tk.Event) -> None:
        """Distrugge la finestra tooltip."""
        if self._tip_window:
            self._tip_window.destroy()
            self._tip_window = None

    def _on_motion(self, event: tk.Event) -> None:
        """Aggiorna posizione tooltip durante il movimento del mouse."""
        if self._tip_window:
            self._tip_window.wm_geometry(f"+{event.x_root + 12}+{event.y_root + 12}")


class _DownloadProgressDialog(tk.Toplevel):
    """Finestra modale con barra di progresso per il download della lista modelli."""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self.title("Download lista modelli")
        self.resizable(False, False)
        self.grab_set()
        self.transient(parent)

        frame = tk.Frame(self, padx=20, pady=16)
        frame.pack(fill=tk.BOTH, expand=True)

        self._label_var = tk.StringVar(value="Connessione a HuggingFace...")
        tk.Label(frame, textvariable=self._label_var, anchor="w").pack(
            fill="x", pady=(0, 8)
        )

        self._progress = ttk.Progressbar(
            frame, orient="horizontal", length=360, mode="determinate"
        )
        self._progress.pack(fill="x")

        # Centra rispetto al parent
        self.update_idletasks()
        pw = parent.winfo_rootx() + parent.winfo_width() // 2
        ph = parent.winfo_rooty() + parent.winfo_height() // 2
        w, h = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{pw - w // 2}+{ph - h // 2}")

    def update_progress(self, current: int, total: int, message: str = "") -> None:
        """Aggiorna la barra di progresso e il messaggio."""
        if total > 0:
            self._progress["maximum"] = total
            self._progress["value"] = current
        if message:
            self._label_var.set(message)
        self.update_idletasks()


class _ModelListWidget(tk.Frame):
    """Widget compatto per gestire una lista di modelli con selezione del modello attivo.

    Struttura visiva::

        [Combobox modello attivo v]
        +--------------------------+
        | modello1                 |  <- Listbox con tooltip al passaggio del mouse
        | modello2  (*)            |
        | ...                      |
        +--------------------------+
        [↻ Aggiorna da HuggingFace] [- Rimuovi]

    I modelli sono dizionari con chiavi: id, url, format, tags.
    La combobox e la listbox mostrano il campo ``id``.
    Al passaggio del mouse su un elemento della listbox compare un tooltip
    con i tag del modello.

    Il modello selezionato nella combobox è quello che verrà salvato come
    ``<kind>_model``; la listbox mostra l'intera lista ``<kind>_models``.

    Args:
        parent: widget padre.
        models: lista iniziale di dizionari modello.
        active: ID del modello da selezionare come attivo.
        on_fetch: callback ``()`` invocata quando l'utente preme il pulsante
            "↻ Aggiorna da HuggingFace".  La logica di fetch (progress dialog,
            messagebox) è delegata al chiamante.
    """

    def __init__(
        self,
        parent: tk.Widget,
        models: list[dict[str, Any]],
        active: str,
        on_fetch: Optional[callable] = None,
    ) -> None:
        super().__init__(parent)
        self.columnconfigure(0, weight=1)

        self._models: list[dict[str, Any]] = [dict(m) for m in models]
        self._on_fetch_cb = on_fetch

        # Garantisce che il modello attivo sia in lista
        ids = [model_entry_id(m) for m in self._models]
        if active and active not in ids:
            self._models.insert(0, {"id": active, "url": active, "format": "safetensors", "tags": []})

        # ── Combobox modello attivo ──────────────────────────────────────
        self._active_var = tk.StringVar(value=active)
        self._combo = ttk.Combobox(
            self,
            textvariable=self._active_var,
            values=[model_entry_id(m) for m in self._models],
            state="readonly",
            width=50,
        )
        self._combo.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))

        # ── Listbox con scrollbar ────────────────────────────────────────
        list_frame = tk.Frame(self)
        list_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._listbox = tk.Listbox(list_frame, height=5, selectmode=tk.SINGLE)
        self._listbox.grid(row=0, column=0, sticky="nsew")

        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self._listbox.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self._listbox.configure(yscrollcommand=sb.set)

        # Tooltip sulla listbox basato sull'elemento sotto il cursore
        self._tooltip_win: Optional[tk.Toplevel] = None
        self._listbox.bind("<Motion>", self._on_listbox_motion)
        self._listbox.bind("<Leave>", self._on_listbox_leave)

        self._refresh_listbox()

        # ── Pulsanti Aggiorna / Aggiungi / Rimuovi ───────────────────────
        btn_frame = tk.Frame(self)
        btn_frame.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        ttk.Button(
            btn_frame,
            text="↻ Aggiorna da HuggingFace",
            command=self._on_fetch_clicked,
        ).pack(side="left", padx=(0, 4))
        ttk.Button(btn_frame, text="+ Aggiungi", command=self._on_add).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(btn_frame, text="- Rimuovi", command=self._on_remove).pack(
            side="left"
        )

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def get_models(self) -> list[dict[str, Any]]:
        """Restituisce la lista corrente dei modelli come lista di dizionari."""
        return [dict(m) for m in self._models]

    def get_active(self) -> str:
        """Restituisce l'ID del modello attivo selezionato."""
        return self._active_var.get()

    def replace_models(self, new_models: list[dict[str, Any]], active: str) -> str:
        """Sostituisce completamente la lista dei modelli con quella fornita.

        Se il modello *active* non è presente nella nuova lista, la selezione
        ricade sul primo modello disponibile.

        Args:
            new_models: nuova lista di dizionari modello.
            active: ID del modello da impostare come attivo.

        Returns:
            ID del modello effettivamente impostato come attivo.  Corrisponde
            ad *active* se il modello era nella nuova lista, altrimenti all'ID
            del primo modello della lista.
        """
        self._models = [dict(m) for m in new_models]
        ids = [model_entry_id(m) for m in self._models]
        if not self._models:
            active = ""
        elif active not in ids:
            active = model_entry_id(self._models[0])
        self._active_var.set(active)
        self._refresh_listbox()
        self._refresh_combo()
        return active

    # ------------------------------------------------------------------
    # Tooltip sulla listbox
    # ------------------------------------------------------------------

    def _get_tooltip_text_for_index(self, idx: int) -> str:
        if 0 <= idx < len(self._models):
            entry = self._models[idx]
            tags = model_entry_tags(entry)
            fmt = entry.get("format", "")
            url = entry.get("url", "")
            parts = []
            if fmt:
                parts.append(f"Formato: {fmt}")
            if url:
                parts.append(f"URL: {url}")
            if tags:
                parts.append("Tag: " + ", ".join(tags))
            return "\n".join(parts)
        return ""

    def _on_listbox_motion(self, event: tk.Event) -> None:
        idx = self._listbox.nearest(event.y)
        text = self._get_tooltip_text_for_index(idx)
        if not text:
            self._hide_tooltip()
            return
        x = event.x_root + 12
        y = event.y_root + 12
        if self._tooltip_win is None:
            self._tooltip_win = tw = tk.Toplevel(self._listbox)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{x}+{y}")
            self._tooltip_label = tk.Label(
                tw,
                text=text,
                justify="left",
                background="#ffffe0",
                relief="solid",
                borderwidth=1,
                font=("", 9),
                wraplength=360,
            )
            self._tooltip_label.pack(ipadx=4, ipady=2)
        else:
            self._tooltip_label.config(text=text)
            self._tooltip_win.wm_geometry(f"+{x}+{y}")

    def _on_listbox_leave(self, _event: tk.Event) -> None:
        """Nasconde il tooltip quando il cursore lascia la listbox."""
        self._hide_tooltip()

    def _hide_tooltip(self) -> None:
        """Distrugge la finestra tooltip della listbox se visibile."""
        if self._tooltip_win:
            self._tooltip_win.destroy()
            self._tooltip_win = None

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _on_fetch_clicked(self) -> None:
        """Invoca il callback di fetch registrato dal chiamante."""
        if self._on_fetch_cb:
            self._on_fetch_cb()

    def _on_add(self) -> None:
        """Apre _AddModelDialog per inserire un nuovo modello manualmente."""
        dialog = _AddModelDialog(self.winfo_toplevel())
        self.winfo_toplevel().wait_window(dialog)
        entry = dialog.result
        if entry is None:
            return
        mid = model_entry_id(entry)
        ids = [model_entry_id(m) for m in self._models]
        if mid and mid not in ids:
            self._models.append(entry)
            self._refresh_listbox()
            self._refresh_combo()

    def _on_remove(self) -> None:
        """Rimuove il modello selezionato nella listbox (non permette di rimuovere l'ultimo)."""
        sel = self._listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        model_id = model_entry_id(self._models[idx])

        if len(self._models) == 1:
            messagebox.showwarning(
                "Rimozione non consentita",
                "Deve essere presente almeno un modello.",
                parent=self.winfo_toplevel(),
            )
            return

        self._models.pop(idx)
        # Se il modello rimosso era quello attivo, passa al primo della lista
        if self._active_var.get() == model_id:
            self._active_var.set(model_entry_id(self._models[0]))
        self._refresh_listbox()
        self._refresh_combo()

    # ------------------------------------------------------------------
    # Helpers interni
    # ------------------------------------------------------------------

    def _refresh_listbox(self) -> None:
        """Ricarica il contenuto della listbox a partire da :attr:`_models`.

        Il modello attivo viene marcato con ``(*)`` nel testo visualizzato.
        I modelli vengono ordinati alfabeticamente per ID.
        """
        active = self._active_var.get()
        self._listbox.delete(0, tk.END)
        for m in sorted(self._models, key=lambda x: model_entry_id(x).lower()):
            mid = model_entry_id(m)
            label = f"{mid}  (*)" if mid == active else mid
            self._listbox.insert(tk.END, label)

    def _refresh_combo(self) -> None:
        """Aggiorna i valori della combobox con la lista corrente dei modelli.

        Se il valore attualmente selezionato non è più presente nella lista,
        seleziona automaticamente il primo modello disponibile.
        """
        ids = [model_entry_id(m) for m in self._models]
        self._combo["values"] = ids
        # Se il valore attuale non e' piu' valido, seleziona il primo
        if self._active_var.get() not in ids and ids:
            self._active_var.set(ids[0])


# Formati noti per la combobox nella dialog di aggiunta modello
_MODEL_FORMATS = ["gguf", "safetensors"]


class _AddModelDialog(tk.Toplevel):
    """Finestra modale per aggiungere manualmente un modello alla lista.

    Struttura visiva::

        ID modello (repo-id o URL):  [_________________________]
        Formato:                     [gguf          v]
        URL download:                [_________________________]
        Tag (virgola-separati):      [_________________________]
                    [Recupera da HF]   [Aggiungi]   [Chiudi]

    Il pulsante "Recupera da HF" compila automaticamente tutti i campi
    a partire dall'ID inserito interrogando l'API HuggingFace.
    Al termine, ``result`` contiene il dizionario modello oppure ``None``
    se l'utente ha annullato.
    """

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self.title("Aggiungi modello")
        self.resizable(False, False)
        self.grab_set()
        self.transient(parent)

        # Dizionario risultante; None finché l'utente non conferma
        self.result: Optional[dict[str, Any]] = None

        self._build_ui()

        self.update_idletasks()
        px = parent.winfo_rootx() + parent.winfo_width() // 2
        py = parent.winfo_rooty() + parent.winfo_height() // 2
        w, h = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{px - w // 2}+{py - h // 2}")

    # ------------------------------------------------------------------
    # Costruzione UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = tk.Frame(self, padx=14, pady=12)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(1, weight=1)

        row = 0

        # ── ID (repo-id o URL) ────────────────────────────────────────────
        tk.Label(outer, text="ID modello (repo-id o URL):", anchor="w").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=3
        )
        self._id_var = tk.StringVar()
        ttk.Entry(outer, textvariable=self._id_var, width=52).grid(
            row=row, column=1, sticky="ew", pady=3
        )
        row += 1

        # ── Formato ──────────────────────────────────────────────────────
        tk.Label(outer, text="Formato:", anchor="w").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=3
        )
        self._format_var = tk.StringVar(value="gguf")
        ttk.Combobox(
            outer,
            textvariable=self._format_var,
            values=_MODEL_FORMATS,
            state="readonly",
            width=16,
        ).grid(row=row, column=1, sticky="w", pady=3)
        row += 1

        # ── URL download ──────────────────────────────────────────────────
        tk.Label(outer, text="URL download:", anchor="w").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=3
        )
        self._url_var = tk.StringVar()
        ttk.Entry(outer, textvariable=self._url_var, width=52).grid(
            row=row, column=1, sticky="ew", pady=3
        )
        row += 1

        # ── Tag (virgola-separati) ────────────────────────────────────────
        tk.Label(outer, text="Tag (separati da virgola):", anchor="w").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=3
        )
        self._tags_var = tk.StringVar()
        ttk.Entry(outer, textvariable=self._tags_var, width=52).grid(
            row=row, column=1, sticky="ew", pady=3
        )
        row += 1

        # ── Separatore ───────────────────────────────────────────────────
        ttk.Separator(outer, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(10, 6)
        )
        row += 1

        # ── Pulsanti ─────────────────────────────────────────────────────
        btn_frame = tk.Frame(outer)
        btn_frame.grid(row=row, column=0, columnspan=2, sticky="e")

        ttk.Button(
            btn_frame, text="↻ Recupera da HF", command=self._on_fetch
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            btn_frame, text="Aggiungi", command=self._on_confirm
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            btn_frame, text="Chiudi", command=self.destroy
        ).pack(side="left")

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _on_fetch(self) -> None:
        """Recupera i metadati del modello da HuggingFace e compila i campi."""
        repo_id = self._id_var.get().strip()
        if not repo_id:
            messagebox.showwarning(
                "ID mancante",
                "Inserire un ID modello o un'URL HuggingFace prima di recuperare.",
                parent=self,
            )
            return

        # Disabilita i pulsanti durante il fetch
        for child in self.winfo_children():
            self._set_state(child, tk.DISABLED)

        result_holder: dict = {}

        def _do_fetch() -> None:
            try:
                from ragchat.core.huggingface import HuggingFace
                hf = HuggingFace()
                result_holder["data"] = hf.fetch_model_info(repo_id)
            except Exception as exc:  # noqa: BLE001
                result_holder["error"] = str(exc)
            finally:
                self.after(0, _on_done)

        def _on_done() -> None:
            # Riabilita i pulsanti
            for child in self.winfo_children():
                self._set_state(child, tk.NORMAL)

            if "error" in result_holder:
                messagebox.showerror(
                    "Errore recupero",
                    f"Impossibile recuperare le informazioni del modello:\n"
                    f"{result_holder['error']}",
                    parent=self,
                )
                return

            info = result_holder["data"]
            # Compila i campi con i dati recuperati
            self._id_var.set(info["id"])
            self._format_var.set(info.get("format", "safetensors"))
            self._url_var.set(info.get("url", ""))
            tags = info.get("tags", [])
            self._tags_var.set(", ".join(tags))

        threading.Thread(target=_do_fetch, daemon=True).start()

    def _on_confirm(self) -> None:
        """Valida i campi e chiude la finestra impostando ``result``."""
        mid = self._id_var.get().strip()
        if not mid:
            messagebox.showwarning(
                "ID mancante",
                "L'ID del modello non può essere vuoto.",
                parent=self,
            )
            return

        url = self._url_var.get().strip() or mid
        fmt = self._format_var.get().strip() or "safetensors"
        raw_tags = self._tags_var.get()
        tags = [t.strip() for t in raw_tags.split(",") if t.strip()]

        self.result = {
            "id": mid,
            "url": url,
            "format": fmt,
            "tags": tags,
        }
        self.destroy()

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _set_state(widget: tk.Widget, state) -> None:
        """Imposta ricorsivamente lo stato dei widget figli."""
        try:
            widget.configure(state=state)
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            _AddModelDialog._set_state(child, state)


class SettingsDialog(tk.Toplevel):
    """Finestra modale per la modifica di config.json."""

    def __init__(
        self,
        parent: tk.Widget,
        current_db_model: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.title("Impostazioni")
        self.resizable(True, True)
        self.grab_set()  # modale rispetto alla finestra principale

        # Modello embedding del DB correntemente caricato (se presente)
        self._current_db_model = current_db_model

        # Carica valori correnti
        self._values = Config.load()
        self._vars: dict[str, tk.Variable | tk.Text] = {}
        self._model_widgets: dict[str, _ModelListWidget] = {}

        self._build_ui()

        # Centra rispetto al parent
        self.update_idletasks()
        pw = parent.winfo_rootx() + parent.winfo_width() // 2
        ph = parent.winfo_rooty() + parent.winfo_height() // 2
        w, h = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{pw - w // 2}+{ph - h // 2}")

    # ------------------------------------------------------------------
    # Costruzione UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = tk.Frame(self, padx=12, pady=10)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(1, weight=1)

        row = 0

        # ── Modelli di Embedding ─────────────────────────────────────────
        row = self._add_model_list_row(outer, row, "embedding")

        # ── Modelli LLM ──────────────────────────────────────────────────
        row = self._add_model_list_row(outer, row, "query")

        # ── top_k ───────────────────────────────────────────────────────
        row = self._add_spinbox_row(outer, row, "top_k", from_=1, to=20)

        # ── log_level ───────────────────────────────────────────────────
        row = self._add_combobox_row(outer, row, "log_level", _LOG_LEVELS)

        # ── default_db_path ─────────────────────────────────────────────
        row = self._add_path_row(outer, row, "default_db_path")

        # ── prompt_template ─────────────────────────────────────────────
        row = self._add_text_area_row(outer, row, "prompt_template")

        # ── Separatore + pulsanti ────────────────────────────────────────
        ttk.Separator(outer, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(12, 6)
        )
        row += 1

        btn_frame = tk.Frame(outer)
        btn_frame.grid(row=row, column=0, columnspan=2, sticky="e")

        ttk.Button(btn_frame, text="Salva", command=self._on_save).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(btn_frame, text="Annulla", command=self.destroy).pack(side="left")

    # ------------------------------------------------------------------
    # Helper per le righe del form
    # ------------------------------------------------------------------

    def _label(self, parent: tk.Widget, row: int, key: str) -> None:
        """Inserisce una label nella colonna 0 del form per la chiave *key*.

        Args:
            parent: Widget contenitore (griglia).
            row:    Riga del grid in cui inserire la label.
            key:    Chiave di configurazione; il testo viene ricavato da
                    :data:`_LABELS` (fallback al nome della chiave).
        """
        tk.Label(parent, text=_LABELS.get(key, key) + ":", anchor="w").grid(
            row=row, column=0, sticky="nw", padx=(0, 10), pady=4
        )

    def _add_model_list_row(self, parent: tk.Widget, row: int, kind: str) -> int:
        """Aggiunge una riga con il widget _ModelListWidget per embedding o query."""
        list_key = f"{kind}_models"
        active_key = f"{kind}_model"
        label_text = "Modelli Embedding" if kind == "embedding" else "Modelli LLM"

        tk.Label(parent, text=label_text + ":", anchor="nw").grid(
            row=row, column=0, sticky="nw", padx=(0, 10), pady=4
        )

        models = self._values.get(list_key, [])
        active = self._values.get(active_key, model_entry_id(models[0]) if models else "")

        widget = _ModelListWidget(
            parent,
            models=models,
            active=active,
            on_fetch=lambda k=kind: self._on_fetch_models(k),
        )
        widget.grid(row=row, column=1, sticky="ew", pady=4)
        self._model_widgets[kind] = widget
        return row + 1

    def _add_spinbox_row(
        self, parent: tk.Widget, row: int, key: str, from_: int, to: int
    ) -> int:
        """Aggiunge una riga con una ``Spinbox`` al form.

        Args:
            parent: Widget contenitore (griglia).
            row:    Riga di inserimento.
            key:    Chiave di configurazione da leggere/scrivere.
            from_:  Valore minimo dello spinbox.
            to:     Valore massimo dello spinbox.

        Returns:
            Indice della riga successiva (``row + 1``).
        """
        self._label(parent, row, key)
        var = tk.StringVar(value=self._values.get(key, str(from_)))
        self._vars[key] = var
        ttk.Spinbox(parent, from_=from_, to=to, textvariable=var, width=6).grid(
            row=row, column=1, sticky="w", pady=4
        )
        return row + 1

    def _add_combobox_row(
        self, parent: tk.Widget, row: int, key: str, choices: list[str]
    ) -> int:
        """Aggiunge una riga con una ``Combobox`` (sola lettura) al form.

        Args:
            parent:  Widget contenitore (griglia).
            row:     Riga di inserimento.
            key:     Chiave di configurazione da leggere/scrivere.
            choices: Lista di valori ammessi nella combobox.

        Returns:
            Indice della riga successiva (``row + 1``).
        """
        self._label(parent, row, key)
        var = tk.StringVar(value=self._values.get(key, choices[0]))
        self._vars[key] = var
        cb = ttk.Combobox(
            parent, textvariable=var, values=choices, state="readonly", width=14
        )
        cb.grid(row=row, column=1, sticky="w", pady=4)
        return row + 1

    def _add_path_row(self, parent: tk.Widget, row: int, key: str) -> int:
        """Aggiunge una riga con campo testo e pulsante "Sfoglia..." per un percorso.

        Args:
            parent: Widget contenitore (griglia).
            row:    Riga di inserimento.
            key:    Chiave di configurazione (percorso cartella).

        Returns:
            Indice della riga successiva (``row + 1``).
        """
        self._label(parent, row, key)
        var = tk.StringVar(value=self._values.get(key, ""))
        self._vars[key] = var

        frame = tk.Frame(parent)
        frame.grid(row=row, column=1, sticky="ew", pady=4)
        frame.columnconfigure(0, weight=1)

        ttk.Entry(frame, textvariable=var).grid(row=0, column=0, sticky="ew")

        def _browse() -> None:
            current = var.get()
            initial = current if Path(current).is_dir() else str(Path.home())
            chosen = filedialog.askdirectory(
                parent=self,
                title="Seleziona cartella DB di default",
                initialdir=initial,
            )
            if chosen:
                var.set(chosen)

        ttk.Button(frame, text="Sfoglia...", command=_browse).grid(
            row=0, column=1, padx=(6, 0)
        )
        return row + 1

    def _add_text_area_row(self, parent: tk.Widget, row: int, key: str) -> int:
        """Aggiunge una riga con un'area di testo multiriga al form.

        Usata per la chiave ``prompt_template``.  Il widget ``tk.Text``
        viene memorizzato in :attr:`_vars` per essere letto in
        :meth:`_on_save`.

        Args:
            parent: Widget contenitore (griglia).
            row:    Riga di inserimento.
            key:    Chiave di configurazione (testo libero multiriga).

        Returns:
            Indice della riga successiva (``row + 1``).
        """
        self._label(parent, row, key)
        current = self._values.get(key, "")

        frame = tk.Frame(parent)
        frame.grid(row=row, column=1, sticky="ew", pady=4)
        frame.columnconfigure(0, weight=1)

        text = tk.Text(frame, height=8, wrap="word", font=("", 9))
        text.insert("1.0", current)
        text.grid(row=0, column=0, sticky="ew")

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=scrollbar.set)

        self._vars[key] = text  # type: ignore[assignment]
        return row + 1

    # ------------------------------------------------------------------
    # Download lista modelli da HuggingFace
    # ------------------------------------------------------------------

    def _on_fetch_models(self, kind: str) -> None:
        """Avvia il download della lista modelli per *kind* in un thread separato.

        Args:
            kind: ``"query"`` per i modelli LLM, ``"embedding"`` per gli embedding.
        """
        # Mappa kind → (pipeline_tag, label per messaggi)
        _KIND_META = {
            "query":     ("text-generation",   "LLM"),
            "embedding": ("feature-extraction", "Embedding"),
        }
        pipeline_tag, label = _KIND_META[kind]

        progress_dialog = _DownloadProgressDialog(self)
        progress_dialog.update_progress(0, 1, f"Recupero modelli {label}...")

        result_holder: dict = {}

        def _do_fetch() -> None:
            try:
                from ragchat.core.huggingface import HuggingFace

                def _on_progress(current: int, total: int) -> None:
                    self.after(
                        0,
                        lambda c=current, t=total: progress_dialog.update_progress(
                            c, t, f"Recupero modelli {label}..." if c < t else "Completato."
                        ),
                    )

                hf = HuggingFace()
                # Fetch solo la categoria richiesta
                if kind == "query":
                    models = hf._fetch_models_for_task(pipeline_tag, ["gguf", "safetensors"])
                else:
                    models = hf._fetch_models_for_task(pipeline_tag, ["safetensors"])
                result_holder["models"] = models
                _on_progress(1, 1)
            except Exception as exc:  # noqa: BLE001
                result_holder["error"] = str(exc)
            finally:
                self.after(0, _on_done)

        def _on_done() -> None:
            progress_dialog.destroy()

            if "error" in result_holder:
                messagebox.showerror(
                    "Errore download",
                    f"Impossibile scaricare la lista modelli {label}:\n{result_holder['error']}",
                    parent=self,
                )
                return

            new_models: list[dict] = result_holder.get("models", [])
            if not new_models:
                messagebox.showwarning(
                    "Nessun modello trovato",
                    f"HuggingFace non ha restituito modelli {label}.\n"
                    "Controlla la connessione.",
                    parent=self,
                )
                return

            widget = self._model_widgets[kind]
            prev_active = widget.get_active()
            new_active = widget.replace_models(new_models, prev_active)

            # Warning se il modello di default non è più nella nuova lista
            if prev_active and new_active != prev_active:
                messagebox.showwarning(
                    "Modello di default non disponibile",
                    f"Il modello precedentemente selezionato:\n"
                    f"  «{prev_active}»\n"
                    f"non è presente nella nuova lista scaricata.\n\n"
                    f"Il nuovo modello di default è stato impostato a:\n"
                    f"  «{new_active}»",
                    parent=self,
                )

            messagebox.showinfo(
                "Lista aggiornata",
                f"Lista modelli {label} aggiornata: {len(new_models)} modelli.",
                parent=self,
            )

        thread = threading.Thread(target=_do_fetch, daemon=True)
        thread.start()

    # ------------------------------------------------------------------
    # Azioni
    # ------------------------------------------------------------------

    def _on_save(self) -> None:
        """Raccoglie i valori dal form, li valida e li persiste su disco.

        Legge tutti i widget registrati in :attr:`_vars` e
        :attr:`_model_widgets`, converte ``top_k`` a intero, salva tramite
        :meth:`~ragchat.utils.config.Config.save` e mostra un avviso se il
        modello di embedding è cambiato mentre un DB con modello diverso è
        aperto.  Chiude la finestra al termine.
        """
        new_values: dict = {}

        # Raccoglie i valori dai widget standard
        for key, widget in self._vars.items():
            if isinstance(widget, tk.Text):
                new_values[key] = widget.get("1.0", "end-1c")
            else:
                new_values[key] = widget.get()  # type: ignore[union-attr]

        # Coerce top_k a intero (lo spinbox usa StringVar)
        if "top_k" in new_values:
            new_values["top_k"] = int(new_values["top_k"])

        # Raccoglie lista modelli (dict) e modello attivo (stringa id) dai ModelListWidget
        for kind, mw in self._model_widgets.items():
            new_values[f"{kind}_models"] = mw.get_models()  # list[dict]
            new_values[f"{kind}_model"] = mw.get_active()   # str (id)

        # Verifica se il modello embedding è cambiato rispetto alla config attuale
        old_model = Config.load()["embedding_model"]
        new_embedding_model = new_values["embedding_model"]

        try:
            Config.save(new_values)
            logger.info("Impostazioni salvate.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Errore nel salvataggio delle impostazioni: %s", exc)
            messagebox.showerror(
                "Errore",
                f"Impossibile salvare le impostazioni:\n{exc}",
                parent=self,
            )
            return

        # Se il modello embedding è cambiato e un DB con modello diverso è caricato
        if new_embedding_model != old_model:
            if (
                self._current_db_model is not None
                and self._current_db_model != new_embedding_model
            ):
                messagebox.showwarning(
                    "Modello embedding cambiato",
                    f"Hai cambiato modello di embedding in '{new_embedding_model}'.\n\n"
                    f"Il DB caricato usa '{self._current_db_model}'.\n\n"
                    f"La ricerca RAG potrebbe restituire risultati inaffidabili "
                    f"perché gli embedding del DB sono stati generati con un modello "
                    f"diverso.\n\n"
                    f"Carica un DB indicizzato con '{new_embedding_model}' "
                    f"o riavvia l'applicazione.",
                    icon="warning",
                    parent=self,
                )

        self.destroy()
