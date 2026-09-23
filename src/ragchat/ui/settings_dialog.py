"""Finestra di dialogo per la modifica delle impostazioni dell'applicazione.

Apre una toplevel modale con un form che rispecchia le chiavi del file
``config.properties``.  Le modifiche vengono persistite su disco solo al
clic di "Salva".
"""

import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import ragchat.config as cfg

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


class _ModelListWidget(tk.Frame):
    """Widget compatto per gestire una lista di modelli con selezione del modello attivo.

    Struttura visiva::

        [Combobox modello attivo v]
        +--------------------------+
        | modello1                 |  <- Listbox
        | modello2  (*)            |
        | ...                      |
        +--------------------------+
        [+ Aggiungi] [- Rimuovi]

    Il modello selezionato nella combobox e' quello che verra' salvato come
    ``<kind>_model``; la listbox mostra l'intera lista ``<kind>_models``.
    I due valori sono sempre mantenuti coerenti:
    - aggiungere/rimuovere dalla listbox aggiorna automaticamente la combobox.
    - cambiare la combobox non tocca la listbox (ma aggiunge il valore se assente).
    """

    def __init__(
        self,
        parent: tk.Widget,
        models: list[str],
        active: str,
    ) -> None:
        super().__init__(parent)
        self.columnconfigure(0, weight=1)

        self._models: list[str] = list(models)
        # Garantisce che il modello attivo sia in lista
        if active not in self._models:
            self._models.insert(0, active)

        # ── Combobox modello attivo ──────────────────────────────────────
        self._active_var = tk.StringVar(value=active)
        self._combo = ttk.Combobox(
            self,
            textvariable=self._active_var,
            values=self._models,
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

        self._refresh_listbox()

        # ── Pulsanti Aggiungi / Rimuovi ──────────────────────────────────
        btn_frame = tk.Frame(self)
        btn_frame.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        ttk.Button(btn_frame, text="+ Aggiungi", command=self._on_add).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(btn_frame, text="- Rimuovi", command=self._on_remove).pack(
            side="left"
        )

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def get_models(self) -> list[str]:
        """Restituisce la lista corrente dei modelli."""
        return list(self._models)

    def get_active(self) -> str:
        """Restituisce il modello attivo selezionato."""
        return self._active_var.get()

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _on_add(self) -> None:
        """Apre una piccola finestra di input per aggiungere un nuovo modello."""
        dialog = tk.Toplevel(self.winfo_toplevel())
        dialog.title("Aggiungi modello")
        dialog.resizable(False, False)
        dialog.grab_set()
        dialog.transient(self.winfo_toplevel())

        tk.Label(dialog, text="Nome / repo-id del modello:").pack(
            padx=12, pady=(10, 2), anchor="w"
        )
        entry_var = tk.StringVar()
        entry = ttk.Entry(dialog, textvariable=entry_var, width=52)
        entry.pack(padx=12, pady=(0, 8))
        entry.focus_set()

        def _confirm() -> None:
            name = entry_var.get().strip()
            if not name:
                return
            if name not in self._models:
                self._models.append(name)
                self._refresh_listbox()
                self._refresh_combo()
            dialog.destroy()

        entry.bind("<Return>", lambda _e: _confirm())

        btn_row = tk.Frame(dialog)
        btn_row.pack(padx=12, pady=(0, 10))
        ttk.Button(btn_row, text="Aggiungi", command=_confirm).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(btn_row, text="Annulla", command=dialog.destroy).pack(side="left")

        # Centra sul parent
        dialog.update_idletasks()
        px = self.winfo_toplevel().winfo_rootx() + self.winfo_toplevel().winfo_width() // 2
        py = self.winfo_toplevel().winfo_rooty() + self.winfo_toplevel().winfo_height() // 2
        dialog.geometry(f"+{px - dialog.winfo_width() // 2}+{py - dialog.winfo_height() // 2}")

    def _on_remove(self) -> None:
        """Rimuove il modello selezionato nella listbox (non permette di rimuovere l'ultimo)."""
        sel = self._listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        model = self._models[idx]

        if len(self._models) == 1:
            messagebox.showwarning(
                "Rimozione non consentita",
                "Deve essere presente almeno un modello.",
                parent=self.winfo_toplevel(),
            )
            return

        self._models.pop(idx)
        # Se il modello rimosso era quello attivo, passa al primo della lista
        if self._active_var.get() == model:
            self._active_var.set(self._models[0])
        self._refresh_listbox()
        self._refresh_combo()

    # ------------------------------------------------------------------
    # Helpers interni
    # ------------------------------------------------------------------

    def _refresh_listbox(self) -> None:
        active = self._active_var.get()
        self._listbox.delete(0, tk.END)
        for m in self._models:
            label = f"{m}  (*)" if m == active else m
            self._listbox.insert(tk.END, label)

    def _refresh_combo(self) -> None:
        self._combo["values"] = self._models
        # Se il valore attuale non e' piu' valido, seleziona il primo
        if self._active_var.get() not in self._models:
            self._active_var.set(self._models[0])


class SettingsDialog(tk.Toplevel):
    """Finestra modale per la modifica di config.properties."""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self.title("Impostazioni")
        self.resizable(True, True)
        self.grab_set()  # modale rispetto alla finestra principale

        # Carica valori correnti
        self._values = cfg.load()
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

        models = cfg.str_to_models(self._values.get(list_key, ""))
        active = self._values.get(active_key, models[0] if models else "")

        widget = _ModelListWidget(parent, models=models, active=active)
        widget.grid(row=row, column=1, sticky="ew", pady=4)
        self._model_widgets[kind] = widget
        return row + 1

    def _add_spinbox_row(
        self, parent: tk.Widget, row: int, key: str, from_: int, to: int
    ) -> int:
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
        self._label(parent, row, key)
        var = tk.StringVar(value=self._values.get(key, choices[0]))
        self._vars[key] = var
        cb = ttk.Combobox(
            parent, textvariable=var, values=choices, state="readonly", width=14
        )
        cb.grid(row=row, column=1, sticky="w", pady=4)
        return row + 1

    def _add_path_row(self, parent: tk.Widget, row: int, key: str) -> int:
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
    # Azioni
    # ------------------------------------------------------------------

    def _on_save(self) -> None:
        new_values: dict[str, str] = {}

        # Raccoglie i valori dai widget standard
        for key, widget in self._vars.items():
            if isinstance(widget, tk.Text):
                new_values[key] = widget.get("1.0", "end-1c")
            else:
                new_values[key] = widget.get()  # type: ignore[union-attr]

        # Raccoglie lista modelli e modello attivo dai ModelListWidget
        for kind, mw in self._model_widgets.items():
            new_values[f"{kind}_models"] = cfg.models_to_str(mw.get_models())
            new_values[f"{kind}_model"] = mw.get_active()

        try:
            cfg.save(new_values)
            logger.info("Impostazioni salvate.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Errore nel salvataggio delle impostazioni: %s", exc)
            messagebox.showerror(
                "Errore",
                f"Impossibile salvare le impostazioni:\n{exc}",
                parent=self,
            )
            return

        self.destroy()
