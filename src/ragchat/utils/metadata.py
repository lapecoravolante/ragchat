"""
Gestione persistenza JSON dei metadati dei documenti inseriti nel DB FAISS.
"""

import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

_METADATA_FILENAME = "metadata.json"
_EMPTY_STORE: dict = {"documents": {}}


class MetadataStore:
    """Gestisce il file JSON dei metadati associato a un DB FAISS.

    Il file ``metadata.json`` viene mantenuto nella stessa cartella del DB.
    Ogni entrata mappa il nome del file sorgente ai chunk IDs corrispondenti
    e al timestamp di inserimento.

    Args:
        db_path: Percorso della cartella contenente il DB FAISS.
    """

    def __init__(self, db_path: str) -> None:
        self._json_path = os.path.join(db_path, _METADATA_FILENAME)
        self._data: dict = {}
        self.load()

    # ------------------------------------------------------------------
    # Persistenza
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Carica il file JSON da disco.

        Se il file non esiste inizializza una struttura vuota.
        Se il file è corrotto lo segnala con un warning e inizializza vuoto.
        """
        if not os.path.exists(self._json_path):
            self._data = {"documents": {}}
            return

        try:
            with open(self._json_path, encoding="utf-8") as fh:
                self._data = json.load(fh)
            if "documents" not in self._data:
                self._data["documents"] = {}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Impossibile leggere '%s' (%s). Il file verrà reinizializzato.",
                self._json_path,
                exc,
            )
            self._data = {"documents": {}}

    def save(self) -> None:
        """Serializza e salva il JSON su disco."""
        try:
            with open(self._json_path, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.error("Impossibile salvare '%s': %s", self._json_path, exc)

    # ------------------------------------------------------------------
    # API documenti
    # ------------------------------------------------------------------

    def add_document(self, name: str, chunk_ids: list[int]) -> None:
        """Aggiunge o sostituisce l'entrata per il documento *name*.

        Salva automaticamente dopo la modifica.
        """
        self._data["documents"][name] = {
            "filename": name,
            "chunk_ids": chunk_ids,
            "inserted_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.save()

    def remove_document(self, name: str) -> list[int]:
        """Rimuove l'entrata del documento *name*.

        Returns:
            Lista dei chunk IDs associati al documento, o lista vuota se
            il documento non era presente.
        """
        entry = self._data["documents"].pop(name, None)
        if entry is None:
            return []
        self.save()
        return entry.get("chunk_ids", [])

    def list_documents(self) -> list[dict]:
        """Restituisce la lista di tutte le entrate presenti nel metadata store."""
        return list(self._data["documents"].values())

    def get_document(self, name: str) -> dict | None:
        """Restituisce l'entrata del documento *name*, o ``None`` se assente."""
        return self._data["documents"].get(name)
