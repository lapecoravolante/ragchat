"""Wrapper FAISS per la gestione del ciclo di vita del DB vettoriale su disco.

Il DB viene creato fisicamente solo al primo ``add_documents``: prima di quel
momento ``_db`` è ``None`` e tutte le operazioni di lettura restituiscono
risultati vuoti / sollevano ``RuntimeError`` dove appropriato.
"""

import logging
import os

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from ragchat.core.embeddings import get_embeddings
from ragchat.utils.metadata import MetadataStore

logger = logging.getLogger(__name__)


class FAISSStore:
    """Gestisce creazione, apertura, salvataggio e rimozione documenti nel DB FAISS.

    Il DB FAISS viene serializzato su disco nella cartella ``db_path``.
    La sincronizzazione dei metadati (nome file → chunk IDs) è delegata a
    :class:`~ragchat.utils.metadata.MetadataStore`.

    Parametri:
        db_path: Percorso della cartella che contiene (o conterrà) il DB.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._db: FAISS | None = None
        self._metadata = MetadataStore(db_path)

    # ------------------------------------------------------------------
    # Metodi di fabbrica (class-level)
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, db_path: str) -> "FAISSStore":
        """Crea un nuovo DB FAISS vuoto nella cartella *db_path*.

        La cartella viene creata se non esiste.  Il DB fisico viene
        inizializzato solo al primo :meth:`add_documents` (``_db = None``).

        Parametri:
            db_path: Percorso della cartella di destinazione.

        Restituisce:
            Istanza ``FAISSStore`` pronta all'uso.
        """
        os.makedirs(db_path, exist_ok=True)
        logger.info("Nuovo DB FAISS in '%s'.", db_path)
        return cls(db_path)

    @classmethod
    def load(cls, db_path: str) -> "FAISSStore":
        """Carica un DB FAISS esistente da *db_path*.

        Se la cartella è vuota o non contiene un DB FAISS valido, ``_db``
        rimane ``None`` (nessun errore viene sollevato).

        Parametri:
            db_path: Percorso della cartella del DB.

        Restituisce:
            Istanza ``FAISSStore`` con il DB caricato (o vuota se assente).
        """
        store = cls(db_path)

        index_file = os.path.join(db_path, "index.faiss")
        if not os.path.exists(index_file):
            logger.info(
                "Nessun DB FAISS trovato in '%s'. Il DB sarà creato al primo inserimento.",
                db_path,
            )
            return store

        try:
            store._db = FAISS.load_local(
                db_path,
                get_embeddings(),
                allow_dangerous_deserialization=True,
            )
            logger.info("DB FAISS caricato da '%s'.", db_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Impossibile caricare il DB FAISS da '%s': %s", db_path, exc)

        return store

    # ------------------------------------------------------------------
    # Persistenza
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Salva il DB su disco. Se ``_db`` è ``None`` non fa nulla."""
        if self._db is None:
            return
        self._db.save_local(self.db_path)
        logger.debug("DB FAISS salvato in '%s'.", self.db_path)

    def _remove_saved_index(self) -> None:
        """Rimuove i file persistiti dell'indice vuoto."""
        for filename in ("index.faiss", "index.pkl"):
            try:
                os.remove(os.path.join(self.db_path, filename))
            except FileNotFoundError:
                pass

    # ------------------------------------------------------------------
    # Operazioni sui documenti
    # ------------------------------------------------------------------

    def add_documents(self, docs: list[Document], filename: str) -> int:
        """Aggiunge i chunk *docs* al DB e aggiorna i metadati.

        Se il DB non è ancora stato creato fisicamente, lo inizializza con
        :meth:`FAISS.from_documents`. Se *filename* esiste già, i suoi vecchi
        chunk vengono sostituiti dai nuovi. Il DB viene salvato automaticamente.

        Parametri:
            docs:     Lista di :class:`~langchain_core.documents.Document` da inserire.
            filename: Nome del file sorgente (usato come chiave nei metadati).

        Restituisce:
            Numero di chunk indicizzati per il documento.
        """
        if not docs:
            logger.warning("add_documents chiamato con lista vuota per '%s'.", filename)
            return 0

        previous_entry = self._metadata.get_document(filename)
        previous_ids = set(previous_entry.get("chunk_ids", [])) if previous_entry else set()

        if self._db is None:
            keys_before: set[str] = set()
            self._db = FAISS.from_documents(docs, get_embeddings())
        else:
            keys_before = set(self._db.docstore._dict.keys())
            previous_ids.update(
                doc_id
                for doc_id, document in self._db.docstore._dict.items()
                if document.metadata.get("source") == filename
            )
            self._db.add_documents(docs)

        keys_after: set[str] = set(self._db.docstore._dict.keys())
        new_ids: list[str] = list(keys_after - keys_before)

        # Registra il modello insieme agli ID vecchi e nuovi prima di toccare
        # l'indice, così un errore successivo non rende i vecchi chunk orfani.
        from ragchat.core.embeddings import _get_model_name
        model_name = _get_model_name()
        try:
            self._metadata.add_document(
                filename,
                sorted(previous_ids | set(new_ids)),
                embedding_model=model_name,
            )
        except OSError:
            if new_ids:
                self._db.delete(ids=new_ids)
            if not self._db.docstore._dict:
                self._db = None
            raise

        existing_ids = set(self._db.index_to_docstore_id.values())
        obsolete_ids = sorted(previous_ids & existing_ids)
        if obsolete_ids:
            self._db.delete(ids=obsolete_ids)

        self.save()
        self._metadata.add_document(filename, new_ids)

        logger.info("Aggiunti %d chunk per '%s'.", len(new_ids), filename)
        return len(new_ids)

    def remove_document(self, filename: str) -> bool:
        """Rimuove dal DB tutti i chunk associati a *filename*.

        Elimina i vettori associati al documento usando gli ID FAISS, senza
        ricostruire l'indice né modificare gli ID dei documenti rimanenti.

        Parametri:
            filename: Nome del file da rimuovere.

        Restituisce:
            ``True`` se il documento era presente ed è stato rimosso,
            ``False`` se non era nel metadata store.
        """
        if self._db is None:
            return False

        entry = self._metadata.get_document(filename)
        if entry is None:
            logger.warning("'%s' non trovato nel metadata store.", filename)
            return False

        indexed_ids = set(self._db.index_to_docstore_id.values())
        ids_by_source = {
            doc_id
            for doc_id, document in self._db.docstore._dict.items()
            if document.metadata.get("source") == filename
        }
        ids_to_remove = (set(entry.get("chunk_ids", [])) | ids_by_source) & indexed_ids
        if ids_to_remove:
            self._db.delete(ids=sorted(ids_to_remove))

        if not self._db.docstore._dict:
            self._db = None

        if self._db is None:
            self._remove_saved_index()
        else:
            self.save()
        self._metadata.remove_document(filename)
        logger.info("Documento '%s' rimosso (%d chunk eliminati).", filename, len(ids_to_remove))
        return True

    def list_documents(self) -> list[dict]:
        """Restituisce la lista dei documenti indicizzati.

        Delega a :class:`~ragchat.utils.metadata.MetadataStore`.
        """
        return self._metadata.list_documents()

    def as_retriever(self, k: int = 4):
        """Restituisce un retriever LangChain per il DB.

        Parametri:
            k: Numero di chunk da recuperare per ogni query.

        Restituisce:
            Istanza ``VectorStoreRetriever``.

        Solleva:
            RuntimeError: se il DB è vuoto (nessun documento indicizzato).
        """
        if self._db is None:
            raise RuntimeError("Nessun documento nel DB")
        return self._db.as_retriever(search_kwargs={"k": k})

    def is_empty(self) -> bool:
        """Restituisce ``True`` se il DB non contiene documenti."""
        if self._db is None:
            return True
        return len(self._db.docstore._dict) == 0
