"""Catena RAG: retrieval FAISS + prompt template + LLM locale.

Il LLM e le dipendenze pesanti vengono importati in modo lazy all'interno di
:meth:`RAGChain.ask` per evitare import circolari e rallentamenti all'avvio.
"""

import logging

from ragchat.core.vectorstore import FAISSStore

logger = logging.getLogger(__name__)


class RAGChain:
    """Catena RAG che combina retrieval FAISS, prompt e LLM locale.

    Args:
        faiss_store: Istanza :class:`~ragchat.core.vectorstore.FAISSStore`
            già caricata.  Il LLM **non** viene caricato nel costruttore
            (lazy loading in :meth:`ask`).
    """

    def __init__(self, faiss_store: FAISSStore) -> None:
        self._store = faiss_store

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def ask(self, question: str) -> str:
        """Esegue la pipeline RAG completa e restituisce la risposta.

        Pipeline:
        1. Verifica che il DB non sia vuoto.
        2. Recupera i chunk più rilevanti dal FAISS store (top-4).
        3. Costruisce il prompt con il contesto recuperato.
        4. Invoca il LLM locale e restituisce la risposta.

        Args:
            question: Domanda dell'utente in linguaggio naturale.

        Returns:
            Stringa di risposta oppure un messaggio informativo se il DB è
            vuoto o il LLM non è disponibile.
        """
        # 1. Verifica DB vuoto
        if self._store.is_empty():
            return (
                "Nessun documento nel database. "
                "Carica prima dei documenti nel pannello DB."
            )

        try:
            # 2. Retrieval
            from ragchat.core.embeddings import format_query  # lazy import

            retriever = self._store.as_retriever(k=4)
            query_with_prefix = format_query(question)
            docs = retriever.invoke(query_with_prefix)

            # 3. Costruzione contesto
            context = "\n\n---\n\n".join(doc.page_content for doc in docs)

            # 4. Prompt
            prompt_text = (
                "Sei un assistente utile. Rispondi alla domanda basandoti "
                "esclusivamente sul contesto fornito.\n"
                "Se il contesto non contiene informazioni sufficienti per "
                "rispondere, dillo chiaramente.\n\n"
                f"Contesto:\n{context}\n\n"
                f"Domanda: {question}\n\n"
                "Risposta:"
            )

            # 5. LLM
            from ragchat.core.llm import get_llm, is_llm_available  # lazy import

            if not is_llm_available():
                return (
                    "LLM non disponibile. "
                    "Installa llama-cpp-python per usare il chatbot."
                )

            llm = get_llm()
            response = llm.invoke(prompt_text)
            return response.strip()

        except Exception as exc:  # noqa: BLE001
            logger.error("Errore durante la pipeline RAG: %s", exc, exc_info=True)
            return f"Si è verificato un errore durante l'elaborazione: {exc}"

    def update_store(self, faiss_store: FAISSStore) -> None:
        """Aggiorna il FAISSStore (chiamato quando l'utente cambia DB).

        Args:
            faiss_store: Nuova istanza :class:`~ragchat.core.vectorstore.FAISSStore`.
        """
        self._store = faiss_store
        logger.debug("FAISSStore aggiornato in RAGChain.")
