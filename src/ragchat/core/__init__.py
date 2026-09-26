"""Package ``ragchat.core`` — logica applicativa.

Contiene i moduli che implementano la pipeline RAG:

- :mod:`~ragchat.core.embeddings` — singleton lazy del modello di embedding.
- :mod:`~ragchat.core.vectorstore` — wrapper FAISS per il DB vettoriale.
- :mod:`~ragchat.core.ingestor` — pipeline Docling → chunking → FAISS.
- :mod:`~ragchat.core.rag` — catena RAG completa (retrieval + LLM).
- :mod:`~ragchat.core.huggingface` — integrazione HuggingFace Hub e LLM locale.
"""
