"""Pipeline di ingestione documenti: Docling → chunking → FAISS store."""

import logging
import os
from typing import TYPE_CHECKING
from docling.datamodel.base_models import FormatToExtensions, InputFormat

if TYPE_CHECKING:
    from ragchat.core.vectorstore import FAISSStore

logger = logging.getLogger(__name__)


def get_supported_extensions() -> dict[InputFormat, list[str]]:
    """Restituisce le estensioni file supportate da Docling."""    
    result={}
    for input_format, extensions in FormatToExtensions.items():
        # input_format è un enum (es. InputFormat.PDF)
        result[input_format.name]=extensions
    return result


def ingest_document(file_path: str, faiss_store: "FAISSStore") -> int:
    """Esegue la pipeline completa di ingestione di un documento.

    1. Verifica esistenza file.
    2. Converte il documento con Docling (→ markdown strutturato).
    3. Divide in chunk con RecursiveCharacterTextSplitter.
    4. Crea oggetti Document con metadati.
    5. Inserisce i chunk nel FAISS store.

    Args:
        file_path: Percorso assoluto o relativo al documento da indicizzare.
        faiss_store: Istanza FAISSStore in cui inserire i chunk.

    Returns:
        Numero di chunk inseriti nel DB.

    Raises:
        FileNotFoundError: Se ``file_path`` non esiste.
        ValueError: Se Docling non riesce a convertire il documento.
    """
    # Lazy imports per evitare caricamento di torch al semplice import del modulo
    from langchain_text_splitters.character import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from ragchat.core.embeddings import format_passage

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File non trovato: {file_path}")

    logger.info("Conversione documento con Docling: %s", file_path)
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(file_path)
        text = result.document.export_to_markdown()
    except Exception as e:
        raise ValueError(f"Impossibile convertire {file_path}: {e}") from e

    logger.info("Chunking del testo (chunk_size=512, overlap=50)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    logger.info("Prodotti %d chunk.", len(chunks))

    filename = os.path.basename(file_path)
    docs = [
        Document(
            page_content=format_passage(chunk),
            metadata={"source": filename, "chunk_index": i},
        )
        for i, chunk in enumerate(chunks)
    ]

    logger.info("Inserimento %d chunk nel FAISS store...", len(docs))
    inserted = faiss_store.add_documents(docs, filename)
    logger.info("Ingestione completata: %d chunk inseriti.", inserted)
    return inserted
