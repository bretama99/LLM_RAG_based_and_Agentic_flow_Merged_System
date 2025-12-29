"""
local_rag_langchain.py
Local RAG using LangChain, similar to professor's lab code:

- PyPDFLoader
- RecursiveCharacterTextSplitter
- HuggingFaceEmbeddings (all-MiniLM-L6-v2)
- InMemoryVectorStore

This module handles:
- init_local_rag()       → initialize empty vector store (idempotent)
- ingest_pdfs(pdf_paths) → load + chunk + embed + index
- local_search(query,k)  → retrieve top-k relevant chunks
"""

import logging
import os
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from app.config import MAX_PDF_PAGES

# Globals
_embeddings = None
_vector_store = None
_retriever = None

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5


def _ensure_embeddings():
    """Load HuggingFaceEmbeddings once."""
    global _embeddings
    if _embeddings is None:
        logging.info("[LOCAL RAG] Loading HuggingFaceEmbeddings (all-MiniLM-L6-v2)...")
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        logging.info("[LOCAL RAG] Embeddings model loaded.")
    return _embeddings


def init_local_rag():
    """
    Initialize an empty InMemoryVectorStore and retriever.

    Idempotent: if the vector store already exists, it will NOT be recreated.
    This prevents us from wiping previously ingested documents on Streamlit reruns.
    """
    global _vector_store, _retriever

    if _vector_store is not None and _retriever is not None:
        # Already initialized, do nothing
        logging.info("[LOCAL RAG] Vector store already initialized, skipping re-init.")
        return _vector_store, _retriever

    embeddings = _ensure_embeddings()
    logging.info("[LOCAL RAG] Initializing empty InMemoryVectorStore...")
    _vector_store = InMemoryVectorStore(embeddings)
    _retriever = _vector_store.as_retriever(search_kwargs={"k": DEFAULT_TOP_K})
    logging.info("[LOCAL RAG] InMemoryVectorStore initialized.")
    return _vector_store, _retriever


def ingest_pdfs(pdf_paths: List[str]):
    """
    Ingest one or more PDFs into the global InMemoryVectorStore.

    Steps:
    - Load each PDF with PyPDFLoader (page-level Documents)
    - Optionally limit to MAX_PDF_PAGES
    - Split into chunks
    - Add chunks to the vector store
    """
    global _vector_store, _retriever

    # Ensure store exists, but DO NOT wipe if it already has data
    init_local_rag()

    if not pdf_paths:
        logging.info("[LOCAL RAG] No PDF paths provided to ingest.")
        return

    logging.info(f"[LOCAL RAG] Ingesting {len(pdf_paths)} PDF(s)...")
    all_docs = []

    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            logging.warning(f"[LOCAL RAG] PDF not found, skipping: {pdf_path}")
            continue

        logging.info(f"[LOCAL RAG] Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        if len(docs) > MAX_PDF_PAGES:
            docs = docs[:MAX_PDF_PAGES]
            logging.info(
                f"[LOCAL RAG] Limiting to first {MAX_PDF_PAGES} pages for {pdf_path}"
            )

        all_docs.extend(docs)

    if not all_docs:
        logging.info("[LOCAL RAG] No documents loaded; nothing to ingest.")
        return

    logging.info(f"[LOCAL RAG] Loaded {len(all_docs)} page-level documents.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    chunks = splitter.split_documents(all_docs)
    logging.info(f"[LOCAL RAG] Created {len(chunks)} text chunks total.")

    ids = _vector_store.add_documents(chunks)
    logging.info(f"[LOCAL RAG] Indexed {len(ids)} chunks into InMemoryVectorStore.")

    # Refresh retriever (now it sees the new chunks)
    _retriever = _vector_store.as_retriever(search_kwargs={"k": DEFAULT_TOP_K})
    logging.info("[LOCAL RAG] Retriever updated with new chunks.")


def local_search(query: str, k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """
    Perform semantic search over the global InMemoryVectorStore.

    Returns a list of dicts:
    - 'text'     → chunk text
    - 'metadata' → LangChain's metadata (page, source, etc.)
    """
    global _retriever

    if not query.strip():
        return []

    if _retriever is None:
        logging.info("[LOCAL RAG] Retriever is None. Did you ingest any PDFs?")
        return []

    logging.info(f"[LOCAL RAG] Retrieving for query: {query}")
    docs = _retriever.invoke(query)

    results: List[Dict[str, Any]] = []
    for d in docs[:k]:
        results.append(
            {
                "text": d.page_content,
                "metadata": d.metadata,
            }
        )

    logging.info(f"[LOCAL RAG] Retrieved {len(results)} chunk(s) for query.")
    return results
