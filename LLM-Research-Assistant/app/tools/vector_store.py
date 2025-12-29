"""
vector_store.py
Lightweight in-memory "vector store" (actually a simple chunk store)
to avoid heavy embeddings and ChromaDB while keeping the RAG pipeline
structure for this project.

For each PDF, we:
- extract text
- chunk it
- store chunks in a global list with metadata

For search, we:
- do a naive keyword-based scoring over the chunks
- return top-k chunks
"""

import os
import logging
from typing import List, Dict, Any

from app.tools.pdf_loader import extract_text_from_pdf, chunk_text

# Global in-memory store of chunks
# Each item is: {"text": <chunk>, "metadata": {...}}
DOCUMENT_STORE: List[Dict[str, Any]] = []


def init_vector_store():
    """
    Initialize or get the global in-memory document store.

    In this simplified version, we just return the global list.
    No embeddings, no external DB → very lightweight.
    """
    logging.info("Using lightweight in-memory document store (no embeddings).")
    return DOCUMENT_STORE


def index_pdf_file(store, pdf_path: str, source_label: str = "uploaded"):
    """
    Extract text from a PDF, chunk it, and add it to the in-memory store.

    store: the global DOCUMENT_STORE (list).
    source_label: "global" or "uploaded".
    """
    logging.info(f"[INDEX] Starting indexing PDF: {pdf_path} (source={source_label})")

    text = extract_text_from_pdf(pdf_path)
    logging.info(f"[INDEX] Extracted text length: {len(text)} characters.")

    chunks = chunk_text(text)
    logging.info(f"[INDEX] Created {len(chunks)} chunks from PDF.")

    filename = os.path.basename(pdf_path)

    for i, ch in enumerate(chunks):
        store.append(
            {
                "text": ch,
                "metadata": {
                    "source": source_label,
                    "path": pdf_path,
                    "filename": filename,
                    "chunk_index": i,
                },
            }
        )

    logging.info(f"[INDEX] Finished indexing PDF: {filename}. Total chunks in store: {len(store)}")


def search_vector_store(store, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Naive keyword-based search over the in-memory document store.

    - Splits query into lowercased words.
    - Scores each chunk by how many query words appear in the text.
    - Returns top-k chunks with score > 0 (if any), or first k if all scores are 0.
    """
    if not query.strip():
        return []

    if not store:
        logging.info("[SEARCH] Document store is empty. No local results.")
        return []

    query_words = set(query.lower().split())
    scored = []

    for item in store:
        text_l = item["text"].lower()
        score = sum(1 for w in query_words if w in text_l)
        scored.append((score, item))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Filter out zero-score chunks if we have some with positive score
    max_score = scored[0][0] if scored else 0
    if max_score > 0:
        scored = [s for s in scored if s[0] > 0]

    top_items = [s[1] for s in scored[:k]]
    logging.info(f"[SEARCH] Query='{query}' → returning {len(top_items)} local chunks.")
    return top_items
