"""
rag_local.py
Perform local RAG over PDFs stored in the vector store.
"""

from typing import List, Dict, Any

from app.tools.vector_store import search_vector_store


def rag_local(store, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Local retrieval over the vector store.
    'store' is the Chroma collection returned by init_vector_store().
    """
    return search_vector_store(store, query, k=k)
