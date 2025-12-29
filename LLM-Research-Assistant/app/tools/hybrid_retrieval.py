"""
hybrid_retrieval.py
Select between local, web, or hybrid retrieval.
"""

from typing import List
from app.tools.rag_local import rag_local
from app.tools.rag_web import rag_web

def hybrid_retrieval(store, query: str, mode: str, k: int = 5) -> List[dict]:
    """
    mode: "local", "web", or "hybrid"
    """
    mode = mode.lower()
    if mode == "local":
        return rag_local(store, query, k=k)
    if mode == "web":
        return rag_web(query, k=k)
    # hybrid
    local_results = rag_local(store, query, k=k)
    web_results = rag_web(query, k=k)
    return local_results + web_results
