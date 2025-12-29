"""
Basic test for local RAG placeholder.
"""

from app.tools.vector_store import init_vector_store, add_document_to_vector_store
from app.tools.rag_local import rag_local

def test_rag_local_placeholder():
    store = init_vector_store()
    add_document_to_vector_store(store, ["dummy chunk"], {"source": "test"})
    results = rag_local(store, "test query")
    assert len(results) >= 1
