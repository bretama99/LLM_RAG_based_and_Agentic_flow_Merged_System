"""
config.py
Global configuration settings for the Research Assistant.
"""

import os

# ======================
# MODEL CONFIG
# ======================
# Replace with your actual model identifiers (OpenAI, Gemini, Ollama, etc.)
LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ======================
# PATHS
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")
PDF_GLOBAL_DIR = os.path.join(DATA_DIR, "pdfs_global")
PDF_UPLOADED_DIR = os.path.join(DATA_DIR, "pdfs_uploaded")
EXPORTS_DIR = os.path.join(DATA_DIR, "exports")

## ======================
# RAG SETTINGS
# ======================
CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 150       # overlap between chunks
TOP_K_RESULTS = 5         # retrieval top-k

# Safety limits to avoid MemoryError / heavy CPU on huge PDFs
MAX_PDF_CHARS = 20_000        # max characters per PDF (very conservative)
MAX_PDF_PAGES = 20            # max pages per PDF
MAX_CHUNKS_PER_PDF = 50       # max chunks per PDF to embed

# ======================
# LOGGING
# ======================
LOG_FILE = os.path.join(BASE_DIR, "logs", "app.log")
