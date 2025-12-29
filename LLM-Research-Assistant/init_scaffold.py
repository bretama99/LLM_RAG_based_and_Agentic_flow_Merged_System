"""
init_scaffold.py
Bootstrap script to populate the project files with initial template code.

Run this once from the project root:
    python init_scaffold.py

It will OVERWRITE existing files with these templates.
"""

import os
from pathlib import Path

# Base directory = folder where this script lives
BASE_DIR = Path(__file__).resolve().parent

def write_file(rel_path: str, content: str):
    """Create/overwrite a file with the given content."""
    file_path = BASE_DIR / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"[OK] Wrote {rel_path}")

# =========================
# TEMPLATES
# =========================

CONFIG_PY = r'''
"""
config.py
Global configuration settings for the Research Assistant.
"""

import os

# ======================
# MODEL CONFIG
# ======================
# Replace with your actual model identifiers (OpenAI, Gemini, Ollama, etc.)
LLM_MODEL = "gpt-4o-mini"
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

# ======================
# RAG SETTINGS
# ======================
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_RESULTS = 5

# ======================
# LOGGING
# ======================
LOG_FILE = os.path.join(BASE_DIR, "logs", "app.log")
'''

LOGGING_CONFIG_PY = r'''
"""
logging_config.py
Application-wide logging setup.
"""

import logging
from app.config import LOG_FILE

def setup_logging():
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )
    logging.info("Logging initialized.")
'''

AGENTS_PY = r'''
"""
agents.py
Defines CrewAI agents: planner, retrieval, summarizer, writer, reflection.

NOTE: LLM and tools will be attached in the orchestrator.
"""

from crewai import Agent

planner_agent = Agent(
    role="Planner",
    goal=(
        "Determine whether the user is asking for summarization, comparison, "
        "explanation, related work, or general Q&A, and plan the reasoning steps."
    ),
    backstory="You analyze research questions and decide the appropriate workflow.",
    llm=None,
)

retrieval_agent = Agent(
    role="Retrieval",
    goal="Retrieve the most relevant evidence from local PDFs and/or the web.",
    backstory="You specialize in hybrid retrieval-augmented generation (RAG).",
    llm=None,
)

summarizer_agent = Agent(
    role="Summarizer",
    goal="Produce structured notes from retrieved documents (methods, metrics, etc.).",
    backstory="You are good at turning raw text into clear bullet-point summaries.",
    llm=None,
)

writer_agent = Agent(
    role="Writer",
    goal=(
        "Synthesize a structured academic-style answer with inline numeric citations "
        "and a References section."
    ),
    backstory="You write like a research assistant helping with scientific papers.",
    llm=None,
)

reflection_agent = Agent(
    role="Reflection",
    goal=(
        "Review the draft answer, detect unsupported claims or missing citations, "
        "and propose improvements."
    ),
    backstory="You act as a critic to improve factuality and clarity.",
    llm=None,
)
'''

TASKS_PY = r'''
"""
tasks.py
Defines high-level tasks for CrewAI.

For now we keep this simple; the orchestrator will construct tasks dynamically.
"""

from crewai import Task

def create_research_task(question: str, source_mode: str, params: dict) -> Task:
    """
    Create a generic research task description for the crew.
    """
    description = (
        f"User question: {question}\n\n"
        f"Source mode: {source_mode}\n"
        f"Temperature: {params.get('temperature')}\n"
        f"Max tokens: {params.get('max_tokens')}\n\n"
        "Respond with a structured academic-style answer including:\n"
        "- Short executive summary\n"
        "- Detailed explanation with headings\n"
        "- Inline numeric citations like [1], [2]\n"
        "- A References section at the end."
    )

    return Task(
        description=description,
        expected_output="A final answer in academic style as described above.",
        agent=None,  # Will be assigned in orchestrator
    )
'''

PDF_LOADER_PY = r'''
"""
pdf_loader.py
Functions to load and chunk PDF documents.

This is a placeholder; later you will plug in pypdf, pdfplumber, or unstructured.
"""

from typing import List

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract raw text from a PDF file.

    TODO: Implement using a real PDF library.
    """
    # Placeholder implementation
    return f"TEXT_FROM_PDF({pdf_path})"

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Naive text chunking: split into chunks of fixed size with overlap.
    """
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if start >= n:
            break
    return chunks
'''

VECTOR_STORE_PY = r'''
"""
vector_store.py
Vector store initialization and search utilities.

You can later replace the in-memory placeholders with Chroma, FAISS, etc.
"""

from typing import List, Any

# Very simple in-memory store placeholder
_VECTOR_STORE = []

def init_vector_store():
    """
    Initialize the vector store.

    TODO: Replace with a real vector DB (e.g., Chroma or FAISS).
    """
    global _VECTOR_STORE
    _VECTOR_STORE = []
    return _VECTOR_STORE

def add_document_to_vector_store(store: Any, chunks: List[str], metadata: dict):
    """
    Add chunks with associated metadata into the vector store.

    For now, store them as a list of dicts (placeholder).
    """
    for ch in chunks:
        store.append({"text": ch, "metadata": metadata})

def search_vector_store(store: Any, query: str, k: int = 5) -> List[dict]:
    """
    Perform a placeholder search in the vector store.

    TODO: Implement embeddings + similarity search.
    """
    # For now, just return first k items
    return store[:k]
'''

RAG_LOCAL_PY = r'''
"""
rag_local.py
Perform local RAG over PDFs stored in the vector store.
"""

from typing import List
from app.tools.vector_store import search_vector_store

def rag_local(store, query: str, k: int = 5) -> List[dict]:
    """
    Local retrieval over the vector store.
    """
    results = search_vector_store(store, query, k=k)
    return results
'''

RAG_WEB_PY = r'''
"""
rag_web.py
Perform web-based retrieval using a search API (e.g., Tavily, SerpAPI).

For now, this is a placeholder that returns dummy results.
"""

from typing import List

def rag_web(query: str, k: int = 5) -> List[dict]:
    """
    Web retrieval placeholder.

    TODO: Plug in a real web search client.
    """
    return [
        {
            "text": f"WEB_RESULT for query: {query}",
            "metadata": {"source": "web", "url": "https://example.com"}
        }
    ]
'''

HYBRID_RETRIEVAL_PY = r'''
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
'''

EXPORT_UTILS_PY = r'''
"""
export_utils.py
Convert final answer into PDF bytes for download.

This is a placeholder; you can later integrate reportlab or similar.
"""

def convert_to_pdf(text: str) -> bytes:
    """
    Convert text to a minimal PDF representation.

    TODO: Implement a real PDF export. For now, return dummy bytes.
    """
    dummy_pdf = f"%PDF-1.4\n% Dummy PDF content\n{text}\n%%EOF"
    return dummy_pdf.encode("utf-8")
'''

ORCHESTRATOR_PY = r'''
"""
orchestrator.py
Main logic for running the research assistant workflow.

For now, this is a high-level skeleton that you will progressively enhance.
"""

import logging
from typing import Tuple, List

from app.logging_config import setup_logging
from app.tools.vector_store import init_vector_store
from app.tools.hybrid_retrieval import hybrid_retrieval

# Initialize logging once
setup_logging()

# Initialize vector store once (placeholder)
VECTOR_STORE = init_vector_store()

def run_research_query(
    question: str,
    source_mode: str,
    llm_params: dict,
    debug_mode: bool = False
) -> Tuple[str, List[str], List[dict]]:
    """
    Orchestrate the end-to-end pipeline:
    - plan task (later via Planner agent)
    - retrieve evidence (local/web/hybrid)
    - summarize (later)
    - write answer (later with LLM)
    - reflect (later)

    Currently returns placeholder answer and references.
    """
    logging.info(f"Received query: {question} | mode={source_mode}")

    # STEP 1: (Future) detect task type -> summarization / comparison / qa / etc.
    task_type = "qa"
    if debug_mode:
        logging.info(f"[PLANNER] Task detected: {task_type}")

    # STEP 2: Retrieval
    retrieval_results = hybrid_retrieval(VECTOR_STORE, question, source_mode, k=5)
    if debug_mode:
        logging.info(f"[RETRIEVAL] Retrieved {len(retrieval_results)} items")

    # STEP 3: (Future) Summarization over retrieval_results

    # STEP 4: (Future) LLM-based writing using writer_agent
    final_answer = (
        "This is a placeholder answer. The real implementation will use a language "
        "model to synthesize a structured academic-style response based on retrieved "
        "evidence and user parameters."
    )

    # STEP 5: (Future) Reflection agent pass

    # Placeholder references
    references = ["[1] Placeholder Reference (to be replaced with real citations)."]

    return final_answer, references, retrieval_results
'''

UI_STREAMLIT_PY = r'''
"""
ui_streamlit.py
Streamlit front-end for the LLM-Powered Research Assistant.
"""

import streamlit as st
from app.pipeline.orchestrator import run_research_query
from app.tools.export_utils import convert_to_pdf

st.set_page_config(page_title="LLM Research Assistant", layout="wide")

st.title("🔍 LLM-Powered Research Assistant")

st.markdown(
    "Ask research questions, summarize or compare papers, and get structured "
    "answers with citations. This is a prototype built with CrewAI + RAG."
)

# ============= Sidebar =============
st.sidebar.header("⚙ Settings")

source_mode = st.sidebar.radio(
    "Sources to use:",
    options=["local", "web", "hybrid"],
    index=2,
    help="Choose whether to rely on local PDFs, web search, or both."
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.2, 0.3, 0.1)
max_tokens = st.sidebar.number_input("Max tokens", 100, 4000, 1200, step=100)
debug_mode = st.sidebar.checkbox("Show logs (debug mode)", value=False)

uploaded_pdfs = st.sidebar.file_uploader(
    "Upload PDFs (optional)",
    type=["pdf"],
    accept_multiple_files=True,
    help="These PDFs will be used in local RAG (not yet implemented)."
)

# TODO: Save uploaded PDFs into data/pdfs_uploaded/ and index them.

# ============= Main Area =============
question = st.text_area(
    "Enter your research question or task:",
    placeholder=(
        "Examples:\n"
        "- Summarize the key contributions of my uploaded paper.\n"
        "- Compare methods for chest X-ray report generation.\n"
        "- Explain common evaluation metrics used in medical imaging.\n"
    ),
    height=150,
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question before asking.")
    else:
        params = {"temperature": float(temperature), "max_tokens": int(max_tokens)}
        answer, refs, logs = run_research_query(
            question, source_mode, params, debug_mode
        )

        st.markdown("## 📘 Answer")
        st.write(answer)

        st.markdown("## 📚 References")
        for ref in refs:
            st.write(ref)

        # Download PDF
        pdf_bytes = convert_to_pdf(answer)
        st.download_button(
            "⬇ Download Answer as PDF",
            pdf_bytes,
            file_name="research_answer.pdf",
            mime="application/pdf",
        )

        # Simple copy hint (true clipboard copy requires JS hack)
        st.info("You can select and copy the answer text above.")

        if debug_mode:
            st.markdown("## 🧪 Debug (retrieval results placeholder)")
            st.write(logs)
'''

TEST_RAG_LOCAL_PY = r'''
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
'''

TEST_PIPELINE_BASIC_PY = r'''
"""
Basic pipeline test to ensure run_research_query returns expected structure.
"""

from app.pipeline.orchestrator import run_research_query

def test_pipeline_basic():
    answer, refs, logs = run_research_query(
        "What is this system supposed to do?", "hybrid", {"temperature": 0.3, "max_tokens": 500}, False
    )
    assert isinstance(answer, str)
    assert isinstance(refs, list)
    assert isinstance(logs, list)
'''

README_MD = r'''
# LLM-Powered Research Assistant (CrewAI + RAG)

This project implements a research assistant that:
- Answers research questions using local PDFs and/or web search (RAG).
- Produces structured academic-style answers with inline numeric citations and a References section.
- Provides a Streamlit-based UI with temperature/max_tokens control, PDF upload, logging, and export.

This scaffold was generated and will be progressively filled with:
- Real PDF parsing
- Embeddings and vector store
- CrewAI multi-agent workflow
- Web search integration
- Proper citation handling
'''

RUN_STREAMLIT_PY = r'''
"""
Entry script to launch the Streamlit UI.

Usage:
    streamlit run app/ui/ui_streamlit.py
"""
'''

# =========================
# MAIN
# =========================

def main():
    # Core app files
    write_file("app/config.py", CONFIG_PY)
    write_file("app/logging_config.py", LOGGING_CONFIG_PY)
    write_file("app/agents.py", AGENTS_PY)
    write_file("app/tasks.py", TASKS_PY)

    # Tools
    write_file("app/tools/pdf_loader.py", PDF_LOADER_PY)
    write_file("app/tools/vector_store.py", VECTOR_STORE_PY)
    write_file("app/tools/rag_local.py", RAG_LOCAL_PY)
    write_file("app/tools/rag_web.py", RAG_WEB_PY)
    write_file("app/tools/hybrid_retrieval.py", HYBRID_RETRIEVAL_PY)
    write_file("app/tools/export_utils.py", EXPORT_UTILS_PY)

    # Pipeline
    write_file("app/pipeline/orchestrator.py", ORCHESTRATOR_PY)

    # UI
    write_file("app/ui/ui_streamlit.py", UI_STREAMLIT_PY)

    # Tests
    write_file("tests/test_rag_local.py", TEST_RAG_LOCAL_PY)
    write_file("tests/test_pipeline_basic.py", TEST_PIPELINE_BASIC_PY)

    # Root files
    write_file("README.md", README_MD)
    write_file("run_streamlit.py", RUN_STREAMLIT_PY)

if __name__ == "__main__":
    main()
