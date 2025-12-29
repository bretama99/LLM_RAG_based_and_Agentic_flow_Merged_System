"""
ui_streamlit.py
Streamlit UI for the LLM-Powered Research Assistant.
"""

import os
import streamlit as st

from app.pipeline.orchestrator import run_research_query
from app.config import PDF_UPLOADED_DIR
from app.tools.local_rag_langchain import ingest_pdfs
from app.tools.export_utils import convert_to_pdf


# ---------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="LLM-Powered Research Assistant",
    layout="wide",
)

# ---------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []
if "indexed_files" not in st.session_state:
    st.session_state["indexed_files"] = []
if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = ""
if "last_references" not in st.session_state:
    st.session_state["last_references"] = []
if "last_retrieval" not in st.session_state:
    st.session_state["last_retrieval"] = []
if "last_question" not in st.session_state:
    st.session_state["last_question"] = ""
if "last_debug" not in st.session_state:
    st.session_state["last_debug"] = {}
if "last_source_mode" not in st.session_state:
    st.session_state["last_source_mode"] = "local"


# ---------------------------------------------------------------------
# Sidebar – settings
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    source_mode = st.radio(
        "Sources to use:",
        options=["local", "web", "hybrid"],
        index=0,
        help="Choose where the assistant should retrieve information from.",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
    )

    max_tokens = st.number_input(
        "Max tokens",
        min_value=128,
        max_value=4096,
        value=800,
        step=64,
    )

    debug_mode = st.checkbox("Show logs & internals (debug mode)", value=False)

    st.markdown("---")
    st.subheader("Upload PDFs (optional)")

    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Save uploaded files to disk
    saved_paths = []
    if uploaded_files:
        os.makedirs(PDF_UPLOADED_DIR, exist_ok=True)
        for uf in uploaded_files:
            save_path = os.path.join(PDF_UPLOADED_DIR, uf.name)
            with open(save_path, "wb") as f:
                f.write(uf.getbuffer())
            saved_paths.append(save_path)

        if saved_paths:
            st.session_state["uploaded_files"] = saved_paths
            st.success(f"{len(saved_paths)} file(s) saved.")

    # Button to ingest PDFs into local RAG
    if st.button("Ingest PDFs into Local RAG (LangChain)"):
        if not st.session_state["uploaded_files"]:
            st.warning("No PDFs uploaded yet.")
        else:
            with st.spinner("Starting semantic ingestion (LangChain)..."):
                try:
                    ingest_pdfs(st.session_state["uploaded_files"])
                    st.success("Ingestion completed. Local LangChain RAG index updated.")
                except Exception as e:
                    st.error(f"Error during ingestion: {e}")


# ---------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------
st.title("🔍 LLM-Powered Research Assistant")
st.write(
    "Ask research questions, summarize or compare papers, and get structured answers "
    "with citations. This assistant uses LangChain RAG, Tavily Web Search, and Reflection."
)

st.markdown("### Enter your research question or task:")
user_question = st.text_area(
    "User Question",
    placeholder="Enter your research question or task...",
    label_visibility="collapsed",
    key="user_question",
    height=150,
)

ask_clicked = st.button("Ask")

answer = ""
references: list[str] = []
retrieval_results: list[dict] = []
debug_data: dict = {}

if ask_clicked and user_question.strip():
    with st.spinner("Thinking..."):
        try:
            answer, references, retrieval_results, debug_data = run_research_query(
                question=user_question.strip(),
                source_mode=source_mode,
                llm_params={"temperature": temperature, "max_tokens": max_tokens},
                debug_mode=debug_mode,
            )
            st.session_state["last_answer"] = answer
            st.session_state["last_references"] = references
            st.session_state["last_retrieval"] = retrieval_results
            st.session_state["last_question"] = user_question.strip()
            st.session_state["last_debug"] = debug_data
            st.session_state["last_source_mode"] = source_mode

        except Exception as e:
            st.error(f"Error while running pipeline: {e}")
else:
    answer = st.session_state.get("last_answer", "")
    references = st.session_state.get("last_references", [])
    retrieval_results = st.session_state.get("last_retrieval", [])
    debug_data = st.session_state.get("last_debug", {})


# ---------------------------------------------------------------------
# Answer display (always clean – NO critique and NO LLM references)
# ---------------------------------------------------------------------
st.markdown("## 📘 Answer")

if answer:
    st.write(answer)
else:
    st.info("Ask a question to get a structured academic-style answer.")


# ---------------------------------------------------------------------
# References display
# ---------------------------------------------------------------------
st.markdown("## 📚 References")

if references:
    for ref in references:
        st.write(ref)
else:
    st.write("No references produced yet.")


# ---------------------------------------------------------------------
# PDF download (Question + Answer + Clean References)
# ---------------------------------------------------------------------
if answer:
    question_text = st.session_state.get("last_question", "")

    pdf_text = ""

    if question_text:
        pdf_text += "Question:\n"
        pdf_text += question_text + "\n\n"

    pdf_text += "Answer:\n"
    pdf_text += answer

    if references:
        pdf_text += "\n\nReferences:\n"
        for ref in references:
            pdf_text += f"{ref}\n"

    pdf_bytes = convert_to_pdf(pdf_text)

    st.download_button(
        label="⬇️ Download Answer as PDF",
        data=pdf_bytes,
        file_name="research_answer.pdf",
        mime="application/pdf",
    )

st.markdown("You can select and copy the answer text above.")


# ---------------------------------------------------------------------
# Debug panel – writer draft, critique, raw retrieval
# ---------------------------------------------------------------------
if debug_mode:
    st.markdown("## 🧪 Debug – LLM Internals")

    # Writer draft
    with st.expander("✏️ Writer Draft Answer", expanded=False):
        draft = debug_data.get("draft_answer") or ""
        if draft.strip():
            st.write(draft)
        else:
            st.write("No draft answer available (run a query first).")

    # Reflection critique
    with st.expander("🤯 Reflection Critique", expanded=False):
        critique = debug_data.get("critique") or ""
        if critique.strip():
            st.write(critique)
        else:
            st.write("No reflection critique available.")

    # Raw retrieval
    if retrieval_results:
        # Try to infer whether results are local / web
        has_web = any((item.get("metadata") or {}).get("url") for item in retrieval_results)
        has_local = any(not (item.get("metadata") or {}).get("url") for item in retrieval_results)

        source_labels = []
        if has_local:
            source_labels.append("local")
        if has_web:
            source_labels.append("web")

        if not source_labels:
            # fallback to last_source_mode
            source_labels.append(st.session_state.get("last_source_mode", "local"))

        label = " + ".join(source_labels)

        with st.expander(f"📂 Raw Retrieval Results ({label})", expanded=False):
            st.json(retrieval_results)
