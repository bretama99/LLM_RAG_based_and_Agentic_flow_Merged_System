"""
orchestrator.py
Main logic for running the research assistant workflow.

Now includes:
- Local RAG via LangChain (InMemoryVectorStore + embeddings)
- Web RAG via Tavily
- LLM writer using Ollama (llama3.2) to produce academic-style answers
  grounded in retrieved context with simple numeric citations [1], [2], ...
- Reflection step: critique + improved answer
  (critique logged + exposed only in debug UI)
- Simple task planner: qa / summary / compare / plan / survey
"""

import logging
import os
import re
from typing import Tuple, List, Dict, Any

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from app.logging_config import setup_logging
from app.config import LLM_MODEL
from app.tools.local_rag_langchain import local_search
from app.tools.rag_web import rag_web

# Initialize logging once
setup_logging()
LOGGER = logging.getLogger(__name__)

# -------------------------------------------------------------------
# WRITER & REFLECTOR PROMPTS (ACADEMIC + REFLECTION)
# -------------------------------------------------------------------

WRITER_SYSTEM_PROMPT = (
    "You are an academic research assistant. "
    "You write well-structured, concise answers suitable for a research context. "
    "You MUST base your answers ONLY on the provided context. "
    "If the context does not contain enough information, explicitly say so and "
    "do not invent facts. Use a neutral, formal tone. "
    "Use inline numeric citations like [1], [2] that refer to the numbered "
    "evidence segments in the context. "
    "IMPORTANT: Do NOT include a separate 'References' or 'Bibliography' "
    "section in your answer. Only use inline citations [1], [2], [3], etc. "
    "The system will handle the final reference list."
)

# ---- Default QA template (your original 5-section structure) ----
WRITER_USER_TEMPLATE = """
Context (numbered evidence snippets):
{context}

Question:
{question}

Write the answer using the following sections (in this exact order and numbering):
1. Introduction
2. Key Concepts
3. Evidence from Context
4. Limitations / Open Questions
5. Conclusion

If the context does not actually discuss the main topic of the question,
you MUST clearly say that the context does not cover this topic and that
you cannot provide a substantive answer.

Use inline references like [1], [2] that refer to the numbered evidence above
when appropriate. If the context is insufficient, clearly say so in the answer.
"""

# ---- Summary template ----
WRITER_USER_TEMPLATE_SUMMARY = """
Context (numbered evidence snippets):
{context}

Question:
{question}

Write a concise summary based ONLY on the context. Use the following sections:

1. Main Topic
2. Key Contributions / Findings
3. Methods / Approaches (if mentioned)
4. Limitations
5. Takeaway

If the context does not actually discuss the main topic of the question,
you MUST clearly say that the context does not cover this topic and that
you cannot provide a substantive summary.

Use inline references like [1], [2] that refer to the numbered evidence above
when appropriate.
"""

# ---- Compare template ----
WRITER_USER_TEMPLATE_COMPARE = """
Context (numbered evidence snippets):
{context}

Question:
{question}

Write a comparison-style answer based ONLY on the context, using these sections:

1. Compared Items and Context
2. Similarities
3. Differences
4. When to Prefer Each
5. Limitations / Open Questions

If the context does not actually compare the requested items,
you MUST clearly say that the context does not cover this comparison and that
you cannot provide a substantive answer.

Use inline references like [1], [2] that refer to the numbered evidence above.
"""

# ---- Plan template ----
WRITER_USER_TEMPLATE_PLAN = """
Context (numbered evidence snippets):
{context}

Question:
{question}

The user is asking for a research plan or project plan.
Using ONLY the context, draft a high-level research plan with these sections:

1. Background & Motivation
2. Research Questions / Objectives
3. Related Work from Context
4. Data / Datasets (if mentioned)
5. Methods & Baselines
6. Evaluation & Metrics
7. High-Level Timeline / Next Steps

If the context does not give enough information to propose a grounded plan,
you MUST clearly say which parts are missing (e.g., no dataset details, no methods)
and avoid inventing specifics.

Use inline references like [1], [2] for any grounded statements.
"""

# ---- Survey template ----
WRITER_USER_TEMPLATE_SURVEY = """
Context (numbered evidence snippets):
{context}

Question:
{question}

Write a short literature-style survey based ONLY on the context, with sections:

1. Area & Motivation
2. Main Research Themes / Directions
3. Representative Works from Context
4. Gaps / Open Problems
5. Conclusion

If the context does not provide enough material for a survey,
you MUST clearly say so and avoid generic or invented research directions.

Use inline references like [1], [2] for all claims based on the context.
"""

REFLECTOR_SYSTEM_PROMPT = (
    "You are a critical reviewer of research answers. "
    "You receive a question, a context, and a draft answer. "
    "Your job is to:\n"
    "1) Critique the draft (clarity, completeness, grounding in context, possible hallucinations).\n"
    "2) Then produce an improved final answer that is fully grounded in the context.\n"
    "If information is missing in the context, the improved answer must say so explicitly.\n\n"
    "STRUCTURE CONSTRAINT:\n"
    "- The draft answer is organized into sections with headings (for example numbered "
    "sections like '1. Introduction', '2. Key Concepts', etc.).\n"
    "- Your IMPROVED ANSWER MUST KEEP THE SAME SECTION HEADINGS AND ORDER as the draft. "
    "Do NOT add, remove, or rename sections. Improve only the content inside each section.\n\n"
    "REFERENCES RULE:\n"
    "- In your improved answer, do NOT include a separate 'References' or "
    "'Bibliography' section. Only use inline numeric citations [1], [2], etc. "
    "The system will produce the final reference list.\n\n"
    "If the context does not talk about the topic of the question at all, "
    "your improved answer MUST clearly say that the context does not cover this "
    "topic and avoid giving generic definitions or external facts."
)

REFLECTOR_USER_TEMPLATE = """
Question:
{question}

Context (numbered evidence snippets):
{context}

Draft answer:
{draft}

First, provide a short Critique section listing 3–6 bullet points about the draft
(what is good, what is missing, where it might be hallucinating, etc.).

Then, provide an Improved Answer section that fixes these issues.

Your Improved Answer MUST:
- Keep the same section headings and order as the draft.
- Only improve the content within each section, without adding new sections
  or removing any of the existing ones.

Use the following structure exactly:

Critique:
- ...

Improved Answer:
[write the refined answer here, preserving the same headings and order as the draft]
"""

# -------------------------------------------------------------------
# Task planner (light rule-based)
# -------------------------------------------------------------------

def _plan_task_type(question: str) -> str:
    """
    Very simple rule-based planner that maps the user question
    to a task type: 'qa', 'summary', 'compare', 'plan', or 'survey'.
    Default is 'qa'.
    """
    q = question.lower()

    # research / thesis / project plan
    if "research plan" in q or "thesis plan" in q or "project plan" in q or "roadmap" in q:
        return "plan"

    # summarization
    if "summarize" in q or "summary" in q or "in 5 points" in q or "briefly summarize" in q:
        return "summary"

    # comparison
    if "compare" in q or "difference between" in q or "vs." in q or "versus" in q:
        return "compare"

    # survey / overview of directions
    if "recent research" in q or "research directions" in q or "state of the art" in q or "literature overview" in q:
        return "survey"

    # fallback
    return "qa"

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def shortname(path: str) -> str:
    """Return only the filename from a full path."""
    if not path:
        return "Unknown source"
    return os.path.basename(path)


def _build_context_and_references(
    retrieval_results: List[Dict[str, Any]]
) -> Tuple[str, List[str]]:
    """
    Build a formatted context string for the LLM and a reference list
    for the UI from retrieval_results.

    Each retrieved item has:
    - 'text': chunk text
    - 'metadata': info like page, source path, url, etc.

    Returns:
      context_str: numbered context segments for the LLM
      references: ordered list of clean reference strings
    """

    segments: List[str] = []
    references: List[str] = []
    seen_refs: List[str] = []

    for idx, item in enumerate(retrieval_results, start=1):
        text = item.get("text", "")
        meta = item.get("metadata", {}) or {}

        source_path = meta.get("source", "")
        filename = shortname(source_path)
        page = meta.get("page")
        url = meta.get("url")

        ref_parts: List[str] = []

        if filename:
            ref_parts.append(filename)

        if isinstance(page, int):
            ref_parts.append(f"p. {page + 1}")

        if url:
            ref_parts.append(url)

        if not ref_parts:
            ref_parts.append("Unknown source")

        ref_str = f"[{idx}] " + ", ".join(ref_parts)

        segment_header = f"[{idx}] {ref_str}"
        segment_body = text.strip()
        segments.append(f"{segment_header}\n{segment_body}\n")

        if ref_str not in seen_refs:
            seen_refs.append(ref_str)
            references.append(ref_str)

    context_str = "\n---\n".join(segments) if segments else "NO_CONTEXT_AVAILABLE"
    return context_str, references


def _build_llm(temperature: float, max_tokens: int) -> ChatOllama:
    """
    Create a ChatOllama instance using the configured model.
    """
    LOGGER.info("Creating ChatOllama model: %s", LLM_MODEL)
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=temperature,
        num_predict=max_tokens,
    )
    return llm


def _strip_llm_references_and_critique(answer_text: str) -> str:
    """
    Clean the final answer so that:
      - It does NOT contain its own 'References:' or 'Bibliography:' section.
      - It does NOT contain a leading 'Critique:' block.
      - It does NOT contain the label 'Improved Answer:'.
    """
    if not answer_text:
        return answer_text

    cleaned = answer_text.strip()

    # Remove leading "Critique: ... Improved Answer:"
    cleaned = re.sub(
        r"^\s*Critique\s*:.*?(Improved Answer\s*:)",
        r"\1",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove "Improved Answer:" label at the start
    cleaned = re.sub(
        r"^\s*Improved Answer\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Strip trailing reference sections
    patterns = [
        r"\n\s*References\s*:\s*[\s\S]*$",
        r"\n\s*Reference\s*:\s*[\s\S]*$",
        r"\n\s*Bibliography\s*:\s*[\s\S]*$",
    ]
    for pattern in patterns:
        cleaned_new = re.sub(
            pattern,
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if cleaned_new != cleaned:
            LOGGER.info("Stripped trailing reference section from LLM answer.")
            cleaned = cleaned_new

    return cleaned.strip()


def _format_headings_markdown(text: str) -> str:
    """
    Make section titles bold and move their content to a new line.

    Example:
      '1. Introduction This is...' ->
      '**1. Introduction**\\nThis is...'

    NOTE: This is tuned for the QA 5-section template.
    Other templates may not fully match these headings, which is fine.
    """
    if not text:
        return text

    sections = [
        "Introduction",
        "Key Concepts",
        "Evidence from Context",
        "Limitations / Open Questions",
        "Conclusion",
    ]

    for idx, title in enumerate(sections, start=1):
        pattern = rf"{idx}\.\s+{re.escape(title)}(.*)"
        replacement = rf"**{idx}. {title}**\n\1"
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )

    return text


def _format_critique_markdown(text: str) -> str:
    """
    Make the critique easier to read:
    - put 'Critique:' on its own line
    - turn '•' bullets into markdown '- ' bullets on separate lines
    """
    if not text:
        return text

    # Ensure "Critique:" is followed by a newline
    text = re.sub(
        r"Critique\s*:\s*",
        "Critique:\n",
        text,
        count=1,
        flags=re.IGNORECASE,
    )

    # Each '•' becomes a markdown bullet on its own line
    text = text.replace("•", "\n- ")

    return text.strip()


def _run_writer_and_reflector(
    question: str,
    context_str: str,
    llm_params: dict,
    task_type: str = "qa",
) -> Tuple[str, str, str]:
    """
    Run the writer LLM (academic style) and then a reflection step
    that produces a critique + improved answer.

    Returns:
      final_answer (cleaned),
      draft_answer,
      critique_text
    """

    temperature = float(llm_params.get("temperature", 0.3))
    max_tokens = int(llm_params.get("max_tokens", 1200))

    # 1) Writer: draft answer
    writer_llm = _build_llm(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Choose user template based on task_type
    if task_type == "summary":
        user_template = WRITER_USER_TEMPLATE_SUMMARY
    elif task_type == "compare":
        user_template = WRITER_USER_TEMPLATE_COMPARE
    elif task_type == "plan":
        user_template = WRITER_USER_TEMPLATE_PLAN
    elif task_type == "survey":
        user_template = WRITER_USER_TEMPLATE_SURVEY
    else:
        # default QA behaviour (your original 5-section template)
        user_template = WRITER_USER_TEMPLATE

    writer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", WRITER_SYSTEM_PROMPT),
            ("user", user_template),
        ]
    )
    writer_chain = writer_prompt | writer_llm

    LOGGER.info("Calling writer LLM with academic prompt...")
    writer_response = writer_chain.invoke(
        {
            "context": context_str,
            "question": question,
        }
    )

    draft_answer = getattr(writer_response, "content", str(writer_response)).strip()
    LOGGER.info(
        "Writer draft answer generated (first 200 chars): %s",
        draft_answer[:200],
    )

    # 2) Reflector: critique + improved answer
    reflector_llm = writer_llm  # reuse same model
    reflector_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REFLECTOR_SYSTEM_PROMPT),
            ("user", REFLECTOR_USER_TEMPLATE),
        ]
    )
    reflector_chain = reflector_prompt | reflector_llm

    LOGGER.info("Starting reflection step (critique + improved answer)...")
    reflector_response = reflector_chain.invoke(
        {
            "context": context_str,
            "question": question,
            "draft": draft_answer,
        }
    )
    reflector_text = getattr(reflector_response, "content", str(reflector_response))

    critique = ""
    improved_answer = ""

    split_marker = "Improved Answer:"
    if split_marker in reflector_text:
        parts = reflector_text.split(split_marker, 1)
        critique_part = parts[0].strip()
        improved_part = parts[1].strip()

        if critique_part.lower().startswith("critique"):
            critique = critique_part
        else:
            critique = "Critique:\n" + critique_part

        improved_answer = improved_part
    else:
        LOGGER.warning(
            "Reflection output did not contain 'Improved Answer:' marker. "
            "Using draft answer as final answer. Logging full reflection text as critique."
        )
        critique = reflector_text.strip()
        improved_answer = draft_answer

    if critique:
        critique = _format_critique_markdown(critique)
        LOGGER.info("Reflection critique:\n%s", critique)

    final_answer = improved_answer.strip() if improved_answer else draft_answer
    final_answer = _strip_llm_references_and_critique(final_answer)
    final_answer = _format_headings_markdown(final_answer)

    LOGGER.info(
        "Reflection step completed. Final answer length: %d characters.",
        len(final_answer),
    )

    return final_answer, draft_answer, critique


# -------------------------------------------------------------------
# Main public function
# -------------------------------------------------------------------

def run_research_query(
    question: str,
    source_mode: str,
    llm_params: dict,
    debug_mode: bool = False,
) -> Tuple[str, List[str], List[dict], Dict[str, str]]:
    """
    Orchestrate the end-to-end pipeline:
    - Local RAG via LangChain (InMemoryVectorStore + embeddings)
    - Web RAG via Tavily
    - Hybrid = local + web
    - LLM writer via Ollama to generate academic-style answer with citations
    - Reflection step for critique + improved answer

    Returns:
      final_answer,
      references,
      retrieval_results,
      debug_data = {
        "task_type": ...,
        "draft_answer": ...,
        "critique": ...
      }
    """

    LOGGER.info("========================================")
    LOGGER.info("Received user query: %s", question)
    LOGGER.info("Selected source mode: %s", source_mode)
    LOGGER.info("LLM params: %s", llm_params)
    LOGGER.info("========================================")

    # STEP 1 — Task Planning (simple rule-based)
    LOGGER.info("Starting task planning...")
    task_type = _plan_task_type(question)
    LOGGER.info("Planner determined task type: %s", task_type)

    # STEP 2 — Retrieval
    LOGGER.info("Starting retrieval...")
    retrieval_results: List[Dict[str, Any]] = []

    if source_mode in ("local", "hybrid"):
        LOGGER.info("Performing LOCAL semantic retrieval via LangChain...")
        local_results = local_search(question, k=5)
        retrieval_results.extend(local_results)

    if source_mode in ("web", "hybrid"):
        LOGGER.info("Performing WEB retrieval via Tavily Web RAG...")
        web_results = rag_web(question, k=5)
        retrieval_results.extend(web_results)

    LOGGER.info("Retrieval completed. Retrieved items: %d", len(retrieval_results))

    if debug_mode:
        LOGGER.info("Debug mode ON: retrieval results will be returned to UI.")

    # STEP 3 — Build context + reference list
    LOGGER.info("Building context for LLM from retrieval results...")
    context_str, references_from_context = _build_context_and_references(
        retrieval_results
    )

    draft_for_debug = ""
    critique_for_debug = ""

    # STEP 4 — Writing + Reflection
    if context_str == "NO_CONTEXT_AVAILABLE":
        LOGGER.info("No context available; returning fallback answer.")
        final_answer = (
            "I could not find any relevant context in the indexed documents "
            "or web results for your question. Please upload papers or enable "
            "web search with appropriate documents."
        )
        references: List[str] = []
    else:
        final_answer, draft_for_debug, critique_for_debug = _run_writer_and_reflector(
            question=question,
            context_str=context_str,
            llm_params=llm_params,
            task_type=task_type,
        )
        references = references_from_context

    debug_data = {
        "task_type": task_type,
        "draft_answer": draft_for_debug,
        "critique": _format_critique_markdown(critique_for_debug),
    }

    LOGGER.info("Pipeline completed successfully.")
    LOGGER.info("========================================\n")

    return final_answer, references, retrieval_results, debug_data
