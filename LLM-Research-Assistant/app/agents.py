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
