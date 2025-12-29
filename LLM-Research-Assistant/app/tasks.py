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
