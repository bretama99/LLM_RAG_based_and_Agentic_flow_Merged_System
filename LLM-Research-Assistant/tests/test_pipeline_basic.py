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
