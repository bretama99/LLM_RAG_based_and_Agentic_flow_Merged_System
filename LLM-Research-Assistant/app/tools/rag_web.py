"""
rag_web.py
Web RAG retrieval using the Tavily Search API.

This module exposes:
    rag_web(query: str, k: int = 5) -> List[Dict[str, Any]]

It returns results in the same format as local RAG:
    {"text": "...", "metadata": {...}}

So that the orchestrator can treat local + web results uniformly.
"""

import logging
import os
from typing import List, Dict, Any

import requests

LOGGER = logging.getLogger(__name__)

TAVILY_API_URL = "https://api.tavily.com/search"


def _call_tavily_api(query: str, k: int = 5) -> Dict[str, Any]:
    """
    Call Tavily's search API and return the parsed JSON.

    We assume TAVILY_API_KEY is set in the environment:
        setx TAVILY_API_KEY "tvly-dev-...."
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        LOGGER.warning("TAVILY_API_KEY is not set; web RAG is disabled.")
        return {}

    payload = {
        "api_key": api_key,
        "query": query,
        # A bit deeper search for research-oriented queries
        "search_depth": "advanced",
        "max_results": k,
        # We only need text, not images or a separate answer
        "include_answer": False,
        "include_images": False,
    }

    try:
        LOGGER.info("[WEB RAG] Calling Tavily API...")
        resp = requests.post(TAVILY_API_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data
    except Exception as e:
        LOGGER.exception("Error while calling Tavily API: %s", e)
        return {}


def rag_web(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Perform web retrieval using Tavily and convert results into the
    generic RAG format expected by the orchestrator:

        [
            {
                "text": "<content snippet>",
                "metadata": {
                    "source": "web",
                    "url": "...",
                    "title": "...",
                    "rank": 1
                }
            },
            ...
        ]
    """
    LOGGER.info("[WEB RAG] Querying Tavily for: %s", query)
    data = _call_tavily_api(query, k=k)
    if not data:
        LOGGER.info("[WEB RAG] No data returned from Tavily.")
        return []

    raw_results = data.get("results") or []
    items: List[Dict[str, Any]] = []

    for idx, r in enumerate(raw_results[:k], start=1):
        # Tavily usually returns "content" as a text snippet / extracted content
        content = (r.get("content") or "").strip()
        # If for some reason content is empty, skip that result
        if not content:
            continue

        title = r.get("title") or ""
        url = r.get("url") or ""

        metadata = {
            "source": "web",
            "url": url,
            "title": title,
            "rank": idx,
        }

        items.append(
            {
                "text": content,
                "metadata": metadata,
            }
        )

    LOGGER.info("[WEB RAG] Retrieved %d web result(s) from Tavily.", len(items))
    return items
