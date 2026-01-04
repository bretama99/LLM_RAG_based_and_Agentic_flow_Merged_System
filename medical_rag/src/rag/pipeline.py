# src/rag/pipeline.py
"""RAG pipeline: PDF indexing + hybrid retrieval + strict evidence gating + optional whitelisted web."""
from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.config import load_config, Config
from src.llm.ollama_client import ask_ollama
from src.prompting.prompts import (
    SYSTEM, USER,
    MULTIQUERY_SYSTEM, MULTIQUERY_USER,
    EVIDENCE_CHECK_SYSTEM, EVIDENCE_CHECK_USER,
    WEB_SUMMARY_SYSTEM,
)

# Optional deps (handled gracefully where used)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None  # type: ignore

# -----------------------------------------------------------------------------
# Globals / caches
# -----------------------------------------------------------------------------
_CFG: Optional[Config] = None
_DENSE_INDEX: Optional["DenseIndex"] = None
_DOCS: Optional[List[Dict[str, Any]]] = None
_BM25: Optional["BM25Okapi"] = None  # type: ignore

# Tavily client cache (safe micro-speedup)
_TAVILY_CLIENT = None


def _get_cfg() -> Config:
    global _CFG
    if _CFG is None:
        _CFG = load_config()
    return _CFG


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


# -----------------------------------------------------------------------------
# PDF loading + chunking
# -----------------------------------------------------------------------------
def load_pdfs(pdf_paths: List[str]) -> List[Dict[str, Any]]:
    """Load PDFs into per-page docs: {source, page, text}."""
    docs: List[Dict[str, Any]] = []
    try:
        import pdfplumber
    except Exception:
        return docs

    for p in pdf_paths or []:
        path = str(p)
        if not path or not os.path.exists(path):
            continue
        try:
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    txt = (page.extract_text() or "").strip()
                    if txt:
                        docs.append({"source": os.path.basename(path), "page": i, "text": txt})
        except Exception:
            continue
    return docs


def chunk_docs(docs: List[Dict[str, Any]], chunk_size: int = 900, chunk_overlap: int = 120) -> List[Dict[str, Any]]:
    """Chunk per-page docs into overlapping character windows."""
    out: List[Dict[str, Any]] = []
    cs = max(200, int(chunk_size or 900))
    ov = max(0, int(chunk_overlap or 0))
    step = max(50, cs - ov)

    for d in docs or []:
        text = (d.get("text") or "").strip()
        if not text:
            continue
        src = d.get("source", "unknown.pdf")
        page = d.get("page", "?")

        start = 0
        n = len(text)
        while start < n:
            end = min(n, start + cs)
            chunk = text[start:end].strip()
            if chunk:
                out.append({"source": src, "page": page, "start": start, "end": end, "text": chunk})
            if end >= n:
                break
            start += step

    return out


# -----------------------------------------------------------------------------
# Dense index (persisted)
# -----------------------------------------------------------------------------
class DenseIndex:
    def __init__(self, persist_dir: str, embedding_model: str):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed")

        self.persist_dir = Path(persist_dir)
        self.embedding_model = embedding_model
        self.docs_path = self.persist_dir / "docs.jsonl"
        self.emb_path = self.persist_dir / "embeddings.npy"

        self.embedder = SentenceTransformer(embedding_model)
        self._docs: Optional[List[Dict[str, Any]]] = None
        self._embeddings: Optional[np.ndarray] = None

    @property
    def docs(self) -> List[Dict[str, Any]]:
        if self._docs is None:
            self.load()
        return self._docs or []

    def _load_docs(self) -> None:
        docs: List[Dict[str, Any]] = []
        if self.docs_path.exists():
            with self.docs_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            docs.append(json.loads(line))
                        except Exception:
                            continue
        self._docs = docs

    def _load_embeddings(self) -> None:
        if self.emb_path.exists():
            self._embeddings = np.load(self.emb_path)
        else:
            self._embeddings = np.zeros((0, 384), dtype="float32")

    def build(self, chunks: List[Dict[str, Any]]) -> int:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        with self.docs_path.open("w", encoding="utf-8") as f:
            for ch in chunks:
                f.write(json.dumps(ch, ensure_ascii=False) + "\n")

        texts = [c.get("text", "") for c in chunks]
        embs = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self._embeddings = embs.astype("float32")
        np.save(self.emb_path, self._embeddings)
        self._docs = chunks
        return len(chunks)

    def load(self) -> None:
        self._load_docs()
        self._load_embeddings()

    def search_dense(self, query: str, top_k: int) -> Dict[int, float]:
        if self._embeddings is None or self._docs is None:
            self.load()
        if self._embeddings is None or self._docs is None or len(self._embeddings) == 0:
            return {}
        q_emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        scores = np.dot(self._embeddings, q_emb)
        top_k = min(max(int(top_k), 1), len(scores))
        idxs = np.argpartition(-scores, top_k - 1)[:top_k]
        idxs = idxs[np.argsort(-scores[idxs])]
        return {int(i): float(scores[i]) for i in idxs}


def _get_dense_index() -> DenseIndex:
    global _DENSE_INDEX
    if _DENSE_INDEX is None:
        cfg = _get_cfg()
        _DENSE_INDEX = DenseIndex(cfg.rag.persist_dir, cfg.rag.embedding_model)
    return _DENSE_INDEX


def _get_docs() -> List[Dict[str, Any]]:
    global _DOCS
    if _DOCS is None:
        _DOCS = _get_dense_index().docs
    return _DOCS


def _ensure_bm25() -> Optional["BM25Okapi"]:
    global _BM25
    if _BM25 is None:
        if BM25Okapi is None:
            return None
        docs = _get_docs()
        if not docs:
            return None
        corpus = [_tokenize(d.get("text", "")) for d in docs]
        _BM25 = BM25Okapi(corpus)
    return _BM25


# -----------------------------------------------------------------------------
# Retrieval
# -----------------------------------------------------------------------------
def _make_queries(question: str, cfg: Config) -> List[str]:
    """Multi-query expansion."""
    q = (question or "").strip()
    if not q or not cfg.retrieval.use_multi_query:
        return [q] if q else []
    n = max(1, int(getattr(cfg.retrieval, "num_query_variants", 1) or 1))
    raw = ask_ollama(
        system=MULTIQUERY_SYSTEM,
        user=MULTIQUERY_USER.format(question=q, n=n),
        model=cfg.app.model,
        temperature=0.2,
        num_predict=200,
        raise_on_fail=False,
    )
    if not (raw or "").strip():
        return [q]
    lines = [re.sub(r"^\s*\d+[\).\s]+", "", line).strip() for line in raw.splitlines() if line.strip()]
    out, seen = [q], {q.lower()}
    for x in lines:
        if x.lower() not in seen:
            out.append(x)
            seen.add(x.lower())
        if len(out) >= (n + 1):
            break
    return out


def hybrid_retrieve(question: str, cfg: Config) -> List[Dict[str, Any]]:
    """Hybrid BM25 + dense retrieval."""
    docs = _get_docs()
    if not docs:
        return []
    top_k = max(1, int(cfg.retrieval.top_k or 8))
    mode = (cfg.retrieval.mode or "hybrid").lower()

    dense_scores = _get_dense_index().search_dense(question, top_k * 2) if mode in ("dense", "hybrid") else {}

    bm25_scores: Dict[int, float] = {}
    if mode in ("bm25", "hybrid"):
        bm25 = _ensure_bm25()
        if bm25:
            q_tokens = _tokenize(question)
            scores_arr = bm25.get_scores(q_tokens)
            if len(scores_arr) > 0:
                top_idxs = np.argpartition(-scores_arr, min(top_k * 2, len(scores_arr)) - 1)[: top_k * 2]
                bm25_scores = {int(i): float(scores_arr[i]) for i in top_idxs if scores_arr[i] > 0}

    if mode == "hybrid":
        alpha = float(getattr(cfg.retrieval, "hybrid_alpha", 0.6))
        all_ids = set(dense_scores.keys()) | set(bm25_scores.keys())
        fused = {i: alpha * dense_scores.get(i, 0) + (1 - alpha) * bm25_scores.get(i, 0) for i in all_ids}
    elif mode == "bm25":
        fused = bm25_scores
    else:
        fused = dense_scores

    sorted_ids = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)[:top_k]
    return [docs[i] for i in sorted_ids if i < len(docs)]


def _group_chunks_by_pdf(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for ch in chunks:
        src = ch.get("source") or "unknown.pdf"
        grouped.setdefault(src, []).append(ch)
    return [{"source": src, "matches": arr} for src, arr in grouped.items()]


def _support_score(question: str, chunks: List[Dict[str, Any]]) -> float:
    """Cheap overlap-based score (for display / fallback)."""
    q_tokens = {t for t in _tokenize(question) if len(t) >= 4}
    if not q_tokens:
        return 0.0
    e = " ".join((c.get("text") or "") for c in (chunks or [])).lower()
    hits = sum(1 for t in q_tokens if t in e)
    return hits / max(1, len(q_tokens))


# -----------------------------------------------------------------------------
# Evidence checking + extractive fallbacks
# -----------------------------------------------------------------------------
def _heuristic_evidence_check(question: str, evidence: str) -> Tuple[bool, str]:
    q_tokens = {t for t in _tokenize(question) if len(t) >= 4}
    e_lower = evidence.lower()
    hits = sum(1 for t in q_tokens if t in e_lower)
    if hits >= max(1, len(q_tokens) * 0.4):
        return True, "Heuristic: sufficient overlap"
    return False, f"Heuristic: only {hits}/{len(q_tokens)} tokens matched"


def _evidence_check(question: str, evidence: str, cfg: Config) -> Tuple[bool, str]:
    raw = ask_ollama(
        system=EVIDENCE_CHECK_SYSTEM,
        user=EVIDENCE_CHECK_USER.format(question=question, evidence=evidence),
        model=cfg.app.model,
        temperature=0.0,
        num_predict=220,
        raise_on_fail=False,
    )
    if not (raw or "").strip():
        return _heuristic_evidence_check(question, evidence)
    try:
        obj = json.loads(raw)
        return bool(obj.get("found")), (obj.get("reason") or "").strip()
    except Exception:
        return _heuristic_evidence_check(question, evidence)


def _extractive_pdf_answer(question: str, ui_groups: List[Dict[str, Any]], max_items: int = 12) -> str:
    q_tokens = {t for t in _tokenize(question) if len(t) >= 4}
    scored = []
    for g in ui_groups:
        src = g.get("source", "unknown.pdf")
        for m in (g.get("matches") or [])[:6]:
            text = (m.get("text") or "")
            for s in re.split(r"(?<=[.!?])\s+", text):
                s = s.strip()
                if len(s) >= 35:
                    score = sum(1 for t in q_tokens if t in s.lower())
                    if score > 0:
                        scored.append((score, src, m.get("page", "?"), s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return "\n".join([f"- {sent} [{src}:p{page}]" for _, src, page, sent in scored[:max_items]]) if scored else ""


def _extractive_web_answer(question: str, web_sources: List[Dict[str, Any]], max_items: int = 10) -> str:
    q_tokens = {t for t in _tokenize(question) if len(t) >= 4}
    scored = []
    for r in (web_sources or [])[:10]:
        url = r.get("url") or r.get("title") or "WEB"
        for s in re.split(r"(?<=[.!?])\s+", (r.get("snippet") or "")):
            s = s.strip()
            if len(s) >= 35:
                score = sum(1 for t in q_tokens if t in s.lower())
                if score > 0:
                    scored.append((score, url, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return "\n".join([f"- {sent} [WEB:{url}]" for _, url, sent in scored[:max_items]]) if scored else ""


# -----------------------------------------------------------------------------
# Whitelisted web search
# -----------------------------------------------------------------------------
def _get_tavily_client():
    global _TAVILY_CLIENT
    if _TAVILY_CLIENT is None:
        import tavily
        api_key = os.environ.get("TAVILY_API_KEY", "").strip()
        _TAVILY_CLIENT = tavily.TavilyClient(api_key=api_key)
    return _TAVILY_CLIENT


def _in_whitelist(url: str, wl: List[str]) -> bool:
    import urllib.parse
    try:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
    except Exception:
        return False
    if not host:
        return False
    for d in (wl or []):
        dd = (d or "").lower().lstrip(".")
        if host == dd or host.endswith("." + dd):
            return True
    return False


def _web_search_whitelisted_resources(question: str, cfg: Config) -> Dict[str, Any]:
    """
    FAST: fetch-only whitelisted web resources (NO evidence-judge LLM call, NO web-summary LLM call).
    Returns the same resource objects as _web_search_whitelisted().
    """
    if not cfg.web.allow_web_fallback:
        return {"resources": [], "citations": [], "has_sources": False, "reason": "Web fallback disabled."}

    try:
        import tavily  # noqa: F401
    except ImportError:
        return {"resources": [], "citations": [], "has_sources": False, "reason": "tavily not installed."}

    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return {"resources": [], "citations": [], "has_sources": False, "reason": "Missing TAVILY_API_KEY."}

    wl = list(getattr(cfg.web, "domain_whitelist", []) or [])
    max_results = int(getattr(cfg.web, "tavily_top_k", 6) or 6)

    try:
        client = _get_tavily_client()
        results = client.search(
            query=question,
            search_depth="basic",
            topic="general",
            include_raw_content=False,
            include_answer=False,
            max_results=max_results,
            include_domains=wl if wl else None,
        )
    except Exception as e:
        return {"resources": [], "citations": [], "has_sources": False, "reason": f"Search failed: {e}"}

    raw_results = (results or {}).get("results", []) or []
    raw_results = [r for r in raw_results if _in_whitelist(r.get("url") or "", wl)]

    resources, citations = [], []
    for i, r in enumerate(raw_results, 1):
        url = r.get("url")
        title = r.get("title") or url or "Web Source"
        snippet = _clean(r.get("content") or r.get("raw_content") or "")
        if len(snippet) > 700:
            snippet = snippet[:700] + "."
        resources.append(
            {"kind": "web", "rank": i, "title": title, "url": url, "snippet": snippet, "citation": url or title}
        )
        if url:
            citations.append(url)

    if not resources:
        return {"resources": [], "citations": [], "has_sources": False, "reason": "No results found in whitelisted domains."}

    return {"resources": resources, "citations": citations, "has_sources": True, "reason": ""}


def _web_search_whitelisted(question: str, cfg: Config) -> Dict[str, Any]:
    """Original (full) web search: resources + evidence-judge + web-summary (kept for full functionality)."""
    base = _web_search_whitelisted_resources(question, cfg)
    resources = base.get("resources", []) or []
    citations = base.get("citations", []) or []
    has_sources = bool(base.get("has_sources"))
    if not has_sources:
        return {
            "answer": "",
            "resources": [],
            "citations": [],
            "has_sources": False,
            "answerable": False,
            "reason": base.get("reason", "No results found in whitelisted domains."),
        }

    # Build evidence text for judging + summarization
    evidence = "\n\n".join(
        f"[WEB-{i}] {r.get('title') or r.get('url')}\n{(r.get('snippet') or '')}"
        for i, r in enumerate(resources, 1)
    )
    evidence = _clean(evidence)[:6000]

    answerable, reason = _evidence_check(question, evidence, cfg)
    if not answerable:
        return {
            "answer": "",
            "resources": resources,
            "citations": citations,
            "has_sources": True,
            "answerable": False,
            "reason": reason or "Whitelisted sources did not contain enough evidence.",
        }

    web_answer = ask_ollama(
        system=WEB_SUMMARY_SYSTEM,
        user=(
            f"User question:\n{question}\n\n"
            f"Web snippets (whitelisted):\n{evidence[:9000]}\n\n"
            f"Synthesize answer with citations [WEB:url]."
        ),
        model=cfg.app.model,
        temperature=cfg.app.temperature,
        num_predict=cfg.app.max_new_tokens,
        raise_on_fail=False,
    ).strip()

    if web_answer and ("[WEB:" not in web_answer and "[WEB-" not in web_answer):
        web_answer = ""

    if not web_answer:
        fallback = _extractive_web_answer(question, resources)
        if fallback:
            return {
                "answer": "Extracted from whitelisted web sources (LLM unavailable):\n" + fallback,
                "resources": resources,
                "citations": citations,
                "has_sources": True,
                "answerable": True,
                "reason": "",
            }
        return {
            "answer": "",
            "resources": resources,
            "citations": citations,
            "has_sources": True,
            "answerable": True,
            "reason": "Whitelisted sources found, but LLM unavailable.",
        }

    return {"answer": web_answer, "resources": resources, "citations": citations, "has_sources": True, "answerable": True, "reason": ""}


# -----------------------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------------------
def build_index_from_files(pdf_paths: List[str]) -> Dict[str, Any]:
    """Build index from PDFs."""
    cfg = _get_cfg()
    docs = load_pdfs(pdf_paths)
    chunks = chunk_docs(docs, cfg.rag.chunk_size, cfg.rag.chunk_overlap)
    di = DenseIndex(cfg.rag.persist_dir, cfg.rag.embedding_model)
    n = di.build(chunks)
    global _DENSE_INDEX, _DOCS, _BM25
    _DENSE_INDEX = None
    _DOCS = None
    _BM25 = None
    return {"count": n, "persist_dir": cfg.rag.persist_dir}


def answer_question(question: str, *, mode: str = "simple", also_search_web: bool = False) -> Dict[str, Any]:
    """Main answer function - orchestrates PDF and web search."""
    cfg = _get_cfg()
    q = (question or "").strip()
    if not q:
        return {
            "question": "",
            "answer": "",
            "pdf": {"found": False, "has_evidence": False, "answerable": False, "status": "no_evidence", "reason": "Empty question.", "groups": []},
            "web": {"found": False, "has_sources": False, "answerable": False, "status": "skipped", "reason": "Empty question.", "results": []},
        }

    # PDF search
    queries = _make_queries(q, cfg)
    all_chunks = [chunk for qq in queries for chunk in hybrid_retrieve(qq, cfg)]

    # Deduplicate
    seen, deduped = set(), []
    for ch in all_chunks:
        key = (ch.get("source"), ch.get("page"), ch.get("start"), (ch.get("text", "")[:80]))
        if key not in seen:
            seen.add(key)
            deduped.append(ch)

    pdf_groups = _group_chunks_by_pdf(deduped)
    pdf_has_evidence = bool(pdf_groups)

    # Build UI groups
    ui_groups = []
    for g in pdf_groups[:10]:
        src = g.get("source", "unknown.pdf")
        matches = [
            {"page": m.get("page", "?"), "text": (m.get("text") or "").strip(), "citation": f"{src}:p{m.get('page', '?')}"}
            for m in (g.get("matches") or [])[:8]
        ]
        ui_groups.append({"source": src, "per_pdf_answer": "", "matches": matches})

    # Extract evidence
    support = _support_score(q, deduped)
    pdf_evidence = "\n\n".join(
        f"[{g['source']}:p{m.get('page')}] {m.get('text', '')}"
        for g in pdf_groups
        for m in (g.get("matches") or [])[:5]
    )
    pdf_evidence = _clean(pdf_evidence)

    # Judge evidence
    judge_found, judge_reason = (False, "")
    if pdf_evidence.strip():
        judge_found, judge_reason = _evidence_check(q, pdf_evidence[:7000], cfg)
    pdf_answerable = bool(judge_found)

    # Generate answer
    pdf_overall = ""
    if pdf_has_evidence and pdf_answerable:
        ctx = _clean(pdf_evidence)[: (cfg.rag.max_context_chars or 12000)]
        pdf_overall = ask_ollama(
            system=SYSTEM,
            user=USER.format(question=q, context=ctx),
            model=cfg.app.model,
            temperature=cfg.app.temperature,
            num_predict=cfg.app.max_new_tokens,
            raise_on_fail=False,
        ).strip()
        if not pdf_overall:
            fallback = _extractive_pdf_answer(q, ui_groups)
            if fallback:
                pdf_overall = "Extracted from PDFs (LLM unavailable):\n" + fallback

    # PDF status
    if not pdf_has_evidence:
        pdf_status, pdf_reason = "no_evidence", "No information found in PDFs."
    elif not pdf_answerable:
        pdf_status, pdf_reason = "no_answer", judge_reason or f"No information found (support={support:.2f})."
    else:
        pdf_status = "answer" if pdf_overall else "llm_down"
        pdf_reason = "PDF evidence supports answer, but LLM unavailable." if pdf_status == "llm_down" else ""

    pdf_section = {
        "found": bool(pdf_answerable),
        "has_evidence": bool(pdf_has_evidence),
        "answerable": bool(pdf_answerable),
        "status": pdf_status,
        "reason": pdf_reason,
        "overall_summary": pdf_overall,
        "groups": ui_groups,
        "support_score": round(float(support), 3),
        "judge": {"found": bool(judge_found), "reason": (judge_reason or "").strip()},
    }

    # Web search
    if bool(also_search_web):
        web_raw = _web_search_whitelisted(q, cfg)
    else:
        web_raw = {
            "answer": "",
            "resources": [],
            "citations": [],
            "has_sources": False,
            "answerable": False,
            "status": "skipped",
            "reason": "Web search not enabled.",
        }

    web_found = bool(web_raw.get("answerable"))
    web_has_sources = bool(web_raw.get("has_sources"))
    web_status = "answer" if web_found else ("no_sources" if not web_has_sources else "no_answer")
    if web_status == "skipped":
        web_status = web_raw.get("status", "skipped")

    web_section = {
        "found": web_found,
        "has_sources": web_has_sources,
        "answerable": web_found,
        "status": web_status,
        "reason": web_raw.get("reason", ""),
        "overview": web_raw.get("answer", ""),
        "results": web_raw.get("resources", []),
    }

    # Final answer
    final = ""
    if pdf_overall:
        final = pdf_overall
    if web_raw.get("answer"):
        if final:
            final += "\n\n**Additional information from whitelisted web sources:**\n" + web_raw["answer"]
        else:
            final = web_raw["answer"]

    if not final:
        if not pdf_has_evidence and not web_has_sources:
            final = "No information found in PDFs." if not also_search_web else "No information found in PDFs and no information found in whitelisted web sources."
        elif pdf_has_evidence and not pdf_answerable:
            final = pdf_reason
        elif web_has_sources and not web_found:
            final = web_section["reason"]

    return {"question": q, "answer": final, "pdf": pdf_section, "web": web_section}
