# src/rag/verifier.py
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple

_STOP = {
    "what","is","are","the","a","an","and","or","to","of","in","on","for","with","from","as","by",
    "how","why","when","where","which","who","does","do","did","can","could","should","would",
    "about","into","than","then","it","this","that","these","those","be","been","being","at","we","you"
}

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _truncate(s: str, n: int = 260) -> str:
    s = s or ""
    return s if len(s) <= n else (s[:n].rstrip() + "…")

def _tokens(text: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in toks if len(t) >= 3 and t not in _STOP]

def _overlap(qt: List[str], ct: List[str]) -> Tuple[int, float]:
    qs, cs = set(qt), set(ct)
    inter = len(qs & cs)
    score = inter / max(1, len(qs))
    return inter, score

def filter_pdf_chunks(question: str, chunks: List[dict], max_keep: int = 8) -> List[dict]:
    qt = _tokens(question)
    if not qt:
        return []
    min_overlap = 2 if len(set(qt)) >= 4 else 1
    min_score = 0.12

    scored = []
    for ch in chunks or []:
        txt = ch.get("text", "") or ""
        ct = _tokens(txt)
        inter, sc = _overlap(qt, ct)
        if inter >= min_overlap and sc >= min_score:
            scored.append((sc, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:max_keep]]

def filter_web_results(question: str, results: List[dict], max_keep: int = 6) -> List[dict]:
    qt = _tokens(question)
    if not qt:
        return []
    min_overlap = 2 if len(set(qt)) >= 4 else 1
    min_score = 0.10

    scored = []
    for r in results or []:
        snip = (r.get("snippet") or r.get("content") or "") or ""
        ct = _tokens(snip)
        inter, sc = _overlap(qt, ct)
        if inter >= min_overlap and sc >= min_score:
            scored.append((sc, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:max_keep]]

def _split_claims(answer: str, max_claims: int = 10) -> List[str]:
    a = (answer or "").strip()
    if not a:
        return []
    lines = [ln.strip("•*- \t").strip() for ln in a.splitlines() if ln.strip()]
    claims = lines if len(lines) >= 3 else re.split(r"(?<=[.!?])\s+", a)
    return [c.strip() for c in claims if len(c.strip()) >= 25][:max_claims]

def verify_answer(
    question: str,
    answer: str,
    pdf_chunks: Optional[List[dict]] = None,
    web_results: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    claims = _split_claims(answer)
    ev = []

    for ch in pdf_chunks or []:
        src = ch.get("source", "unknown.pdf")
        page = ch.get("page", "?")
        txt = _clean(ch.get("text", "") or "")
        if txt:
            ev.append(("pdf", f"{src}:p{page}", txt))

    for r in web_results or []:
        url = (r.get("url") or r.get("source") or "WEB").strip()
        snip = _clean((r.get("snippet") or r.get("content") or "") or "")
        if snip:
            ev.append(("web", url, snip))

    if not ev:
        return {"ok": False, "support": 0.0, "reason": "No evidence to verify against.", "best_sources": [], "claims": []}

    evt = [(k, s, _tokens(t), t) for (k, s, t) in ev]
    sup = 0.0
    best_sources = []
    rows = []

    for c in claims:
        ct = _tokens(c)
        best = (0.0, None, None, None)
        for kind, src, tok, raw in evt:
            inter = len(set(ct) & set(tok))
            sc = inter / max(1, len(set(ct)))
            if sc > best[0]:
                best = (sc, kind, src, raw)
        sc, kind, src, raw = best

        if sc >= 0.22:
            label, sup = "supported", sup + 1.0
        elif sc >= 0.12:
            label, sup = "partial", sup + 0.5
        else:
            label = "unsupported"

        rows.append({"claim": c, "support": label, "best": src})
        if src and sc >= 0.12:
            best_sources.append({"kind": kind, "source": src, "snippet": _truncate(raw), "score": round(sc, 3)})

    support = sup / max(1, len(claims)) if claims else 0.0
    ok = support >= 0.6
    reason = "Answer is evidence-supported." if ok else "Answer is not sufficiently supported (likely external knowledge)."

    # Dedup sources
    seen, uniq = set(), []
    for b in sorted(best_sources, key=lambda x: x["score"], reverse=True):
        if b["source"] in seen:
            continue
        seen.add(b["source"])
        uniq.append(b)

    return {"ok": ok, "support": float(round(support, 3)), "reason": reason, "best_sources": uniq[:8], "claims": rows}
