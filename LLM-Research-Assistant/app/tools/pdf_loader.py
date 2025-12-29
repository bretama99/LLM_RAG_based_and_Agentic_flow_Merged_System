"""
pdf_loader.py
Functions to load and chunk PDF documents.
"""

import os
from typing import List

from pypdf import PdfReader
from app.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_PDF_CHARS,
    MAX_PDF_PAGES,
)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract raw text from a PDF file using pypdf.

    - Reads at most MAX_PDF_PAGES pages.
    - Caps the final text to MAX_PDF_CHARS characters.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(pdf_path)
    pages_text = []

    num_pages = min(len(reader.pages), MAX_PDF_PAGES)

    for i in range(num_pages):
        page = reader.pages[i]
        text = page.extract_text() or ""
        pages_text.append(text)

    full_text = "\n".join(pages_text)

    if len(full_text) > MAX_PDF_CHARS:
        full_text = full_text[:MAX_PDF_CHARS]

    return full_text


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """
    Simple character-based chunking with overlap.
    """
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

        if start < 0:
            start = 0
        if start >= n:
            break

    return chunks
