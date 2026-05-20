"""
PDF reader tool.
Extracts text (and optionally tables) from PDF files using pdfplumber.
Falls back to PyMuPDF (fitz) for complex layouts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger("pdf_reader")


def _try_pdfplumber(path: str, max_pages: int) -> List[Dict]:
    """Primary extractor using pdfplumber."""
    import pdfplumber

    pages_data = []
    with pdfplumber.open(path) as pdf:
        total = min(len(pdf.pages), max_pages)
        for i, page in enumerate(pdf.pages[:total]):
            text = page.extract_text() or ""
            tables = []
            for tbl in page.extract_tables():
                if tbl:
                    tables.append(tbl)
            pages_data.append(
                {
                    "page": i + 1,
                    "text": text.strip(),
                    "tables": tables,
                }
            )
    return pages_data


def _try_pymupdf(path: str, max_pages: int) -> List[Dict]:
    """Fallback extractor using PyMuPDF."""
    import fitz

    pages_data = []
    doc = fitz.open(path)
    for i in range(min(len(doc), max_pages)):
        page = doc[i]
        text = page.get_text("text") or ""
        pages_data.append(
            {
                "page": i + 1,
                "text": text.strip(),
                "tables": [],
            }
        )
    doc.close()
    return pages_data


def extract_pdf(
    file_path: str,
    max_pages: Optional[int] = None,
    join_pages: bool = False,
) -> Tuple[str, List[Dict]]:
    """
    Extract text and tables from a PDF.

    Args:
        file_path: Path to the PDF file.
        max_pages: Maximum pages to extract. Defaults to config value.
        join_pages: If True, returns a single joined text string.

    Returns:
        (full_text, pages_data)
        - full_text: All page texts joined by newlines.
        - pages_data: List of {page, text, tables} dicts.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    cfg = get_config()
    max_pages = max_pages or cfg.get("tools.pdf.max_pages", 200)

    logger.info(f"Extracting PDF: {file_path} (max_pages={max_pages})")

    try:
        pages_data = _try_pdfplumber(file_path, max_pages)
        logger.info(f"pdfplumber extracted {len(pages_data)} pages")
    except Exception as e:
        logger.warning(f"pdfplumber failed ({e}), falling back to PyMuPDF")
        try:
            pages_data = _try_pymupdf(file_path, max_pages)
        except Exception as e2:
            raise RuntimeError(f"Both PDF extractors failed: pdfplumber={e}, pymupdf={e2}")

    full_text = "\n\n".join(p["text"] for p in pages_data if p["text"])
    return full_text, pages_data


def get_pdf_metadata(file_path: str) -> Dict:
    """Return basic PDF metadata (title, author, pages, size)."""
    try:
        import fitz
        doc = fitz.open(file_path)
        meta = doc.metadata or {}
        info = {
            "title": meta.get("title", Path(file_path).stem),
            "author": meta.get("author", "Unknown"),
            "total_pages": doc.page_count,
            "file_size_kb": round(os.path.getsize(file_path) / 1024, 2),
        }
        doc.close()
        return info
    except Exception:
        return {
            "title": Path(file_path).stem,
            "author": "Unknown",
            "total_pages": -1,
            "file_size_kb": round(os.path.getsize(file_path) / 1024, 2),
        }
