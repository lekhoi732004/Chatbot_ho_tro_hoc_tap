"""
Document router.
Routes file extraction to the correct tool based on file extension.
Supports: .pdf, .docx, .doc, .txt, .md.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

from src.utils.logger import get_logger

logger = get_logger("router")

PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx", ".doc"}
TEXT_EXTS = {".txt", ".md", ".rst", ".csv"}


def extract_document(
    file_path: str,
) -> Tuple[str, Dict]:
    """
    Route a file to the appropriate extractor and return text + metadata.

    Args:
        file_path: Path to the document.
    Returns:
        (text, metadata) tuple.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = Path(file_path).suffix.lower()
    file_name = Path(file_path).name

    logger.info(f"Routing document: {file_name} (ext={ext})")

    # ── Plain text ────────────────────────────────────────────────────────────
    if ext in TEXT_EXTS:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return text, {"type": "text", "file_name": file_name}

    # ── PDF ───────────────────────────────────────────────────────────────────
    if ext in PDF_EXTS:
        from src.tools.pdf_reader import extract_pdf, get_pdf_metadata

        text, pages = extract_pdf(file_path)

        meta = get_pdf_metadata(file_path)
        meta["type"] = "pdf"
        meta["pages_extracted"] = len(pages)
        return text, meta

    # ── DOCX ──────────────────────────────────────────────────────────────────
    if ext in DOCX_EXTS:
        from src.tools.docx_reader import extract_docx, get_docx_metadata

        text, structured = extract_docx(file_path)
        meta = get_docx_metadata(file_path)
        meta["type"] = "docx"
        meta["headings"] = len(structured.get("headings", []))
        meta["tables"] = len(structured.get("tables", []))
        return text, meta

    raise ValueError(
        f"Unsupported file type: {ext}. Supported: "
        f"{PDF_EXTS | DOCX_EXTS | TEXT_EXTS}"
    )
