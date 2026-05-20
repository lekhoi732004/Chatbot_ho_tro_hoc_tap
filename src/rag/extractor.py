"""
Data extractor for vectorstore ingestion pipeline.

Responsibilities:
  1. Extract raw text from any supported file format
  2. Clean and normalise the text
  3. Detect language (vi / en / other)
  4. Smart-chunk with sentence-aware boundaries
  5. Deduplicate chunks via content hash
  6. Emit structured ExtractedDocument objects ready for embedding

Supports: PDF, DOCX, TXT, MD, CSV, JSON, and notebooks.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Set, Tuple

from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger("extractor")

# ── Supported extensions ──────────────────────────────────────────────────────
PDF_EXTS   = {".pdf"}
DOCX_EXTS  = {".docx", ".doc"}
TEXT_EXTS  = {".txt", ".md", ".rst", ".py", ".sql", ".toml"}
CSV_EXTS   = {".csv", ".tsv"}
JSON_EXTS  = {".json", ".jsonl"}
IPYNB_EXTS = {".ipynb"}
ALL_EXTS   = PDF_EXTS | DOCX_EXTS | TEXT_EXTS | CSV_EXTS | JSON_EXTS | IPYNB_EXTS

# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class RawPage:
    """A single page / section extracted from a document before chunking."""
    page_num: int
    text: str
    tables: List[List[List[str]]] = field(default_factory=list)


@dataclass
class ExtractedDocument:
    """Fully processed document ready for embedding."""
    source_file: str
    file_type: str
    title: str
    language: str                       # "vi" | "en" | "mixed" | "unknown"
    chunks: List[Dict]                  # [{text, chunk_id, char_start, metadata}]
    metadata: Dict = field(default_factory=dict)
    total_pages: int = 0
    total_chars: int = 0
    extract_errors: List[str] = field(default_factory=list)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        return self.chunk_count == 0


# ── Text cleaning ─────────────────────────────────────────────────────────────

_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE   = re.compile(r"[ \t]{2,}")
_ZERO_WIDTH    = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")
_CONTROL       = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    """
    Normalise and clean extracted text.
    - Remove zero-width / control characters
    - Normalise unicode (NFC)
    - Collapse excessive whitespace
    - Remove page-number lines and common header/footer noise
    """
    if not text:
        return ""

    # Unicode normalisation
    text = unicodedata.normalize("NFC", text)

    # Strip zero-width and control chars
    text = _ZERO_WIDTH.sub("", text)
    text = _CONTROL.sub("", text)

    # Remove lone page numbers (lines that are just a number)
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)

    # Collapse whitespace
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)

    return text.strip()


# ── Language detection ────────────────────────────────────────────────────────

_VI_CHARS = re.compile(
    r"[àáảãạăắặằẳẵâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
    r"ÀÁẢÃẠĂẮẶẰẲẴÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ]"
)


def detect_language(text: str) -> str:
    """
    Lightweight language detector — no external library.
    Returns 'vi', 'en', 'mixed', or 'unknown'.
    """
    sample = text[:2000]
    vi_count = len(_VI_CHARS.findall(sample))
    ascii_word_count = len(re.findall(r"\b[a-zA-Z]{3,}\b", sample))
    total_chars = max(len(sample), 1)

    vi_ratio = vi_count / total_chars

    if vi_ratio > 0.04:
        if ascii_word_count > 30:
            return "mixed"
        return "vi"
    elif ascii_word_count > 10:
        return "en"
    return "unknown"


# ── Smart chunker ─────────────────────────────────────────────────────────────

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।॥])\s+|\n{2,}")


def smart_chunk(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    metadata: Optional[Dict] = None,
    seen_hashes: Optional[Set[str]] = None,
) -> List[Dict]:
    """
    Sentence-aware chunker with deduplication.

    Tries to break at sentence boundaries rather than mid-word.
    Skips chunks whose content hash is in seen_hashes (dedup).

    Args:
        text: Cleaned text to chunk.
        chunk_size: Target chunk size in words.
        chunk_overlap: Overlap in words between consecutive chunks.
        metadata: Base metadata dict attached to every chunk.
        seen_hashes: Set of MD5 hashes; chunks matching existing hashes are skipped.

    Returns:
        List of chunk dicts with keys:
          text, chunk_id, char_start, word_count, content_hash, metadata
    """
    if not text:
        return []

    stored_hashes = seen_hashes or set()
    local_hashes: Set[str] = set()

    # Split into sentences first for cleaner boundaries
    sentences = _SENTENCE_SPLIT.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: List[Dict] = []
    chunk_id = 0
    buffer_words: List[str] = []
    char_cursor = 0

    def _flush(words: List[str], char_pos: int) -> Optional[Dict]:
        nonlocal chunk_id
        chunk_text = " ".join(words).strip()
        if not chunk_text:
            return None
        content_hash = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()
        if content_hash in stored_hashes or content_hash in local_hashes:
            logger.debug(f"Skipping duplicate chunk hash {content_hash[:8]}")
            return None
        local_hashes.add(content_hash)
        c = {
            "text": chunk_text,
            "chunk_id": chunk_id,
            "char_start": char_pos,
            "word_count": len(words),
            "content_hash": content_hash,
            "metadata": dict(metadata or {}),
        }
        chunk_id += 1
        return c

    for sentence in sentences:
        s_words = sentence.split()
        if not s_words:
            continue

        # If adding this sentence would overflow, flush first
        if len(buffer_words) + len(s_words) > chunk_size and buffer_words:
            result = _flush(buffer_words, char_cursor)
            if result:
                chunks.append(result)
            # Keep overlap
            buffer_words = buffer_words[-chunk_overlap:] if chunk_overlap else []

        buffer_words.extend(s_words)
        char_cursor += len(sentence) + 1

    # Flush remaining
    if buffer_words:
        result = _flush(buffer_words, char_cursor)
        if result:
            chunks.append(result)

    return chunks


# ── Format-specific raw extractors ───────────────────────────────────────────

def _extract_pdf(file_path: str, max_pages: int) -> Tuple[List[RawPage], Dict]:
    """Extract pages from PDF. Tries pdfplumber first, then PyMuPDF."""
    pages: List[RawPage] = []
    errors: List[str] = []
    meta: Dict = {}

    # --- pdfplumber ---
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            meta = {
                "total_pages": len(pdf.pages),
                "title": Path(file_path).stem,
            }
            for i, page in enumerate(pdf.pages[:max_pages]):
                text = page.extract_text() or ""
                tables = []
                try:
                    for tbl in page.extract_tables():
                        if tbl:
                            tables.append(tbl)
                except Exception:
                    pass
                pages.append(RawPage(page_num=i + 1, text=text, tables=tables))

        total_text = " ".join(p.text for p in pages)
        if len(total_text.strip()) < 200:
            raise ValueError("Low text yield")
        return pages, meta

    except Exception as e:
        errors.append(f"pdfplumber: {e}")
        logger.warning(f"pdfplumber failed for {file_path}: {e}")

    # --- PyMuPDF fallback ---
    try:
        import fitz
        doc = fitz.open(file_path)
        meta = {
            "total_pages": doc.page_count,
            "title": doc.metadata.get("title") or Path(file_path).stem,
            "author": doc.metadata.get("author", ""),
        }
        pages = []
        for i in range(min(doc.page_count, max_pages)):
            text = doc[i].get_text("text") or ""
            pages.append(RawPage(page_num=i + 1, text=text))
        doc.close()

        total_text = " ".join(p.text for p in pages)
        if len(total_text.strip()) < 200:
            raise ValueError("Low text yield")
        return pages, meta

    except Exception as e:
        errors.append(f"pymupdf: {e}")
        logger.warning(f"PyMuPDF failed for {file_path}: {e}")

    logger.error(f"All PDF text extractors failed for {file_path}: {errors}")

    return pages, meta


def _extract_docx(file_path: str) -> Tuple[List[RawPage], Dict]:
    from docx import Document
    doc = Document(file_path)
    props = doc.core_properties

    lines: List[str] = []
    for para in doc.paragraphs:
        t = para.text.strip()
        if t:
            if para.style.name.startswith("Heading"):
                lines.append(f"\n## {t}\n")
            else:
                lines.append(t)

    tables: List[List[List[str]]] = []
    for tbl in doc.tables:
        rows = [[c.text.strip() for c in row.cells] for row in tbl.rows]
        if rows:
            tables.append(rows)
            lines.append("\n[TABLE]\n" + "\n".join(" | ".join(r) for r in rows))

    full_text = "\n".join(lines)
    page = RawPage(page_num=1, text=full_text, tables=tables)
    meta = {
        "title": props.title or Path(file_path).stem,
        "author": props.author or "",
        "total_pages": 1,
    }
    return [page], meta


def _extract_text(file_path: str) -> Tuple[List[RawPage], Dict]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    page = RawPage(page_num=1, text=text)
    meta = {"title": Path(file_path).stem, "total_pages": 1}
    return [page], meta


def _extract_csv(file_path: str) -> Tuple[List[RawPage], Dict]:
    """Convert CSV/TSV rows to prose-style text chunks."""
    lines: List[str] = []
    delimiter = "\t" if file_path.endswith(".tsv") else ","
    with open(file_path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        headers = reader.fieldnames or []
        for row_num, row in enumerate(reader, 1):
            # Convert each row to "Field: value. Field: value." format
            parts = [f"{k}: {v}" for k, v in row.items() if v and v.strip()]
            if parts:
                lines.append(f"Row {row_num}: " + ". ".join(parts))
    text = "\n".join(lines)
    meta = {
        "title": Path(file_path).stem,
        "columns": headers,
        "total_pages": 1,
    }
    return [RawPage(page_num=1, text=text)], meta


def _extract_json(file_path: str) -> Tuple[List[RawPage], Dict]:
    """Flatten JSON / JSONL to readable text."""
    lines: List[str] = []
    is_jsonl = file_path.endswith(".jsonl")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        if is_jsonl:
            records = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
            records = data if isinstance(data, list) else [data]

    def _flatten(obj, prefix="") -> List[str]:
        parts = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                parts.extend(_flatten(v, f"{prefix}{k}: " if prefix == "" else f"{prefix}.{k}: "))
        elif isinstance(obj, list):
            for i, item in enumerate(obj[:20]):  # cap to avoid huge outputs
                parts.extend(_flatten(item, f"{prefix}[{i}] "))
        else:
            parts.append(f"{prefix}{obj}")
        return parts

    for i, rec in enumerate(records[:5000]):
        flat = _flatten(rec)
        lines.append(f"Record {i + 1}: " + " | ".join(flat[:15]))

    text = "\n".join(lines)
    meta = {"title": Path(file_path).stem, "record_count": len(records), "total_pages": 1}
    return [RawPage(page_num=1, text=text)], meta


def _extract_ipynb(file_path: str) -> Tuple[List[RawPage], Dict]:
    """Extract markdown and code cells from a Jupyter notebook."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        notebook = json.load(f)

    lines: List[str] = []
    cells = notebook.get("cells", [])
    for idx, cell in enumerate(cells, 1):
        cell_type = cell.get("cell_type", "unknown")
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        source = str(source).strip()
        if not source:
            continue
        lines.append(f"[{cell_type.upper()} CELL {idx}]\n{source}")

    text = "\n\n".join(lines)
    meta = {"title": Path(file_path).stem, "cell_count": len(cells), "total_pages": 1}
    return [RawPage(page_num=1, text=text)], meta


# ── Main extraction entry point ───────────────────────────────────────────────

def extract_document(
    file_path: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    extra_metadata: Optional[Dict] = None,
    seen_hashes: Optional[Set[str]] = None,
) -> ExtractedDocument:
    """
    Full extraction pipeline for a single file.

    1. Detect file type
    2. Extract raw pages/text
    3. Clean text
    4. Detect language
    5. Smart-chunk with deduplication
    6. Return structured ExtractedDocument

    Args:
        file_path: Absolute or relative path to the source file.
        chunk_size: Override config chunk_size.
        chunk_overlap: Override config chunk_overlap.
        extra_metadata: Additional key-value pairs added to every chunk's metadata.
        seen_hashes: Mutable set of already-seen content hashes (cross-document dedup).

    Returns:
        ExtractedDocument ready to be embedded and stored.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    cfg = get_config()
    chunk_size    = chunk_size    or cfg.get("rag.chunk_size", 512)
    chunk_overlap = chunk_overlap or cfg.get("rag.chunk_overlap", 64)
    max_pages     = cfg.get("tools.pdf.max_pages", 200)

    ext = Path(file_path).suffix.lower()
    errors: List[str] = []

    logger.info(f"Extracting: {Path(file_path).name} (type={ext})")

    # ── Dispatch to format extractor ─────────────────────────────────────────
    try:
        if ext in PDF_EXTS:
            raw_pages, file_meta = _extract_pdf(file_path, max_pages)
            file_type = "pdf"
        elif ext in DOCX_EXTS:
            raw_pages, file_meta = _extract_docx(file_path)
            file_type = "docx"
        elif ext in TEXT_EXTS:
            raw_pages, file_meta = _extract_text(file_path)
            file_type = "text"
        elif ext in CSV_EXTS:
            raw_pages, file_meta = _extract_csv(file_path)
            file_type = "csv"
        elif ext in JSON_EXTS:
            raw_pages, file_meta = _extract_json(file_path)
            file_type = "json"
        elif ext in IPYNB_EXTS:
            raw_pages, file_meta = _extract_ipynb(file_path)
            file_type = "ipynb"
        else:
            raise ValueError(f"Unsupported extension: {ext}")
    except Exception as e:
        logger.error(f"Extraction failed for {file_path}: {e}")
        return ExtractedDocument(
            source_file=file_path,
            file_type=ext.lstrip("."),
            title=Path(file_path).stem,
            language="unknown",
            chunks=[],
            extract_errors=[str(e)],
        )

    # ── Assemble full text from all pages ────────────────────────────────────
    full_text_parts: List[str] = []
    for page in raw_pages:
        cleaned = clean_text(page.text)
        if cleaned:
            full_text_parts.append(cleaned)

    full_text = "\n\n".join(full_text_parts)
    language  = detect_language(full_text)

    # ── Build per-chunk metadata ──────────────────────────────────────────────
    base_meta = {
        "source_file": file_path,
        "file_name":   Path(file_path).name,
        "file_type":   file_type,
        "language":    language,
        "title":       file_meta.get("title", Path(file_path).stem),
    }
    if extra_metadata:
        base_meta.update(extra_metadata)

    # Add page-level metadata per page
    all_chunks: List[Dict] = []
    global_chunk_id = 0

    if len(raw_pages) > 1:
        # Chunk per page to preserve page-level metadata
        for page in raw_pages:
            cleaned = clean_text(page.text)
            if not cleaned:
                continue
            page_meta = {**base_meta, "page": page.page_num}
            page_chunks = smart_chunk(
                cleaned,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                metadata=page_meta,
                seen_hashes=seen_hashes,
            )
            for c in page_chunks:
                c["chunk_id"] = global_chunk_id
                global_chunk_id += 1
                all_chunks.append(c)
    else:
        # Single-page doc — chunk the whole thing
        all_chunks = smart_chunk(
            full_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata=base_meta,
            seen_hashes=seen_hashes,
        )

    total_chars = sum(len(c["text"]) for c in all_chunks)

    logger.info(
        f"  → {len(all_chunks)} chunks | lang={language} | "
        f"pages={len(raw_pages)} | chars={total_chars:,}"
    )

    return ExtractedDocument(
        source_file=file_path,
        file_type=file_type,
        title=file_meta.get("title", Path(file_path).stem),
        language=language,
        chunks=all_chunks,
        metadata={**file_meta, **base_meta},
        total_pages=file_meta.get("total_pages", len(raw_pages)),
        total_chars=total_chars,
        extract_errors=errors,
    )


def iter_directory(
    dir_path: str,
    extensions: Optional[List[str]] = None,
    recursive: bool = True,
) -> Iterator[str]:
    """
    Yield file paths from a directory matching the given extensions.

    Args:
        dir_path: Root directory.
        extensions: List like [".pdf", ".docx"]. Defaults to ALL_EXTS.
        recursive: Whether to search subdirectories.
    """
    filter_exts = {e.lower() for e in extensions} if extensions else ALL_EXTS
    root = Path(dir_path)
    pattern = "**/*" if recursive else "*"
    for p in root.glob(pattern):
        if p.is_file() and p.suffix.lower() in filter_exts:
            yield str(p)
