"""
Data extraction pipeline orchestrator.

Ties together:
  extractor  → ExtractedDocument (text + chunks)
  embedder   → numpy embeddings
  vectorstore → FAISS index

Entry points:
  run_pipeline(file_or_dir)  — ingest one file or a whole directory
  IngestPipeline             — class for fine-grained control
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.rag.extractor import (
    ExtractedDocument,
    extract_document,
    iter_directory,
    ALL_EXTS,
)
from src.rag.embedder import embed_texts, load_embed_cache, save_embed_cache
from src.rag.vectorstore import VectorStore, get_vectorstore
from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger("pipeline")


# ── Result objects ────────────────────────────────────────────────────────────

@dataclass
class FileResult:
    """Outcome of processing one file."""
    file_path: str
    status: str          # "ingested" | "skipped" | "failed" | "empty"
    chunks_added: int = 0
    duration_s: float = 0.0
    error: str = ""


@dataclass
class PipelineReport:
    """Summary report from a pipeline run."""
    total_files: int = 0
    ingested: int = 0
    skipped: int = 0
    failed: int = 0
    empty: int = 0
    total_chunks_added: int = 0
    duration_s: float = 0.0
    file_results: List[FileResult] = field(default_factory=list)

    def print_summary(self) -> None:
        print("\n" + "═" * 56)
        print(f"  Pipeline Report")
        print("═" * 56)
        print(f"  Files found:    {self.total_files}")
        print(f"  Ingested:       {self.ingested}")
        print(f"  Skipped:        {self.skipped}  (already up-to-date)")
        print(f"  Failed:         {self.failed}")
        print(f"  Empty:          {self.empty}")
        print(f"  Chunks added:   {self.total_chunks_added:,}")
        print(f"  Duration:       {self.duration_s:.1f}s")
        print("═" * 56)
        if self.failed > 0:
            print("\n  Failed files:")
            for r in self.file_results:
                if r.status == "failed":
                    print(f"    • {Path(r.file_path).name}: {r.error}")
        print()


# ── Pipeline class ────────────────────────────────────────────────────────────

class IngestPipeline:
    """
    Stateful ingest pipeline.

    Usage:
        pipeline = IngestPipeline()
        pipeline.load()                          # load existing index
        report = pipeline.run("data/raw/")       # ingest a directory
        report.print_summary()
    """

    def __init__(
        self,
        store_path: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        embed_batch_size: Optional[int] = None,
        embed_cache_file: Optional[str] = None,
        force_reingest: bool = False,
    ):
        cfg = get_config()
        self.store_path        = store_path or cfg.get("rag.vectorstore_path", "vectorstore/faiss_index")
        self.chunk_size        = chunk_size or cfg.get("rag.chunk_size", 512)
        self.chunk_overlap     = chunk_overlap or cfg.get("rag.chunk_overlap", 64)
        self.embed_batch_size  = embed_batch_size or cfg.get("embedding.batch_size", 32)
        self.embed_cache_file  = embed_cache_file
        self.force_reingest    = force_reingest
        self.vs: Optional[VectorStore] = None

    def load(self) -> "IngestPipeline":
        """Load (or create) the vectorstore. Call before run()."""
        self.vs = VectorStore(self.store_path)
        self.vs.load()
        if self.embed_cache_file:
            load_embed_cache(self.embed_cache_file)
        return self

    def _ensure_loaded(self):
        if self.vs is None:
            self.load()

    # ── Single file ───────────────────────────────────────────────────────────

    def ingest_file(
        self,
        file_path: str,
        extra_metadata: Optional[Dict] = None,
    ) -> FileResult:
        """
        Extract, embed, and store a single file.

        Args:
            file_path: Path to the source document.
            extra_metadata: Extra key-value pairs attached to every chunk.

        Returns:
            FileResult with status and stats.
        """
        self._ensure_loaded()
        t0 = time.time()

        # Skip check
        if self.vs.is_already_ingested(file_path, force=self.force_reingest):
            logger.info(f"Skipping (already ingested): {Path(file_path).name}")
            return FileResult(file_path=file_path, status="skipped")

        # Extract
        try:
            is_user_upload = bool(extra_metadata and extra_metadata.get("source_origin") == "user_upload")
            doc: ExtractedDocument = extract_document(
                file_path,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                extra_metadata=extra_metadata,
                seen_hashes=None if is_user_upload else self.vs.seen_hashes,
            )
        except Exception as e:
            logger.error(f"Extraction failed: {file_path}: {e}")
            return FileResult(file_path=file_path, status="failed", error=str(e), duration_s=time.time()-t0)

        if doc.is_empty:
            logger.warning(f"No content extracted from: {Path(file_path).name}")
            return FileResult(file_path=file_path, status="empty", duration_s=time.time()-t0)

        if extra_metadata and extra_metadata.get("source_origin") == "user_upload":
            upload_scope = str(extra_metadata.get("session_id") or "default")
            upload_name = str(extra_metadata.get("upload_file_name") or Path(file_path).name)
            for chunk in doc.chunks:
                original_hash = chunk.get("content_hash", "")
                salted = f"{upload_scope}:{upload_name}:{original_hash}"
                chunk["content_hash"] = hashlib.md5(salted.encode("utf-8")).hexdigest()

        # Embed
        try:
            texts = [c["text"] for c in doc.chunks]
            embeddings = embed_texts(
                texts,
                batch_size=self.embed_batch_size,
                show_progress=len(texts) > 20,
            )
        except Exception as e:
            logger.error(f"Embedding failed: {file_path}: {e}")
            return FileResult(file_path=file_path, status="failed", error=f"embed: {e}", duration_s=time.time()-t0)

        # Store
        added = self.vs.add(doc.chunks, embeddings)
        hashes = [c.get("content_hash", "") for c in doc.chunks]
        self.vs.record_ingestion(file_path, added, hashes)
        self.vs.save()

        elapsed = round(time.time() - t0, 2)
        logger.info(
            f"  ✓ {Path(file_path).name}: "
            f"{added} chunks | {doc.language} | {elapsed}s"
        )
        return FileResult(
            file_path=file_path,
            status="ingested",
            chunks_added=added,
            duration_s=elapsed,
        )

    # ── Directory ─────────────────────────────────────────────────────────────

    def run(
        self,
        path: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
        extra_metadata: Optional[Dict] = None,
    ) -> PipelineReport:
        """
        Ingest a file or recursively process a directory.

        Args:
            path: Path to a file or directory.
            extensions: Filter extensions, e.g. [".pdf", ".docx"].
            recursive: Search subdirectories.
            extra_metadata: Metadata attached to all chunks in this run.

        Returns:
            PipelineReport with per-file results and summary stats.
        """
        self._ensure_loaded()
        t0 = time.time()
        report = PipelineReport()

        if Path(path).is_file():
            files = [path]
        else:
            files = list(iter_directory(path, extensions=extensions, recursive=recursive))

        report.total_files = len(files)
        logger.info(f"Pipeline starting: {report.total_files} files from {path}")

        for i, file_path in enumerate(files, 1):
            logger.info(f"[{i}/{report.total_files}] {Path(file_path).name}")
            result = self.ingest_file(file_path, extra_metadata=extra_metadata)
            report.file_results.append(result)

            if result.status == "ingested":
                report.ingested += 1
                report.total_chunks_added += result.chunks_added
            elif result.status == "skipped":
                report.skipped += 1
            elif result.status == "failed":
                report.failed += 1
            elif result.status == "empty":
                report.empty += 1

        # Save embedding cache if used
        if self.embed_cache_file:
            save_embed_cache()

        report.duration_s = round(time.time() - t0, 2)
        logger.info(
            f"Pipeline done: {report.ingested} ingested, {report.skipped} skipped, "
            f"{report.failed} failed | {report.total_chunks_added} chunks | {report.duration_s}s"
        )
        return report

    # ── Utilities ─────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """Return vectorstore statistics."""
        self._ensure_loaded()
        return self.vs.stats()

    def delete_source(self, file_path: str) -> int:
        """Remove all chunks from a specific source file."""
        self._ensure_loaded()
        removed = self.vs.delete_source(file_path)
        if removed:
            self.vs.save()
        return removed

    def export(self, output_path: str) -> int:
        """Export all chunks to JSONL for backup."""
        self._ensure_loaded()
        return self.vs.export_jsonl(output_path)

    def reimport(self, jsonl_path: str) -> int:
        """Rebuild vectorstore from a JSONL export."""
        self._ensure_loaded()
        return self.vs.import_jsonl(jsonl_path, embed_fn=embed_texts)


# ── Convenience function ──────────────────────────────────────────────────────

def run_pipeline(
    path: str,
    extensions: Optional[List[str]] = None,
    force: bool = False,
    store_path: Optional[str] = None,
    extra_metadata: Optional[Dict] = None,
    embed_cache_file: Optional[str] = None,
) -> PipelineReport:
    """
    One-call pipeline runner.

    Args:
        path: File or directory to ingest.
        extensions: Filter by extension(s). None = all supported.
        force: Re-ingest even files already in the manifest.
        store_path: Override vectorstore path from config.
        extra_metadata: Metadata added to every chunk.
        embed_cache_file: Path to optional embedding cache file.

    Returns:
        PipelineReport.
    """
    pipeline = IngestPipeline(
        store_path=store_path,
        force_reingest=force,
        embed_cache_file=embed_cache_file,
    )
    pipeline.load()
    return pipeline.run(path, extensions=extensions, extra_metadata=extra_metadata)
