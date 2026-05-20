"""
Compatibility ingest API.

The active ingestion implementation is `src.rag.pipeline` + `src.rag.vectorstore`.
This module keeps older imports working while routing everything to the new
hybrid FAISS IVF+HNSW + BM25 vector database.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.rag.embedder import embed_texts
from src.rag.pipeline import run_pipeline
from src.rag.vectorstore import VectorStore
from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger("ingest")


def ingest_text(
    text: str,
    metadata: Optional[Dict] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    vectorstore: Optional[VectorStore] = None,
) -> VectorStore:
    """
    Ingest raw text directly into the vector database.
    
    Useful for ingesting extracted text from files without needing to write
    temporary files.
    
    Args:
        text: The text content to ingest.
        metadata: Optional metadata dict (e.g., source_file, page, etc.)
        chunk_size: Override config chunk size.
        chunk_overlap: Override config chunk overlap.
        vectorstore: Optional existing vectorstore to add to.
        
    Returns:
        VectorStore with ingested text.
    """
    if not text or len(text.strip()) < 10:
        logger.warning("Skipping ingest_text: text too short or empty")
        vs = vectorstore or VectorStore()
        return vs
    
    cfg = get_config()
    chunk_size = chunk_size or cfg.get("rag.chunk_size", 512)
    chunk_overlap = chunk_overlap or cfg.get("rag.chunk_overlap", 64)
    
    # Simple text chunking
    chunks = _chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
    
    # Embed chunks
    embeddings = embed_texts([chunk["text"] for chunk in chunks], use_cache=False)
    
    # Add to vectorstore
    vs = vectorstore or VectorStore()
    vs.load()
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_meta = metadata.copy() if metadata else {}
        chunk_meta.update({
            "chunk_id": i,
            "chunk_index": i,
            "total_chunks": len(chunks),
        })
        vs.add_document(chunk["text"], embedding, metadata=chunk_meta)
    
    logger.info(f"Ingested {len(chunks)} text chunks (metadata: {metadata})")
    return vs


def _chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[Dict]:
    """
    Simple text chunker by character count with overlap.
    
    Args:
        text: Text to chunk.
        chunk_size: Target chunk size in characters.
        overlap: Overlap size in characters.
        
    Returns:
        List of {"text": chunk_text} dicts.
    """
    if len(text) <= chunk_size:
        return [{"text": text}]
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({"text": chunk_text})
        
        # Move start position forward, but back up to not break words in overlap
        start = end - overlap if end < len(text) else len(text)
        if start < 0:
            start = 0
    
    return chunks


def ingest_file(file_path: str, vectorstore: Optional[VectorStore] = None, extra_metadata: Optional[dict] = None) -> VectorStore:
    store_path = str(vectorstore.store_path) if vectorstore is not None else None
    run_pipeline(file_path, store_path=store_path, extra_metadata=extra_metadata)
    vs = vectorstore or VectorStore(store_path)
    vs.load()
    return vs


def ingest_directory(dir_path: str, extensions: Optional[List[str]] = None) -> VectorStore:
    report = run_pipeline(dir_path, extensions=extensions)
    report.print_summary()
    vs = VectorStore()
    vs.load()
    return vs
