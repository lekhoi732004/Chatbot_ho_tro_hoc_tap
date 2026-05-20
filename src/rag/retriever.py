"""
Hybrid retriever.

Combines:
  - BM25 lexical retrieval
  - FAISS semantic retrieval over IVF+HNSW vector database
  - cross-encoder reranking for final precision
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.rag.embedder import embed_query
from src.rag.vectorstore import VectorStore, get_vectorstore
from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger("retriever")

_vectorstore: Optional[VectorStore] = None


def _get_vectorstore() -> VectorStore:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = get_vectorstore(reload=True)
        if _vectorstore.size == 0:
            logger.warning("Hybrid vector database is empty. Build it from data_main first.")
    return _vectorstore


def retrieve(
    query: str,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    session_id: Optional[str] = None,
) -> List[Dict]:
    cfg = get_config()
    top_k = top_k or cfg.get("rag.top_k_retrieve", 10)
    score_threshold = score_threshold if score_threshold is not None else cfg.get("rag.hybrid_score_threshold", 0.0)
    vector_weight = cfg.get("rag.hybrid.vector_weight", 0.65)
    bm25_weight = cfg.get("rag.hybrid.bm25_weight", 0.35)
    candidate_k = cfg.get("rag.hybrid.candidate_k", max(top_k * 4, 20))
    rerank_enabled = cfg.get("reranker.enabled", True)
    rerank_top_k = cfg.get("reranker.top_k", cfg.get("rag.top_k_rerank", top_k))

    vs = _get_vectorstore()
    if vs.size == 0:
        return []

    query_embedding = embed_query(query)
    results = vs.hybrid_search(
        query=query,
        query_embedding=query_embedding,
        top_k=max(top_k, rerank_top_k),
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        candidate_k=candidate_k,
    )

    results = _prioritize_session_uploads(results, session_id=session_id)
    filtered = [result for result in results if result.get("hybrid_score", 0.0) >= score_threshold]
    if rerank_enabled and filtered:
        try:
            from src.rag.reranker import rerank
            filtered = rerank(query, filtered, top_k=rerank_top_k)
            filtered = _prioritize_session_uploads(filtered, session_id=session_id)
        except Exception as exc:
            logger.warning(f"Reranker failed; using hybrid order: {exc}")
            filtered = filtered[:top_k]
    else:
        filtered = filtered[:top_k]

    logger.info(
        f"Hybrid retrieved {len(filtered)}/{len(results)} chunks "
        f"(vector_weight={vector_weight}, bm25_weight={bm25_weight}, rerank={rerank_enabled})"
    )
    return filtered


def _prioritize_session_uploads(results: List[Dict], session_id: Optional[str]) -> List[Dict]:
    cfg = get_config()
    upload_boost = float(cfg.get("rag.user_upload_score_boost", 0.25))
    restrict_uploads = bool(cfg.get("rag.restrict_uploads_to_session", True))

    adjusted = []
    for result in results:
        meta = result.get("metadata") or {}
        is_upload = bool(meta.get("uploaded_file")) or meta.get("source_origin") == "user_upload"
        result_session = meta.get("session_id")

        if is_upload and restrict_uploads and session_id and result_session and result_session != session_id:
            continue

        item = result.copy()
        if is_upload and (not session_id or not result_session or result_session == session_id):
            base_score = float(
                item.get("base_hybrid_score", item.get("hybrid_score", item.get("score", 0.0))) or 0.0
            )
            boosted = base_score + upload_boost
            item["base_hybrid_score"] = base_score
            item["hybrid_score"] = boosted
            item["score"] = boosted
            item["retrieval_priority"] = "user_upload"
        else:
            item.setdefault("retrieval_priority", "data_main")

        adjusted.append(item)

    adjusted.sort(key=lambda item: item.get("hybrid_score", 0.0), reverse=True)
    return adjusted


def format_context(retrieved_chunks: List[Dict], max_tokens: int = 2000) -> str:
    if not retrieved_chunks:
        return ""

    lines = []
    total_words = 0
    token_limit_words = int(max_tokens / 1.3)

    for i, chunk in enumerate(retrieved_chunks, 1):
        text = chunk["text"]
        words = len(text.split())
        if total_words + words > token_limit_words:
            break

        meta = chunk.get("metadata", {})
        source = meta.get("source_file", "Unknown")
        page = meta.get("page")
        priority = chunk.get("retrieval_priority", "data_main")
        score = chunk.get("hybrid_score", chunk.get("score", 0.0))
        rerank_score = chunk.get("rerank_score")
        vector_score = chunk.get("vector_score", 0.0)
        bm25_score = chunk.get("bm25_score", 0.0)
        location = f", page {page}" if page else ""
        rerank_part = f", rerank={rerank_score:.3f}" if rerank_score is not None else ""

        lines.append(
            f"[Chunk {i} | Source: {source}{location} | "
            f"priority={priority}, hybrid={score:.3f}{rerank_part}, vector={vector_score:.3f}, bm25={bm25_score:.3f}]\n{text}"
        )
        total_words += words

    return "\n\n---\n\n".join(lines)
