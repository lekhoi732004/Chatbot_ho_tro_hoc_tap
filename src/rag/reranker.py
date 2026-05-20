"""
Cross-encoder reranker for hybrid retrieval candidates.

The retriever first collects candidates with BM25 + vector search, then this
module reranks the candidates for final precision before context formatting.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger("reranker")

_reranker_model = None


def _normalize_device(device: str) -> str:
    device = str(device).strip().lower()
    if device == "gpu":
        return "cuda"
    return device


def _get_reranker():
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    
    cfg = get_config()
    model_path = cfg.get("reranker.model_path", "models/reranker/bge-reranker-base")
    requested_device = _normalize_device(cfg.get("reranker.device", "cuda"))
    device = requested_device
    
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA not available! Cannot load reranker model {model_path} on CUDA. "
            "Please install a CUDA-enabled PyTorch build and check NVIDIA drivers."
        )

    logger.info(f"Loading reranker: {model_path} (device={device})")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    model.to(device)

    _reranker_model = (tokenizer, model, device)
    return _reranker_model


def rerank(query: str, chunks: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
    if not chunks:
        return []

    import torch

    cfg = get_config()
    top_k = top_k or cfg.get("reranker.top_k", cfg.get("rag.top_k_rerank", 5))
    batch_size = cfg.get("reranker.batch_size", 16)
    max_length = cfg.get("reranker.max_length", 512)

    tokenizer, model, device = _get_reranker()
    pairs = [(query, chunk.get("text", "")) for chunk in chunks]

    scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        encoded = tokenizer(
            [item[0] for item in batch],
            [item[1] for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            logits = model(**encoded).logits.squeeze(-1)
            batch_scores = torch.sigmoid(logits).detach().cpu().tolist()

        if isinstance(batch_scores, float):
            batch_scores = [batch_scores]
        scores.extend(batch_scores)

    reranked = []
    for chunk, score in zip(chunks, scores):
        item = chunk.copy()
        item["rerank_score"] = float(score)
        item["pre_rerank_score"] = float(item.get("score", item.get("hybrid_score", 0.0)))
        item["score"] = float(score)
        item["retrieval_method"] = "hybrid_bm25_vector_reranked"
        reranked.append(item)

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    result = reranked[:top_k]
    logger.info(f"Reranked {len(chunks)} hybrid candidates -> top {len(result)}")
    return result
