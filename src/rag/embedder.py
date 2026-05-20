"""
Embedding manager.

Wraps sentence-transformers with:
  - Singleton model cache
  - Batched encoding with tqdm progress
  - Automatic retry on OOM / CUDA errors
  - Optional embedding cache (hash → vector) to avoid re-encoding duplicate texts
"""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger("embedder")

_model = None
_embed_cache: Dict[str, np.ndarray] = {}
_cache_path: Optional[str] = None


def _normalize_device(device: str) -> str:
    device = str(device).strip().lower()
    if device == "gpu":
        return "cuda"
    return device


def _get_model():
    global _model
    if _model is not None:
        return _model

    from sentence_transformers import SentenceTransformer

    import torch
    
    cfg = get_config()
    model_path = cfg.get("embedding.model_path", "models/embedding/bge-base-en-v1.5")
    requested_device = _normalize_device(cfg.get("embedding.device", "cuda"))
    device = requested_device
    
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA not available! Cannot load embedding model {model_path} on CUDA. "
            "Please install a CUDA-enabled PyTorch build and check NVIDIA drivers."
        )

    logger.info(f"Loading embedding model: {model_path} (device={device})")
    _model = SentenceTransformer(model_path, device=device)
    logger.info(f"Embedding dim: {_model.get_sentence_embedding_dimension()}")
    return _model


def _cache_key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_embed_cache(cache_file: str) -> None:
    """Load an on-disk embedding cache (optional speedup for repeated runs)."""
    global _embed_cache, _cache_path
    _cache_path = cache_file
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            _embed_cache = pickle.load(f)
        logger.info(f"Loaded embed cache: {len(_embed_cache)} entries from {cache_file}")


def save_embed_cache() -> None:
    """Flush embedding cache to disk."""
    if _cache_path and _embed_cache:
        Path(_cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(_cache_path, "wb") as f:
            pickle.dump(_embed_cache, f)
        logger.info(f"Saved embed cache: {len(_embed_cache)} entries → {_cache_path}")


def embed_texts(
    texts: List[str],
    batch_size: Optional[int] = None,
    show_progress: bool = True,
    use_cache: bool = False,
    max_retries: int = 2,
) -> np.ndarray:
    """
    Embed a list of texts using the BGE embedding model.

    Args:
        texts: List of strings to embed.
        batch_size: Encoding batch size (defaults to config value).
        show_progress: Show tqdm progress bar for large batches.
        use_cache: Use in-memory hash cache to skip duplicate texts.
        max_retries: Retry on CUDA OOM by halving batch size.

    Returns:
        numpy float32 array of shape (N, embedding_dim).
    """
    if not texts:
        return np.empty((0, 768), dtype=np.float32)

    cfg = get_config()
    batch_size = batch_size or cfg.get("embedding.batch_size", 32)
    normalize  = cfg.get("embedding.normalize", True)

    model = _get_model()

    # ── Cache lookup ──────────────────────────────────────────────────────────
    if use_cache and _embed_cache:
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts   = []

        for i, text in enumerate(texts):
            key = _cache_key(text)
            if key in _embed_cache:
                results[i] = _embed_cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if not uncached_texts:
            return np.stack(results)

        logger.debug(f"Cache: {len(texts)-len(uncached_texts)} hits, {len(uncached_texts)} misses")
        new_embs = _encode_with_retry(
            model, uncached_texts, batch_size, normalize, show_progress, max_retries
        )
        for i, emb in zip(uncached_indices, new_embs):
            _embed_cache[_cache_key(texts[i])] = emb
            results[i] = emb
        return np.stack(results)

    # ── Direct encode ─────────────────────────────────────────────────────────
    return _encode_with_retry(model, texts, batch_size, normalize, show_progress, max_retries)


def _encode_with_retry(
    model,
    texts: List[str],
    batch_size: int,
    normalize: bool,
    show_progress: bool,
    max_retries: int,
) -> np.ndarray:
    """Encode with automatic retry on OOM by halving batch size."""
    attempt = 0
    current_batch = batch_size

    while attempt <= max_retries:
        try:
            embeddings = model.encode(
                texts,
                batch_size=current_batch,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress and len(texts) > 50,
                convert_to_numpy=True,
            )
            return embeddings.astype(np.float32)

        except RuntimeError as e:
            if "out of memory" in str(e).lower() and current_batch > 1:
                current_batch = max(1, current_batch // 2)
                logger.warning(f"OOM — retrying with batch_size={current_batch}")
                attempt += 1
            else:
                raise

    raise RuntimeError(f"Embedding failed after {max_retries} retries")


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns:
        1-D numpy float32 array of shape (embedding_dim,).
    """
    # BGE models benefit from a query prefix
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    result = embed_texts([prefixed], show_progress=False)
    return result[0]
