"""
Response cache.

Two layers:
  1. Exact-match LRU cache — identical query + session → instant replay
  2. Semantic cache — similar queries (cosine > threshold) → reuse answer

The semantic cache uses the same BGE embedding already loaded for RAG,
so there's no extra model cost.

Usage:
    cache = get_response_cache()
    hit = cache.get(query, session_id)
    if hit:
        return hit
    ...generate answer...
    cache.set(query, session_id, answer, metadata)
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger("cache")


@dataclass
class CacheEntry:
    query: str
    answer: str
    metadata: Dict
    embedding: Optional[np.ndarray]
    hits: int = 0
    created_at: float = field(default_factory=time.time)
    last_hit: float = field(default_factory=time.time)


class ResponseCache:
    """
    Two-tier response cache.

    Tier 1 — exact LRU (O(1)):
        key = md5(query.lower().strip() + "|" + session_id)
        Capacity: max_exact entries, evicts LRU

    Tier 2 — semantic similarity (O(N)):
        Scans all cached embeddings for cosine similarity > threshold.
        Only active when use_semantic=True and embedding model is loaded.
        Capacity: max_semantic entries (separate pool from exact cache).
    """

    def __init__(
        self,
        max_exact: int = 256,
        max_semantic: int = 128,
        semantic_threshold: float = 0.92,
        ttl_seconds: float = 3600.0,
        use_semantic: bool = True,
    ):
        self.max_exact          = max_exact
        self.max_semantic       = max_semantic
        self.semantic_threshold = semantic_threshold
        self.ttl_seconds        = ttl_seconds
        self.use_semantic       = use_semantic

        self._exact: OrderedDict[str, CacheEntry] = OrderedDict()
        self._semantic: List[CacheEntry] = []

        self._hits_exact    = 0
        self._hits_semantic = 0
        self._misses        = 0

    # ── Public ────────────────────────────────────────────────────────────────

    def get(
        self,
        query: str,
        session_id: str = "",
        embed_fn=None,
    ) -> Optional[Dict]:
        """
        Look up query in cache.

        Args:
            query: User query string.
            session_id: Session scoping (exact cache only).
            embed_fn: Callable(List[str]) → np.ndarray, for semantic lookup.

        Returns:
            {"answer": ..., "metadata": ..., "cache_hit": "exact"|"semantic"}
            or None on miss.
        """
        now = time.time()

        # ── Tier 1: exact ─────────────────────────────────────────────────────
        key = self._exact_key(query, session_id)
        if key in self._exact:
            entry = self._exact[key]
            if now - entry.created_at < self.ttl_seconds:
                self._exact.move_to_end(key)
                entry.hits += 1
                entry.last_hit = now
                self._hits_exact += 1
                logger.debug(f"[CACHE:exact] hit — '{query[:50]}'")
                return {"answer": entry.answer, "metadata": entry.metadata,
                        "cache_hit": "exact"}
            else:
                del self._exact[key]

        # ── Tier 2: semantic ──────────────────────────────────────────────────
        if self.use_semantic and embed_fn and self._semantic:
            q_emb = embed_fn([query])[0]
            best_entry, best_score = self._semantic_lookup(q_emb, now)
            if best_entry is not None:
                best_entry.hits += 1
                best_entry.last_hit = now
                self._hits_semantic += 1
                logger.debug(f"[CACHE:semantic] hit score={best_score:.3f} — '{query[:50]}'")
                return {"answer": best_entry.answer, "metadata": best_entry.metadata,
                        "cache_hit": "semantic", "similarity": best_score}

        self._misses += 1
        return None

    def set(
        self,
        query: str,
        answer: str,
        metadata: Dict,
        session_id: str = "",
        embed_fn=None,
    ) -> None:
        """
        Store a new response in both cache tiers.

        Args:
            query: The query that produced this answer.
            answer: The generated answer.
            metadata: Metadata dict from workflow (category, model_used, etc.).
            session_id: Session scoping for exact cache.
            embed_fn: Callable to produce embedding for semantic cache.
        """
        # Don't cache empty or error answers
        if not answer or len(answer) < 10 or answer.startswith("[Execution failed"):
            return

        embedding = None
        if self.use_semantic and embed_fn:
            try:
                embedding = embed_fn([query])[0]
            except Exception:
                pass

        entry = CacheEntry(
            query=query,
            answer=answer,
            metadata=metadata,
            embedding=embedding,
        )

        # Exact cache
        key = self._exact_key(query, session_id)
        self._exact[key] = entry
        self._exact.move_to_end(key)
        if len(self._exact) > self.max_exact:
            self._exact.popitem(last=False)  # evict oldest

        # Semantic cache (global, no session scoping)
        if embedding is not None:
            self._semantic.append(entry)
            if len(self._semantic) > self.max_semantic:
                # Evict lowest-hit entry
                self._semantic.sort(key=lambda e: e.hits)
                self._semantic = self._semantic[1:]

        logger.debug(f"[CACHE:set] stored — '{query[:50]}'")

    def invalidate(self, session_id: str) -> int:
        """Remove all exact-cache entries for a session."""
        prefix = f"|{session_id}"
        keys = [k for k in self._exact if k.endswith(prefix)]
        for k in keys:
            del self._exact[k]
        return len(keys)

    def clear(self) -> None:
        self._exact.clear()
        self._semantic.clear()
        self._hits_exact = self._hits_semantic = self._misses = 0

    @property
    def stats(self) -> Dict:
        total = self._hits_exact + self._hits_semantic + self._misses
        hit_rate = (self._hits_exact + self._hits_semantic) / max(total, 1)
        return {
            "exact_entries":    len(self._exact),
            "semantic_entries": len(self._semantic),
            "hits_exact":       self._hits_exact,
            "hits_semantic":    self._hits_semantic,
            "misses":           self._misses,
            "hit_rate":         round(hit_rate, 3),
        }

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _exact_key(query: str, session_id: str) -> str:
        raw = query.lower().strip() + "|" + session_id
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _semantic_lookup(
        self,
        q_emb: np.ndarray,
        now: float,
    ) -> Tuple[Optional[CacheEntry], float]:
        """Scan semantic entries for cosine similarity above threshold."""
        best_entry = None
        best_score = 0.0

        live = [e for e in self._semantic
                if now - e.created_at < self.ttl_seconds and e.embedding is not None]

        if not live:
            return None, 0.0

        # Stack all embeddings for vectorised dot product
        emb_matrix = np.stack([e.embedding for e in live])  # (N, dim)
        scores = emb_matrix @ q_emb                          # (N,) cosine (pre-normalised)

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= self.semantic_threshold:
            return live[best_idx], best_score

        return None, best_score


# ── Module-level singleton ────────────────────────────────────────────────────

_cache: Optional[ResponseCache] = None


def get_response_cache() -> ResponseCache:
    global _cache
    if _cache is None:
        cfg = get_config()
        _cache = ResponseCache(
            max_exact=cfg.get("cache.max_exact", 256),
            max_semantic=cfg.get("cache.max_semantic", 128),
            semantic_threshold=cfg.get("cache.semantic_threshold", 0.92),
            ttl_seconds=cfg.get("cache.ttl_seconds", 3600.0),
            use_semantic=cfg.get("cache.use_semantic", True),
        )
    return _cache
