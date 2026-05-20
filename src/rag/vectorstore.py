"""
Hybrid FAISS + BM25 vector database.

FAISS index:
  - IVF for coarse partitioning
  - HNSW quantizer for faster centroid routing
  - Inner product metric, intended for normalized embeddings

Lexical index:
  - Lightweight in-repo BM25 implementation
  - Persisted next to the FAISS index
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger("vectorstore")


@dataclass
class ManifestEntry:
    source_file: str
    file_size: int
    mtime: float
    chunk_count: int
    ingested_at: float
    content_hashes: List[str]


class BM25Index:
    """Small BM25 implementation to avoid adding a retrieval dependency."""

    TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: List[Counter] = []
        self.doc_lengths: List[int] = []
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = unicodedata.normalize("NFD", text.lower())
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        return BM25Index.TOKEN_RE.findall(text)

    def build(self, texts: Iterable[str]) -> None:
        self.doc_freqs = []
        self.doc_lengths = []
        df = Counter()

        for text in texts:
            tokens = self.tokenize(text)
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            self.doc_lengths.append(len(tokens))
            df.update(freqs.keys())

        n_docs = len(self.doc_freqs)
        self.avgdl = sum(self.doc_lengths) / max(n_docs, 1)
        self.idf = {
            token: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in df.items()
        }

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if not self.doc_freqs:
            return []

        query_terms = self.tokenize(query)
        if not query_terms:
            return []

        scores = np.zeros(len(self.doc_freqs), dtype=np.float32)
        for term in query_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx, freqs in enumerate(self.doc_freqs):
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                dl = self.doc_lengths[idx] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-6))
                scores[idx] += idf * (tf * (self.k1 + 1) / denom)

        if scores.size == 0:
            return []

        limit = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:limit]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


class VectorStore:
    INDEX_FILE = "index.faiss"
    META_FILE = "metadata.pkl"
    CHUNKS_FILE = "chunks.jsonl"
    MANIFEST_FILE = "manifest.json"
    HASHES_FILE = "hashes.pkl"
    BM25_FILE = "bm25.pkl"

    def __init__(self, store_path: Optional[str] = None):
        cfg = get_config()
        self.store_path = Path(store_path or cfg.get("rag.vectorstore_path", "vectorstore/faiss_index"))
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.index = None
        self.chunks: List[Dict] = []
        self.manifest: Dict[str, ManifestEntry] = {}
        self.seen_hashes: Set[str] = set()
        self.bm25 = BM25Index(
            k1=cfg.get("rag.bm25.k1", 1.5),
            b=cfg.get("rag.bm25.b", 0.75),
        )
        self._gpu_resources = None
        self._using_gpu = False
        self._loaded = False

    def load(self) -> bool:
        import faiss

        idx_path = self.store_path / self.INDEX_FILE
        meta_path = self.store_path / self.META_FILE
        if not idx_path.exists() or not meta_path.exists():
            logger.info("No existing hybrid vector database found")
            return False

        try:
            self.index = self._maybe_to_gpu(faiss.read_index(str(idx_path)))
            with open(meta_path, "rb") as f:
                self.chunks = pickle.load(f)
            self._ensure_vector_ids()
            self._load_manifest()
            self._load_hashes()
            self._load_or_rebuild_bm25()
            self._loaded = True
            logger.info(
                f"Hybrid vector database loaded: {self.size} chunks | "
                f"faiss_vectors={self.index.ntotal if self.index else 0} | "
                f"device={'gpu' if self._using_gpu else 'cpu'}"
            )
            return True
        except Exception as exc:
            logger.error(f"Failed to load hybrid vector database: {exc}")
            return False

    def save(self) -> None:
        import faiss

        if self.index is not None:
            faiss.write_index(self._index_for_io(), str(self.store_path / self.INDEX_FILE))

        with open(self.store_path / self.META_FILE, "wb") as f:
            pickle.dump(self.chunks, f)

        with open(self.store_path / self.CHUNKS_FILE, "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        self._save_manifest()
        self._save_hashes()
        self._save_bm25()
        logger.info(f"Hybrid vector database saved: {self.size} chunks -> {self.store_path}")

    def _init_index(self, dim: int, train_embeddings: np.ndarray):
        import faiss

        cfg = get_config()
        n_vectors = max(int(train_embeddings.shape[0]), 1)
        nlist_cfg = int(cfg.get("rag.faiss.nlist", 128))
        hnsw_m = int(cfg.get("rag.faiss.hnsw_m", 32))
        nprobe = int(cfg.get("rag.faiss.nprobe", 8))

        nlist = max(1, min(nlist_cfg, n_vectors))
        quantizer = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(train_embeddings.astype(np.float32))
        index.nprobe = min(nprobe, nlist)
        self.index = self._maybe_to_gpu(index)
        logger.info(
            "Initialized FAISS IndexIVFFlat+HNSW quantizer: "
            f"dim={dim}, nlist={nlist}, nprobe={index.nprobe}, "
            f"device={'gpu' if self._using_gpu else 'cpu'}"
        )

    def _want_faiss_gpu(self) -> bool:
        cfg = get_config()
        device = str(cfg.get("rag.faiss.device", "cuda")).lower()
        return device.startswith("cuda") or device == "gpu"

    def _maybe_to_gpu(self, index):
        import faiss
        
        cfg = get_config()
        device = str(cfg.get("rag.faiss.device", "cuda")).lower()
        
        # CUDA enforcement: must use GPU
        if not device.startswith("cuda") and device != "gpu":
            raise ValueError(
                f"FAISS device must be 'cuda' or 'gpu', got '{device}'. "
                "CPU-only FAISS is not supported."
            )
        
        if not hasattr(faiss, "StandardGpuResources") or not hasattr(faiss, "index_cpu_to_gpu"):
            raise RuntimeError(
                "FAISS GPU support required but not available. "
                "Please install faiss-gpu: pip install faiss-gpu"
            )

        try:
            gpu_id = int(cfg.get("rag.faiss.gpu_id", 0))
            self._gpu_resources = faiss.StandardGpuResources()
            options = faiss.GpuClonerOptions()
            if hasattr(options, "allowCpuCoarseQuantizer"):
                options.allowCpuCoarseQuantizer = True
            gpu_index = faiss.index_cpu_to_gpu(self._gpu_resources, gpu_id, index, options)
            self._using_gpu = True
            logger.info(f"Moved FAISS index to GPU {gpu_id} (CUDA enforcement enabled)")
            return gpu_index
        except Exception as exc:
            raise RuntimeError(
                f"Failed to move FAISS index to GPU {gpu_id}: {exc}. "
                "Ensure GPU has sufficient VRAM and FAISS GPU support is properly installed."
            )

    def _index_for_io(self):
        if not self._using_gpu:
            return self.index

        import faiss

        if hasattr(faiss, "index_gpu_to_cpu"):
            return faiss.index_gpu_to_cpu(self.index)
        return self.index

    def add(self, chunks: List[Dict], embeddings: np.ndarray) -> int:
        if not chunks or embeddings is None or len(embeddings) == 0:
            return 0

        keep_idx = []
        for i, chunk in enumerate(chunks):
            content_hash = chunk.get("content_hash")
            if content_hash and content_hash in self.seen_hashes:
                continue
            keep_idx.append(i)
            if content_hash:
                self.seen_hashes.add(content_hash)

        if not keep_idx:
            logger.debug("All chunks already indexed")
            return 0

        filtered_chunks = [chunks[i] for i in keep_idx]
        filtered_embeddings = embeddings[keep_idx].astype(np.float32)

        if self.index is None:
            self._init_index(filtered_embeddings.shape[1], filtered_embeddings)
        elif hasattr(self.index, "is_trained") and not self.index.is_trained:
            self.index.train(filtered_embeddings)

        start_id = len(self.chunks)
        for offset, chunk in enumerate(filtered_chunks):
            chunk["vector_id"] = start_id + offset

        self.index.add(filtered_embeddings)
        self.chunks.extend(filtered_chunks)
        self._rebuild_bm25()
        logger.info(f"Added {len(filtered_chunks)} chunks to hybrid vector database")
        return len(filtered_chunks)

    def vector_search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        if self.index is None or self.index.ntotal == 0:
            return []

        q = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        fetch_k = min(max(top_k, 1), self.index.ntotal)
        scores, indices = self.index.search(q, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx].copy()
            chunk["vector_score"] = float(score)
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    def bm25_search(self, query: str, top_k: int = 10) -> List[Dict]:
        results = []
        for idx, score in self.bm25.search(query, top_k):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx].copy()
            chunk["bm25_score"] = score
            chunk["score"] = score
            results.append(chunk)
        return results

    def hybrid_search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        vector_weight: float = 0.65,
        bm25_weight: float = 0.35,
        candidate_k: Optional[int] = None,
    ) -> List[Dict]:
        candidate_k = candidate_k or max(top_k * 4, 20)
        vector_hits = self.vector_search(query_embedding, top_k=candidate_k)
        bm25_hits = self.bm25_search(query, top_k=candidate_k)

        vector_scores = {hit.get("vector_id", hit.get("chunk_id")): hit.get("vector_score", 0.0) for hit in vector_hits}
        bm25_scores = {hit.get("vector_id", hit.get("chunk_id")): hit.get("bm25_score", 0.0) for hit in bm25_hits}
        by_id = {}
        for hit in vector_hits + bm25_hits:
            key = hit.get("vector_id", hit.get("chunk_id"))
            by_id[key] = hit

        vector_norm = _minmax(vector_scores)
        bm25_norm = _minmax(bm25_scores)

        merged = []
        for key, chunk in by_id.items():
            v_score = vector_scores.get(key, 0.0)
            b_score = bm25_scores.get(key, 0.0)
            hybrid_score = vector_weight * vector_norm.get(key, 0.0) + bm25_weight * bm25_norm.get(key, 0.0)
            out = chunk.copy()
            out["vector_score"] = float(v_score)
            out["bm25_score"] = float(b_score)
            out["hybrid_score"] = float(hybrid_score)
            out["score"] = float(hybrid_score)
            out["retrieval_method"] = "hybrid_bm25_vector"
            merged.append(out)

        merged.sort(key=lambda item: item["hybrid_score"], reverse=True)
        return merged[:top_k]

    def is_already_ingested(self, file_path: str, force: bool = False) -> bool:
        if force or file_path not in self.manifest:
            return False
        entry = self.manifest[file_path]
        try:
            stat = os.stat(file_path)
            return stat.st_size == entry.file_size and abs(stat.st_mtime - entry.mtime) < 1.0
        except OSError:
            return False

    def record_ingestion(self, file_path: str, chunk_count: int, content_hashes: List[str]) -> None:
        try:
            stat = os.stat(file_path)
            self.manifest[file_path] = ManifestEntry(
                source_file=file_path,
                file_size=stat.st_size,
                mtime=stat.st_mtime,
                chunk_count=chunk_count,
                ingested_at=time.time(),
                content_hashes=content_hashes,
            )
        except OSError as exc:
            logger.warning(f"Could not record manifest for {file_path}: {exc}")

    def delete_source(self, file_path: str) -> int:
        before = len(self.chunks)
        removed_hashes = {
            chunk.get("content_hash")
            for chunk in self.chunks
            if chunk.get("metadata", {}).get("source_file") == file_path
        }
        self.chunks = [
            chunk for chunk in self.chunks
            if chunk.get("metadata", {}).get("source_file") != file_path
        ]
        removed = before - len(self.chunks)
        if removed:
            self.seen_hashes.difference_update({h for h in removed_hashes if h})
            self.manifest.pop(file_path, None)
            self._mark_index_stale()
            self._rebuild_bm25()
        return removed

    def stats(self) -> Dict:
        source_counts: Dict[str, int] = {}
        type_counts: Dict[str, int] = {}
        for chunk in self.chunks:
            meta = chunk.get("metadata", {})
            source = meta.get("source_file", "unknown")
            file_type = meta.get("file_type", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
            type_counts[file_type] = type_counts.get(file_type, 0) + 1

        return {
            "total_chunks": self.size,
            "total_sources": len(self.manifest),
            "unique_hashes": len(self.seen_hashes),
            "faiss_index": type(self.index).__name__ if self.index is not None else None,
            "faiss_device": "gpu" if self._using_gpu else "cpu",
            "faiss_vectors": self.index.ntotal if self.index is not None else 0,
            "bm25_documents": len(self.bm25.doc_freqs),
            "file_type_distribution": type_counts,
            "source_chunk_counts": source_counts,
            "store_path": str(self.store_path),
        }

    def export_jsonl(self, output_path: str) -> int:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        return len(self.chunks)

    def import_jsonl(self, input_path: str, embed_fn, batch_size: int = 64) -> int:
        chunks = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

        total_added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embeddings = embed_fn([chunk["text"] for chunk in batch])
            total_added += self.add(batch, embeddings)
        self.save()
        return total_added

    def _load_manifest(self) -> None:
        path = self.store_path / self.MANIFEST_FILE
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.manifest = {key: ManifestEntry(**value) for key, value in raw.items()}

    def _ensure_vector_ids(self) -> None:
        for idx, chunk in enumerate(self.chunks):
            chunk.setdefault("vector_id", idx)

    def _save_manifest(self) -> None:
        with open(self.store_path / self.MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump({key: asdict(value) for key, value in self.manifest.items()}, f, indent=2, ensure_ascii=False)

    def _load_hashes(self) -> None:
        path = self.store_path / self.HASHES_FILE
        if path.exists():
            with open(path, "rb") as f:
                self.seen_hashes = pickle.load(f)

    def _save_hashes(self) -> None:
        with open(self.store_path / self.HASHES_FILE, "wb") as f:
            pickle.dump(self.seen_hashes, f)

    def _load_or_rebuild_bm25(self) -> None:
        path = self.store_path / self.BM25_FILE
        if path.exists():
            with open(path, "rb") as f:
                self.bm25 = pickle.load(f)
        else:
            self._rebuild_bm25()
            self._save_bm25()

    def _save_bm25(self) -> None:
        with open(self.store_path / self.BM25_FILE, "wb") as f:
            pickle.dump(self.bm25, f)

    def _rebuild_bm25(self) -> None:
        self.bm25.build(chunk.get("text", "") for chunk in self.chunks)

    def _mark_index_stale(self) -> None:
        logger.warning("FAISS index is stale after deletion; rebuild the vector database to sync vectors.")

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        return self.size == 0


def _minmax(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low, high = min(values), max(values)
    if math.isclose(high, low):
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


_store: Optional[VectorStore] = None


def get_vectorstore(store_path: Optional[str] = None, reload: bool = False) -> VectorStore:
    global _store
    if _store is None or reload:
        _store = VectorStore(store_path)
        _store.load()
    return _store
