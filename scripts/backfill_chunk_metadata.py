"""
Backfill rich metadata onto every persisted vectorstore chunk.

This updates metadata.pkl and chunks.jsonl only. The FAISS index is unchanged
because vector ids and embeddings are preserved.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config_loader import get_config


def _source_info(source_file: str) -> Dict:
    path = Path(source_file)
    exists = path.exists()
    stat = path.stat() if exists else None
    return {
        "source_dir": str(path.parent),
        "source_stem": path.stem,
        "source_ext": path.suffix.lower(),
        "source_exists": exists,
        "source_size_bytes": stat.st_size if stat else None,
        "source_mtime": stat.st_mtime if stat else None,
    }


def _normalize_metadata(chunk: Dict, source_counts: Counter, source_offsets: Counter) -> Dict:
    metadata = dict(chunk.get("metadata") or {})

    source_file = metadata.get("source_file") or chunk.get("source_file") or ""
    file_name = metadata.get("file_name") or (Path(source_file).name if source_file else "")
    file_type = metadata.get("file_type") or Path(file_name).suffix.lower().lstrip(".") or "unknown"
    vector_id = chunk.get("vector_id", chunk.get("chunk_id"))

    source_offsets[source_file] += 1
    metadata.update(
        {
            "source_file": source_file,
            "file_name": file_name,
            "file_type": file_type,
            "language": metadata.get("language") or "unknown",
            "title": metadata.get("title") or Path(file_name).stem,
            "chunk_id": chunk.get("chunk_id"),
            "vector_id": vector_id,
            "chunk_index_in_source": source_offsets[source_file] - 1,
            "source_chunk_count": source_counts[source_file],
            "content_hash": chunk.get("content_hash"),
            "char_start": chunk.get("char_start"),
            "word_count": chunk.get("word_count"),
        }
    )
    metadata.update(_source_info(source_file) if source_file else {})
    return metadata


def main() -> None:
    cfg = get_config()
    store_path = Path(cfg.get("rag.vectorstore_path", "vectorstore/faiss_index"))
    metadata_path = store_path / "metadata.pkl"
    chunks_path = store_path / "chunks.jsonl"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    with open(metadata_path, "rb") as f:
        chunks: List[Dict] = pickle.load(f)

    source_counts = Counter(
        (chunk.get("metadata") or {}).get("source_file") or chunk.get("source_file") or ""
        for chunk in chunks
    )
    source_offsets: Counter = Counter()

    for idx, chunk in enumerate(chunks):
        chunk.setdefault("chunk_id", idx)
        chunk.setdefault("vector_id", idx)
        chunk["metadata"] = _normalize_metadata(chunk, source_counts, source_offsets)

    with open(metadata_path, "wb") as f:
        pickle.dump(chunks, f)

    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Updated metadata for {len(chunks):,} chunks in {store_path}")


if __name__ == "__main__":
    main()
