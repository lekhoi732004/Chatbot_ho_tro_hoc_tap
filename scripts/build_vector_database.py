"""
Build the production hybrid vector database from data_main.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Build hybrid BM25 + FAISS IVF/HNSW vector database")
    parser.add_argument("--data", default="data_main", help="Source data folder")
    parser.add_argument("--clear", action="store_true", help="Remove existing vector database first")
    parser.add_argument("--force", action="store_true", help="Re-ingest unchanged files")
    args = parser.parse_args()

    from src.rag.pipeline import run_pipeline
    from src.rag.vectorstore import VectorStore
    from src.utils.config_loader import get_config

    cfg = get_config()
    store_path = Path(cfg.get("rag.vectorstore_path", "vectorstore/faiss_index"))
    if args.clear and store_path.exists():
        shutil.rmtree(store_path)

    report = run_pipeline(args.data, force=args.force)
    report.print_summary()

    vs = VectorStore(str(store_path))
    vs.load()
    print(vs.stats())


if __name__ == "__main__":
    main()
