"""
Build the hybrid vector database.

Default source is data_main and the store is configured by config/config.json.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger("ingest_script")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS IVF+HNSW + BM25 hybrid vector database")
    parser.add_argument("--path", "-p", type=str, default="data_main", help="File or directory to ingest")
    parser.add_argument("--ext", nargs="+", default=None, help="File extensions to include, e.g. .pdf .docx .ipynb")
    parser.add_argument("--clear", action="store_true", help="Clear existing vector database before ingesting")
    parser.add_argument("--force", action="store_true", help="Re-ingest files even if unchanged")
    parser.add_argument("--show-stats", action="store_true", help="Print vector database stats and exit")
    args = parser.parse_args()

    from src.rag.pipeline import run_pipeline
    from src.rag.vectorstore import VectorStore
    from src.utils.config_loader import get_config

    cfg = get_config()
    store_path = cfg.get("rag.vectorstore_path", "vectorstore/faiss_index")

    if args.show_stats:
        vs = VectorStore(store_path)
        loaded = vs.load()
        print(vs.stats() if loaded else "Hybrid vector database: empty")
        return

    if args.clear and os.path.exists(store_path):
        shutil.rmtree(store_path)
        logger.info(f"Cleared vector database at {store_path}")

    if not os.path.exists(args.path):
        print(f"[Error] Path not found: {args.path}")
        sys.exit(1)

    report = run_pipeline(args.path, extensions=args.ext, force=args.force)
    report.print_summary()

    vs = VectorStore(store_path)
    vs.load()
    print(f"Done. Hybrid vector database now contains {vs.size} chunks at {store_path}.")


if __name__ == "__main__":
    main()
