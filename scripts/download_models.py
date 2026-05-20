"""
Download all required models from HuggingFace Hub.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --only llm
    python scripts/download_models.py --only embedding reranker
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger("download_models")

MODELS = {
    "llm_executor":   {"repo_id": "Qwen/Qwen2.5-3B-Instruct",    "local_dir": "models/llm/qwen2.5-3b-instruct",  "group": "llm"},
    "llm_classifier": {"repo_id": "Qwen/Qwen2.5-0.5B-Instruct",  "local_dir": "models/llm/qwen2.5-0.5b-instruct","group": "llm"},
    "embedding":     {"repo_id": "BAAI/bge-base-en-v1.5",        "local_dir": "models/embedding/bge-base-en-v1.5",  "group": "embedding"},
    "reranker":      {"repo_id": "BAAI/bge-reranker-base",       "local_dir": "models/reranker/bge-reranker-base", "group": "reranker"},
}


def download_model(repo_id: str, local_dir: str) -> None:
    from huggingface_hub import snapshot_download

    os.makedirs(local_dir, exist_ok=True)
    logger.info(f"Downloading {repo_id} → {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )
    logger.info(f"✓ {repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=["llm", "embedding", "reranker"], default=None)
    parser.add_argument("--token", type=str, default=None)
    args = parser.parse_args()

    if args.token:
        from huggingface_hub import login
        login(token=args.token)

    filter_groups = set(args.only) if args.only else None
    for name, info in MODELS.items():
        if filter_groups and info["group"] not in filter_groups:
            continue
        try:
            download_model(info["repo_id"], info["local_dir"])
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")

    logger.info("All downloads complete.")


if __name__ == "__main__":
    main()
