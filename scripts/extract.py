"""
extract.py — CLI for the edu-agent data extraction pipeline.

Commands:
  ingest    — Extract documents and store in FAISS vectorstore
  stats     — Show vectorstore statistics
  delete    — Remove a source file from the vectorstore
  export    — Export chunks to JSONL
  reimport  — Rebuild vectorstore from a JSONL export
  search    — Quick test search against the vectorstore

Usage examples:
  python scripts/extract.py ingest data/raw/
  python scripts/extract.py ingest lecture.pdf --force
  python scripts/extract.py ingest data/ --ext .pdf .docx --tag subject=math
  python scripts/extract.py stats
  python scripts/extract.py delete data/raw/old_file.pdf
  python scripts/extract.py export backup/chunks.jsonl
  python scripts/extract.py reimport backup/chunks.jsonl
  python scripts/extract.py search "định lý pythagore" --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def cmd_ingest(args) -> None:
    from src.rag.pipeline import IngestPipeline

    extra_meta = {}
    if args.tag:
        for tag in args.tag:
            if "=" in tag:
                k, v = tag.split("=", 1)
                extra_meta[k.strip()] = v.strip()

    pipeline = IngestPipeline(
        store_path=args.store,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        embed_batch_size=args.batch,
        embed_cache_file=args.embed_cache,
        force_reingest=args.force,
    )
    pipeline.load()
    report = pipeline.run(
        path=args.path,
        extensions=args.ext or None,
        recursive=not args.no_recursive,
        extra_metadata=extra_meta or None,
    )
    report.print_summary()

    if report.failed > 0:
        sys.exit(1)


def cmd_stats(args) -> None:
    from src.rag.vectorstore import VectorStore

    vs = VectorStore(args.store)
    vs.load()
    stats = vs.stats()

    print("\n" + "═" * 52)
    print("  VectorStore Statistics")
    print("═" * 52)
    print(f"  Path:           {stats['store_path']}")
    print(f"  Total chunks:   {stats['total_chunks']:,}")
    print(f"  Source files:   {stats['total_sources']}")
    print(f"  Unique hashes:  {stats['unique_hashes']:,}")
    print(f"  FAISS index:    {stats.get('faiss_index')}")
    print(f"  FAISS vectors:  {stats.get('faiss_vectors', 0):,}")
    print(f"  BM25 docs:      {stats.get('bm25_documents', 0):,}")

    if stats.get("file_type_distribution"):
        print(f"\n  File types:")
        for ft, count in sorted(stats["file_type_distribution"].items(), key=lambda x: -x[1]):
            print(f"    {ft:<12} {count:>6,} chunks")

    if args.sources and stats["source_chunk_counts"]:
        print(f"\n  Source files ({stats['total_sources']}):")
        for src, count in sorted(stats["source_chunk_counts"].items(), key=lambda x: -x[1]):
            print(f"    [{count:>4}]  {Path(src).name}")

    print("═" * 52 + "\n")


def cmd_delete(args) -> None:
    from src.rag.pipeline import IngestPipeline

    pipeline = IngestPipeline(store_path=args.store)
    pipeline.load()
    removed = pipeline.delete_source(args.file)
    if removed:
        print(f"✓ Removed {removed} chunks for: {args.file}")
    else:
        print(f"! No chunks found for: {args.file}")


def cmd_export(args) -> None:
    from src.rag.pipeline import IngestPipeline

    pipeline = IngestPipeline(store_path=args.store)
    pipeline.load()
    count = pipeline.export(args.output)
    print(f"✓ Exported {count:,} chunks → {args.output}")


def cmd_reimport(args) -> None:
    from src.rag.pipeline import IngestPipeline

    pipeline = IngestPipeline(store_path=args.store)
    pipeline.load()
    count = pipeline.reimport(args.input)
    print(f"✓ Reimported {count:,} chunks from {args.input}")


def cmd_search(args) -> None:
    from src.rag.embedder import embed_query
    from src.rag.vectorstore import VectorStore

    vs = VectorStore(args.store)
    loaded = vs.load()
    if not loaded or vs.is_empty:
        print("VectorStore is empty. Run 'ingest' first.")
        return

    print(f"\nSearching: {args.query!r} (top_k={args.top_k})\n")
    q_emb = embed_query(args.query)
    results = vs.hybrid_search(
        query=args.query,
        query_embedding=q_emb,
        top_k=max(args.top_k, args.candidates),
        candidate_k=args.candidates,
    )
    if args.lang:
        results = [
            result for result in results
            if result.get("metadata", {}).get("language") == args.lang
        ]
    if args.rerank and results:
        from src.rag.reranker import rerank
        results = rerank(args.query, results, top_k=args.top_k)
    else:
        results = results[:args.top_k]

    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        print(f"── Result {i} ──────────────────────────────────────")
        print(f"   Hybrid:  {r.get('hybrid_score', 0):.4f}")
        print(f"   Vector:  {r.get('vector_score', 0):.4f}")
        print(f"   BM25:    {r.get('bm25_score', 0):.4f}")
        print(f"   Source:  {Path(meta.get('source_file', '?')).name}")
        print(f"   Page:    {meta.get('page', 'N/A')}")
        print(f"   Lang:    {meta.get('language', '?')}")
        print(f"   Text:    {r['text'][:300].strip()}")
        print()


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="extract.py",
        description="edu-agent data extraction pipeline",
    )
    root.add_argument(
        "--store",
        default=None,
        help="Path to vectorstore directory (default: from config.json)",
    )

    sub = root.add_subparsers(dest="command", required=True)

    # ── ingest ────────────────────────────────────────────────────────────────
    p_ingest = sub.add_parser("ingest", help="Extract documents and ingest into FAISS")
    p_ingest.add_argument("path", help="File or directory to ingest")
    p_ingest.add_argument(
        "--ext", nargs="+", metavar="EXT",
        help="File extensions to include, e.g. --ext .pdf .docx",
    )
    p_ingest.add_argument(
        "--force", action="store_true",
        help="Re-ingest even files already in the manifest",
    )
    p_ingest.add_argument(
        "--no-recursive", action="store_true",
        help="Do not recurse into subdirectories",
    )
    p_ingest.add_argument(
        "--chunk-size", type=int, default=None,
        help="Chunk size in words (default: from config)",
    )
    p_ingest.add_argument(
        "--overlap", type=int, default=None,
        help="Chunk overlap in words (default: from config)",
    )
    p_ingest.add_argument(
        "--batch", type=int, default=None,
        help="Embedding batch size",
    )
    p_ingest.add_argument(
        "--embed-cache", default=None, metavar="FILE",
        help="Path to embedding cache file (speeds up repeated runs)",
    )
    p_ingest.add_argument(
        "--tag", nargs="+", metavar="KEY=VALUE",
        help="Extra metadata tags, e.g. --tag subject=math grade=10",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # ── stats ─────────────────────────────────────────────────────────────────
    p_stats = sub.add_parser("stats", help="Show vectorstore statistics")
    p_stats.add_argument(
        "--sources", action="store_true",
        help="List all ingested source files",
    )
    p_stats.set_defaults(func=cmd_stats)

    # ── delete ────────────────────────────────────────────────────────────────
    p_del = sub.add_parser("delete", help="Remove a source file from the vectorstore")
    p_del.add_argument("file", help="Source file path to remove")
    p_del.set_defaults(func=cmd_delete)

    # ── export ────────────────────────────────────────────────────────────────
    p_exp = sub.add_parser("export", help="Export all chunks to JSONL")
    p_exp.add_argument("output", help="Output JSONL path")
    p_exp.set_defaults(func=cmd_export)

    # ── reimport ──────────────────────────────────────────────────────────────
    p_imp = sub.add_parser("reimport", help="Rebuild vectorstore from JSONL export")
    p_imp.add_argument("input", help="Input JSONL path")
    p_imp.set_defaults(func=cmd_reimport)

    # ── search ────────────────────────────────────────────────────────────────
    p_srch = sub.add_parser("search", help="Test search against the vectorstore")
    p_srch.add_argument("query", help="Query string")
    p_srch.add_argument("--top-k", type=int, default=5)
    p_srch.add_argument("--candidates", type=int, default=40)
    p_srch.add_argument("--rerank", action="store_true", help="Apply cross-encoder reranking")
    p_srch.add_argument("--lang", default=None, help="Filter by language: vi / en")
    p_srch.set_defaults(func=cmd_search)

    return root


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
