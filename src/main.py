"""
edu-agent main entrypoint — with response cache + parallel workflow.

Usage:
    python -m src.main                        # interactive chat
    python -m src.main --query "your question"
    python -m src.main --session my_session
    python -m src.main --file doc.pdf
    python -m src.main --stats                # show cache + memory stats
"""

from __future__ import annotations

import argparse
import sys
import uuid

from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger("main")


def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║              EDU-AGENT  |  Multi-Agent LLM           ║
║   Qwen2.5 · RAG · LangGraph · BGE · Fast Cache       ║
╚══════════════════════════════════════════════════════╝
Type your question. Commands: /exit /clear /history /ingest <path> /stats /help
""")


def print_help():
    print("""
Commands:
  /exit              — Exit
  /clear             — Clear conversation history
  /history           — Show scored message history
  /stats             — Cache + memory stats
  /ingest <path>     — Ingest a file or directory
  /session <id>      — Switch session
  /help              — This help
""")


def run_single_query(query: str, session_id: str) -> None:
    from src.graph.workflow import run_workflow
    from src.utils.cache import get_response_cache
    from src.rag.embedder import embed_texts

    cfg   = get_config()
    cache = get_response_cache()

    # Cache lookup
    if cfg.get("cache.enabled", True):
        hit = cache.get(query, session_id, embed_fn=embed_texts if cfg.get("cache.use_semantic") else None)
        if hit:
            print(f"\n{hit['answer']}\n")
            return

    result = run_workflow(query, session_id)
    answer = result.get("answer", "[Không có câu trả lời]")

    print(f"\n{answer}\n")
    # Store in cache
    if cfg.get("cache.enabled", True):
        cache.set(query, answer, result.get("metadata", {}), session_id,
                  embed_fn=embed_texts if cfg.get("cache.use_semantic") else None)


def handle_command(cmd: str, session_id: str) -> str:
    parts   = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    arg     = parts[1] if len(parts) > 1 else ""

    if command == "/exit":
        print("Goodbye!")
        sys.exit(0)

    elif command == "/clear":
        from src.memory.memory_manager import get_memory_manager
        from src.utils.cache import get_response_cache
        get_memory_manager().clear_history(session_id)
        get_response_cache().invalidate(session_id)
        print("[Memory and cache cleared]")

    elif command == "/history":
        from src.memory.memory_manager import get_memory_manager
        mm = get_memory_manager()
        msgs = mm.get_scored_history(session_id)
        if not msgs:
            print("[No history]")
        else:
            print(f"\n{'msg_id':<10} {'role':<12} {'score':<7} {'keep':<6} {'content'}")
            print("─" * 72)
            for m in msgs:
                keep_flag = "✓" if m.keep else "✗"
                print(f"{m.msg_id:<10} {m.role:<12} {m.memory_score:<7.2f} {keep_flag:<6} {m.content[:50]}")

    elif command == "/stats":
        from src.utils.cache import get_response_cache
        from src.memory.memory_manager import get_memory_manager
        cache = get_response_cache()
        mm    = get_memory_manager()
        print(f"\nCache: {cache.stats}")
        print(f"Memory: {mm.active_sessions} active sessions")

    elif command == "/ingest":
        if not arg:
            print("[!] Usage: /ingest <file_or_directory>")
        else:
            import os
            from src.rag.pipeline import run_pipeline
            report = run_pipeline(arg)
            report.print_summary()

    elif command == "/session":
        if arg:
            from src.memory.memory_manager import get_memory_manager
            get_memory_manager().create_session(arg)
            session_id = arg
            print(f"[Session: {session_id}]")
        else:
            print(f"[Current session: {session_id}]")

    elif command == "/help":
        print_help()
    else:
        print(f"[Unknown: {command}] Type /help")

    return session_id


def interactive_loop(session_id: str) -> None:
    print_banner()
    print(f"Session: {session_id}\n")
    from src.memory.memory_manager import get_memory_manager
    get_memory_manager().create_session(session_id)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.startswith("/"):
            session_id = handle_command(user_input, session_id)
            continue
        try:
            run_single_query(user_input, session_id)
        except Exception as e:
            logger.error(f"Workflow error: {e}", exc_info=True)
            print(f"[Error: {e}]")


def main():
    parser = argparse.ArgumentParser(description="edu-agent")
    parser.add_argument("--query",      "-q", type=str)
    parser.add_argument("--session",    "-s", type=str, default=None)
    parser.add_argument("--file",       "-f", type=str)
    parser.add_argument("--ingest-dir",       type=str)
    parser.add_argument("--stats",      action="store_true")
    args = parser.parse_args()

    session_id = args.session or str(uuid.uuid4())[:8]

    if args.stats:
        from src.utils.cache import get_response_cache
        from src.memory.memory_manager import get_memory_manager
        print(f"Cache: {get_response_cache().stats}")
        print(f"Sessions: {get_memory_manager().list_sessions()}")
        return

    if args.file:
        from src.rag.pipeline import run_pipeline
        run_pipeline(args.file).print_summary()

    if args.ingest_dir:
        from src.rag.pipeline import run_pipeline
        run_pipeline(args.ingest_dir).print_summary()

    if args.query:
        from src.memory.memory_manager import get_memory_manager
        get_memory_manager().create_session(session_id)
        run_single_query(args.query, session_id)
    else:
        interactive_loop(session_id)


if __name__ == "__main__":
    main()
