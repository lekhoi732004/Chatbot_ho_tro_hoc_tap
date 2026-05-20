"""
LangSmith/LangChain tracing helpers.

Tracing is controlled entirely by environment variables loaded from .env.
No API keys or endpoints are hard-coded in application code.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Callable, Dict, Optional

from dotenv import load_dotenv


TRUE_VALUES = {"1", "true", "yes", "on"}


def _is_enabled(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


@lru_cache(maxsize=1)
def configure_tracing() -> bool:
    """
    Load .env and normalize LangSmith variables for LangChain/LangGraph.

    Supported input env vars:
      LANGSMITH_TRACING, LANGSMITH_ENDPOINT, LANGSMITH_API_KEY, LANGSMITH_PROJECT

    Compatibility output env vars:
      LANGCHAIN_TRACING_V2, LANGCHAIN_ENDPOINT, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT
    """
    load_dotenv(override=False)

    enabled = _is_enabled(os.getenv("LANGSMITH_TRACING")) or _is_enabled(
        os.getenv("LANGCHAIN_TRACING_V2")
    )
    if not enabled:
        return False

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

    mappings = {
        "LANGSMITH_ENDPOINT": "LANGCHAIN_ENDPOINT",
        "LANGSMITH_API_KEY": "LANGCHAIN_API_KEY",
        "LANGSMITH_PROJECT": "LANGCHAIN_PROJECT",
    }
    for source, target in mappings.items():
        value = os.getenv(source)
        if value:
            os.environ.setdefault(target, value.strip().strip('"').strip("'"))

    return True


def build_trace_config(
    run_name: str,
    session_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build RunnableConfig for LangGraph invoke/ainvoke calls."""
    if not configure_tracing():
        return {}

    trace_metadata = {"session_id": session_id}
    if metadata:
        trace_metadata.update(metadata)

    return {
        "run_name": run_name,
        "tags": ["edu-agent", "langgraph", f"session:{session_id}"],
        "metadata": trace_metadata,
    }


def traceable(name: str) -> Callable:
    """
    Return LangSmith's @traceable decorator when available and enabled.
    Falls back to a no-op decorator when langsmith is unavailable.
    """
    configure_tracing()
    try:
        from langsmith import traceable as langsmith_traceable

        return langsmith_traceable(name=name)
    except Exception:
        def decorator(func: Callable) -> Callable:
            return func

        return decorator
