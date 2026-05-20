"""
Optimised multi-agent workflow with adaptive RAG from classifier output.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from src.utils.config_loader import get_config
from src.utils.logger import get_logger
from src.utils.tracing import build_trace_config, configure_tracing, traceable

logger = get_logger("workflow")
configure_tracing()

class AgentState(TypedDict, total=False):
    query: str
    session_id: str
    category: str
    classification: Dict
    chunks: List[Dict]
    context: str
    prompt_package: Dict
    execution_result: Dict
    verification: Dict
    answer: str
    metadata: Dict
    retry_count: int
    force_executor: bool
    blocked_by_scope: bool
    needs_more_context: bool
    retrieval_best_score: float


# ── SINGLETONS ─────────────────────────────
_classifier = _optimizer = _executor = _verifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        from src.agents.classifier import ClassifierAgent
        _classifier = ClassifierAgent()
    return _classifier


def _get_optimizer():
    global _optimizer
    if _optimizer is None:
        from src.agents.optimizer import OptimizerAgent
        _optimizer = OptimizerAgent()
    return _optimizer


def _get_executor():
    global _executor
    if _executor is None:
        from src.agents.executor import ExecutorAgent
        _executor = ExecutorAgent()
    return _executor


def _get_verifier():
    global _verifier
    if _verifier is None:
        from src.agents.verifier import VerifierAgent
        _verifier = VerifierAgent()
    return _verifier


# ── RETRIEVE ───────────────────────────────
def _do_retrieve(query: str, session_id: str = "default") -> Dict:
    try:
        from src.rag.retriever import retrieve, format_context

        chunks = retrieve(query, session_id=session_id)
        context = format_context(chunks)
        best_score = max((chunk.get("hybrid_score", chunk.get("score", 0.0)) for chunk in chunks), default=0.0)

        return {
            "chunks": chunks,
            "context": context,
            "context_sufficient": bool(chunks),
            "retrieval_best_score": float(best_score),
        }

    except Exception as e:
        logger.warning(f"[RETRIEVE] failed: {e}")
        return {
            "chunks": [],
            "context": "",
            "context_sufficient": False,
            "retrieval_best_score": 0.0,
        }


# ── NODE: CLASSIFY + ADAPTIVE RAG ─────────
@traceable("classify_and_retrieve")
def node_classify_and_retrieve(state: AgentState) -> AgentState:
    query = state["query"]

    classification = _get_classifier().classify(query)
    category = classification["category"]

    if not classification.get("is_in_scope", True):
        logger.info("[CLASSIFIER] blocked out-of-scope query")
        return {
            **state,
            "category": category,
            "classification": classification,
            "chunks": [],
            "context": "",
            "blocked_by_scope": True,
        }

    requires_rag = classification.get("requires_rag", True)

    # ── Adaptive decision ─────────────────
    if not requires_rag:
        logger.info("[ADAPTIVE] skip RAG (category)")
        retrieve_result = {
            "chunks": [],
            "context": "",
            "context_sufficient": True,
            "retrieval_best_score": None,
        }

    else:
        retrieve_result = _do_retrieve(query, session_id=state.get("session_id", "default"))
        logger.info(f"[ADAPTIVE] use RAG ({len(retrieve_result['chunks'])} chunks)")

    return {
        **state,
        "category": category,
        "classification": classification,
        "blocked_by_scope": False,
        "needs_more_context": requires_rag and not retrieve_result.get("context_sufficient", False),
        **retrieve_result,
    }


def route_after_classify(state: AgentState) -> str:
    if state.get("blocked_by_scope"):
        return "finalize"
    if state.get("needs_more_context"):
        return "finalize"
    return "optimize"


# ── OPTIMIZE ─────────────────────────────
@traceable("optimize_prompt")
def node_optimize(state: AgentState) -> AgentState:
    from src.memory.memory_manager import get_memory_manager

    mm = get_memory_manager()
    history = mm.get_history(state.get("session_id", "default"), last_n=8)

    pkg = _get_optimizer().optimize(
        query=state["query"],
        category=state.get("category", "general"),
        context=state.get("context", ""),
        history=history,
        classification=state.get("classification", {}),
    )

    return {**state, "prompt_package": pkg}


# ── EXECUTE ─────────────────────────────
@traceable("execute_generation")
def node_execute(state: AgentState) -> AgentState:
    result = _get_executor().execute(
        state["prompt_package"],
        force_executor=state.get("force_executor", False),
    )

    if not isinstance(result, dict):
        result = {"answer": str(result)}

    result.setdefault("thinking", "")
    result.setdefault("answer", "")
    result.setdefault("model_used", "unknown")
    result.setdefault("generation_time", 0)

    return {**state, "execution_result": result}


# ── VERIFY ─────────────────────────────
@traceable("verify_answer")
def node_verify(state: AgentState) -> AgentState:
    execution_result = state.get("execution_result") or {}
    answer = execution_result.get("answer", "")

    if not answer.strip():
        return {
            **state,
            "verification": {"pass": False, "overall": 0.0},
        }

    v = _get_verifier().verify(
        query=state["query"],
        answer=answer,
        context=state.get("context", ""),
        classifier_thinking=(state.get("classification") or {}).get("thinking", ""),
        executor_thinking=execution_result.get("thinking", ""),
    )

    return {**state, "verification": v}


# ── RETRY ─────────────────────────────
def should_retry(state: AgentState) -> str:
    cfg = get_config()
    max_r = cfg.get("agents.executor.max_retries", 2)
    score_threshold = cfg.get("agents.verifier.score_threshold", 0.7)

    v = state.get("verification") or {}
    if v.get("pass", False) and v.get("overall", 0) >= score_threshold:
        return "finalize"

    if state.get("retry_count", 0) < max_r:
        return "retry"

    return "finalize"


@traceable("retry_setup")
def node_retry_setup(state: AgentState) -> AgentState:
    prompt_package = dict(state.get("prompt_package", {}))
    verification = state.get("verification", {})
    issues = verification.get("issues") or []
    issue_text = "\n".join(f"- {issue}" for issue in issues) if issues else "- Bộ kiểm duyệt chưa chấp nhận câu trả lời."

    retry_instruction = (
        "\n\nREVISION_REQUIRED:\n"
        "Bộ kiểm duyệt chưa chấp nhận câu trả lời trước đó. Hãy sinh lại bằng model executor. "
        "Sửa toàn bộ vấn đề bên dưới, đồng thời giữ nguyên ý định từ classifier, căn cứ RAG, "
        "ngữ cảnh memory và ngôn ngữ của người dùng. Không bịa dữ kiện còn thiếu; "
        "hãy nêu rõ phần ngữ cảnh còn thiếu khi cần.\n"
        f"{issue_text}\n"
    )
    prompt_package["user_prompt"] = prompt_package.get("user_prompt", "") + retry_instruction

    return {
        **state,
        "prompt_package": prompt_package,
        "retry_count": state.get("retry_count", 0) + 1,
        "force_executor": True,
    }


# ── FINALIZE ───────────────────────────
@traceable("finalize_answer")
def node_finalize(state: AgentState) -> AgentState:
    from src.memory.memory_manager import get_memory_manager

    exec_res = state.get("execution_result") or {}
    answer = exec_res.get("answer", "")

    sid = state.get("session_id", "default")
    category = state.get("category", "general")
    classification = state.get("classification", {})

    if state.get("blocked_by_scope"):
        cfg = get_config()
        answer = cfg.get("agents.classifier.out_of_scope_response", "")
        exec_res = {
            "model_used": "cong-phan-loai",
            "generation_time": 0,
        }
    elif state.get("needs_more_context"):
        cfg = get_config()
        answer = cfg.get(
            "rag.no_context_response",
            "Mình chưa tìm thấy dữ liệu đủ liên quan trong tài liệu hiện có. "
            "Bạn hãy cung cấp thêm thông tin, tải lên file liên quan hoặc hỏi cụ thể hơn để mình truy xuất chính xác.",
        )
        exec_res = {
            "model_used": "bo-truy-xuat",
            "generation_time": 0,
        }

    mm = get_memory_manager()
    mm.add_message(sid, "user", state["query"], category=category)
    mm.add_message(sid, "assistant", answer, category=category)

    metadata = {
        "category": category,
        "confidence": classification.get("confidence"),
        "method": classification.get("method"),
        "intent": classification.get("intent"),
        "topic": classification.get("topic"),
        "is_in_scope": classification.get("is_in_scope"),
        "scope_reason": classification.get("scope_reason"),
        "subtasks": classification.get("subtasks"),
        "model_used": exec_res.get("model_used"),
        "generation_time": exec_res.get("generation_time"),
        "verification_logic_score": (state.get("verification") or {}).get("logic"),
        "verification_score": (state.get("verification") or {}).get("overall"),
        "chunks_used": len(state.get("chunks", [])),
        "retrieval_best_score": state.get("retrieval_best_score"),
        "needs_more_context": state.get("needs_more_context", False),
        "retry_count": state.get("retry_count", 0),
    }

    return {**state, "answer": answer.strip(), "metadata": metadata}


# ── GRAPH ─────────────────────────────
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("classify_and_retrieve", node_classify_and_retrieve)
    g.add_node("optimize", node_optimize)
    g.add_node("execute", node_execute)
    g.add_node("verify", node_verify)
    g.add_node("retry_setup", node_retry_setup)
    g.add_node("finalize", node_finalize)

    g.add_edge(START, "classify_and_retrieve")
    g.add_conditional_edges(
        "classify_and_retrieve",
        route_after_classify,
        {"optimize": "optimize", "finalize": "finalize"},
    )
    g.add_edge("optimize", "execute")
    g.add_edge("execute", "verify")

    g.add_conditional_edges(
        "verify",
        should_retry,
        {"finalize": "finalize", "retry": "retry_setup"},
    )

    g.add_edge("retry_setup", "execute")
    g.add_edge("finalize", END)

    return g.compile()


_app = None


def get_workflow():
    global _app
    if _app is None:
        _app = build_graph()
    return _app


def run_workflow(query: str, session_id: str = "default") -> Dict:
    initial_state = {
        "query": query,
        "session_id": session_id,
        "retry_count": 0,
        "force_executor": False,
    }
    trace_config = build_trace_config(
        run_name="edu_agent_workflow",
        session_id=session_id,
        metadata={"query_preview": query[:200]},
    )
    return get_workflow().invoke(initial_state, config=trace_config or None)


async def run_workflow_async(query: str, session_id: str = "default") -> Dict:
    return await asyncio.to_thread(run_workflow, query, session_id)
