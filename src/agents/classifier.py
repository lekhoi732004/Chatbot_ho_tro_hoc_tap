"""
Classifier agent.

Context-only LLM classifier.

Design goals:
- classification is determined only by LLM reasoning over the full query context,
- no heuristic classifier,
- no keyword lists,
- no token-based disambiguation,
- stable output contract for downstream pipeline,
- if context is insufficient, the LLM must request clarification instead of guessing,
- the same LLM pass must decide:
  1) whether the question is in scope (study-only assistant),
  2) the logical plan/subtasks,
  3) which tools should be called.

The public `classify()` method keeps the legacy-compatible output keys.
Additional tool-planning and clarification details are stored internally and can be
retrieved via helper methods without breaking downstream callers.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger("classifier")

# Legacy-compatible task categories.
CATEGORIES = [
    "factual_qa",
    "explanation",
    "problem_solving",
    "summarization",
    "translation",
    "code_generation",
    "math",
    "writing",
    "general",
]

# Standardized academic domains.
DOMAINS = [
    "mathematics",
    "statistics",
    "probability",
    "programming",
    "computer_science",
    "artificial_intelligence",
    "machine_learning",
    "deep_learning",
    "data_science",
    "data_analysis",
    "databases",
    "networking",
    "cybersecurity",
    "software_engineering",
    "operating_systems",
    "algorithms",
    "physics",
    "chemistry",
    "biology",
    "economics",
    "finance",
    "business_analytics",
    "academic_writing",
    "research_methods",
    "education",
    "general",
]

# The classifier itself stays tool-agnostic; available tool names can be configured.
DEFAULT_AVAILABLE_TOOLS = [
    "none",
    "retriever",
    "web_search",
    "calculator",
    "python",
]

SYSTEM_PROMPT = """You are Edu-Agent's classifier and planner.

Your job is routing only. Do not answer the user's question.

You must perform exactly three reasoning tasks at once:
1. Determine whether the user's request is in scope.
   - The assistant only serves learning, study, academic, technical, scientific,
     coding, mathematical, research, and education-related requests.
   - If the request is not about learning/study/academic support, mark it out of scope.
2. Determine the logical plan.
   - Infer the task category.
   - Infer the dominant academic topic.
   - Infer a short subtopic.
   - Produce 1-5 concise logical subtasks for downstream execution.
3. Determine which tools are needed.
   - Use only the available tools list provided below.
   - Choose tools based on context and task needs, not based on single words.
   - If no tool is needed, return ["none"].

Important policy:
- You must reason from full context only.
- Do not use token matching, keyword spotting, surface-form rules, or acronym heuristics.
- Never infer the topic from one isolated word when the broader context is unclear.
- If the context is insufficient to confidently determine the educational topic, scope,
  or required tools, you must ask a clarification question instead of guessing.
- The "thinking" field must be concise and visible-safe: summarize routing logic briefly.
  Do not dump hidden chain-of-thought. Keep it to 1-3 short sentences.

Task categories (category must be exactly one of these):
{categories}

Allowed academic domains (topic must be exactly one of these):
{domains}

Available tools (suggested_tools must be a subset of these):
{tools}

Decision rules:
- category = task type, not academic field.
- topic = single dominant academic field.
- subtopic = short normalized label, e.g. calculus, sql, nlp, regression, thermodynamics.
- If the user request mixes several fields, choose the dominant one.
- If the topic is unclear even after contextual reasoning, use topic="general" and subtopic="general".
- If clarification is needed, set needs_clarification=true and provide one short, direct clarifying_question.
- requires_rag=true only when grounded documents, course materials, citations, uploaded files,
  or external factual retrieval are genuinely needed.
- For self-contained translation, direct code drafting, or self-contained math reasoning,
  requires_rag is usually false unless the context explicitly requires grounding.

Return only valid JSON with no markdown and no extra text.
Use exactly this schema:
{{
  "thinking": "<concise routing rationale based on scope, topic, plan, and tool need>",
  "category": "<one category>",
  "confidence": <float in [0,1]>,
  "intent": "<short routing intent>",
  "subtasks": ["<step1>", "<step2>"],
  "topic": "<one allowed domain>",
  "subtopic": "<normalized subtopic>",
  "is_in_scope": <true|false>,
  "scope_confidence": <float in [0,1]>,
  "scope_reason": "<short reason>",
  "requires_rag": <true|false>,
  "suggested_tools": ["<tool_name>", "..."],
  "tool_reason": "<short reason for tool choice>",
  "needs_clarification": <true|false>,
  "clarifying_question": "<empty string or one short question>"
}}
"""


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return default


def _extract_json(raw: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from raw LLM output."""
    if not raw:
        return {}

    text = raw.strip()

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3].strip()
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    candidate = text[start : end + 1]
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        logger.warning("Khong trich xuat duoc JSON hop le tu dau ra LLM classifier")
        return {}


def _normalize_text_label(value: Any, default: str = "general") -> str:
    text = str(value or "").strip().lower().replace(" ", "_")
    return text if text else default


def _normalize_tool_list(value: Any, allowed_tools: List[str]) -> List[str]:
    if isinstance(value, list):
        raw_items = value
    elif value is None:
        raw_items = []
    else:
        raw_items = [value]

    normalized: List[str] = []
    for item in raw_items:
        tool_name = _normalize_text_label(item, default="")
        if tool_name and tool_name in allowed_tools and tool_name not in normalized:
            normalized.append(tool_name)

    return normalized or ["none"]


def _default_fallback_result() -> Dict[str, Any]:
    return {
        "thinking": "Ngu canh hien tai chua du de dinh tuyen chac chan nen can hoi lai nguoi dung truoc.",
        "category": "general",
        "confidence": 0.0,
        "intent": "Can lam ro yeu cau truoc khi dinh tuyen",
        "subtasks": [
            "Yeu cau nguoi dung lam ro muc tieu hoc tap hoac bai toan",
            "Sau do moi xac dinh huong xu ly phu hop",
        ],
        "topic": "general",
        "subtopic": "general",
        "is_in_scope": True,
        "scope_confidence": 0.0,
        "scope_reason": "Chua du thong tin de ket luan chac chan pham vi",
        "requires_rag": False,
        "suggested_tools": ["none"],
        "tool_reason": "Chua du ngu canh de quyet dinh cong cu",
        "needs_clarification": True,
        "clarifying_question": "Ban muon hoi ve chu de hoc tap cu the nao?",
        "reason": "llm_error",
        "method": "llm_fallback",
    }


def _build_prompt(query: str) -> str:
    return (
        "Phan loai va lap ke hoach cho truy van sau, chi dua tren ngu canh day du cua no:\n"
        f"{query}"
    )


def _llm_classify(
    query: str,
    llm: Any,
    allowed_topics: List[str],
    available_tools: List[str],
) -> Dict[str, Any]:
    system_prompt = SYSTEM_PROMPT.format(
        categories=", ".join(CATEGORIES),
        domains=", ".join(allowed_topics),
        tools=", ".join(available_tools),
    )
    raw = llm.generate(
        prompt=_build_prompt(query),
        system_prompt=system_prompt,
        max_new_tokens=420,
        temperature=0.1,
    )
    result = _extract_json(raw)
    if result:
        result.setdefault("reason", "llm_routing")
        result.setdefault("method", "llm")
    return result


class ClassifierAgent:
    def __init__(self):
        cfg = get_config()
        self.confidence_threshold: float = cfg.get(
            "agents.classifier.confidence_threshold", 0.7
        )
        configured_topics: List[str] = cfg.get(
            "agents.classifier.allowed_topics",
            DOMAINS,
        )
        self.allowed_topics: List[str] = [
            topic for topic in configured_topics if topic in DOMAINS
        ] or DOMAINS.copy()

        configured_tools: List[str] = cfg.get(
            "agents.classifier.available_tools",
            DEFAULT_AVAILABLE_TOOLS,
        )
        self.available_tools: List[str] = [
            _normalize_text_label(tool, default="")
            for tool in configured_tools
            if _normalize_text_label(tool, default="")
        ] or DEFAULT_AVAILABLE_TOOLS.copy()
        if "none" not in self.available_tools:
            self.available_tools.insert(0, "none")

        self._llm = None
        self._last_tool_plan: List[str] = ["none"]
        self._last_tool_reason: str = ""
        self._last_needs_clarification: bool = False
        self._last_clarifying_question: str = ""
        self._last_raw_result: Dict[str, Any] = {}

    def _get_llm(self):
        if self._llm is None:
            from src.utils.llm import load_llm

            self._llm = load_llm("classifier")
        return self._llm

    def classify(self, query: str) -> Dict[str, Any]:
        """LLM-only classification with legacy-compatible output."""
        try:
            raw_result = _llm_classify(
                query=query,
                llm=self._get_llm(),
                allowed_topics=self.allowed_topics,
                available_tools=self.available_tools,
            )
            if not raw_result:
                raise ValueError("LLM classifier tra ve JSON rong hoac khong hop le")
        except Exception as exc:
            logger.warning(f"Phan loai bang LLM that bai: {exc}")
            raw_result = _default_fallback_result()

        self._last_raw_result = dict(raw_result)
        normalized = self._normalize(raw_result)
        return normalized

    def get_last_tool_plan(self) -> List[str]:
        return list(self._last_tool_plan)

    def get_last_tool_reason(self) -> str:
        return self._last_tool_reason

    def needs_clarification(self) -> bool:
        return self._last_needs_clarification

    def get_last_clarifying_question(self) -> str:
        return self._last_clarifying_question

    def get_last_raw_result(self) -> Dict[str, Any]:
        return dict(self._last_raw_result)

    def _normalize(self, result: Dict[str, Any]) -> Dict[str, Any]:
        category = str(result.get("category", "general")).strip()
        if category not in CATEGORIES:
            category = "general"

        confidence = max(0.0, min(_to_float(result.get("confidence"), 0.0), 1.0))

        topic = _normalize_text_label(result.get("topic"), default="general")
        if topic not in DOMAINS:
            topic = "general"

        subtopic = _normalize_text_label(result.get("subtopic"), default="general")

        is_in_scope = _to_bool(result.get("is_in_scope"), True)
        scope_confidence = max(
            0.0,
            min(_to_float(result.get("scope_confidence"), confidence), 1.0),
        )
        scope_reason = str(
            result.get("scope_reason") or "LLM chua giai thich ro quyet dinh pham vi"
        ).strip()

        requires_rag = _to_bool(result.get("requires_rag"), False)

        subtasks = result.get("subtasks")
        if not isinstance(subtasks, list):
            subtasks = []
        subtasks = [str(step).strip() for step in subtasks if str(step).strip()][:5]
        if not subtasks:
            subtasks = ["Lam ro yeu cau truoc khi xu ly"]

        intent = str(result.get("intent") or "Dinh tuyen yeu cau").strip()
        reason = str(result.get("reason") or "llm_routing").strip()
        method = str(result.get("method") or "llm").strip()
        thinking = str(result.get("thinking") or "").strip()

        suggested_tools = _normalize_tool_list(
            result.get("suggested_tools"),
            allowed_tools=self.available_tools,
        )
        tool_reason = str(result.get("tool_reason") or "").strip()

        needs_clarification = _to_bool(result.get("needs_clarification"), False)
        clarifying_question = str(result.get("clarifying_question") or "").strip()

        if needs_clarification:
            confidence = min(confidence, 0.49)
            topic = "general"
            subtopic = "general" if subtopic == "general" else subtopic
            requires_rag = False
            suggested_tools = ["none"]
            if not thinking:
                thinking = (
                    "Ngu canh chua du ro de suy ra dung pham vi va ke hoach xu ly, "
                    "nen can hoi lam ro truoc."
                )
            if not intent:
                intent = "Hoi lai de lam ro yeu cau truoc khi dinh tuyen"
            if not clarifying_question:
                clarifying_question = "Ban co the noi ro hon yeu cau hoc tap cua ban khong?"
            if not subtasks:
                subtasks = ["Hoi lai de lam ro ngu canh"]
            if not scope_reason:
                scope_reason = "Chua du ngu canh de xac dinh pham vi"

        if not thinking:
            tool_summary = ", ".join(suggested_tools)
            step_summary = "; ".join(subtasks[:3])
            thinking = (
                f"Yeu cau duoc dinh tuyen vao nhom {category} thuoc chu de {topic}/{subtopic}. "
                f"Ke hoach chinh: {step_summary}. "
                f"Cong cu du kien: {tool_summary}."
            )

        self._last_tool_plan = suggested_tools
        self._last_tool_reason = tool_reason
        self._last_needs_clarification = needs_clarification
        self._last_clarifying_question = clarifying_question

        return {
            "thinking": thinking,
            "category": category,
            "confidence": confidence,
            "reason": reason,
            "method": method,
            "intent": intent,
            "subtasks": subtasks,
            "topic": topic,
            "subtopic": subtopic,
            "is_in_scope": is_in_scope,
            "scope_confidence": scope_confidence,
            "scope_reason": scope_reason,
            "requires_rag": requires_rag,
            "is_high_confidence": confidence >= self.confidence_threshold + 0.1 and is_in_scope,
        }
