"""
Executor agent.

Runs the executor model with the optimizer-produced prompt package.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger("executor")

HEAVY_CATEGORIES = {"code_generation", "math", "problem_solving", "writing", "explanation"}

STRUCTURED_OUTPUT_INSTRUCTION = """
Return only valid JSON with this exact schema:
{
  "thinking": "<concise visible reasoning summary, 2-6 short lines or sentences>",
  "answer": "<final answer for the user>"
}

Rules:
- "thinking" must summarize the logic you used. Do not dump hidden chain-of-thought.
- "answer" must be complete, direct, and in the user's language.
- No markdown fences. No extra keys. No text outside JSON.
""".strip()


def _extract_json(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}

    text = raw.strip()

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3].strip()
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    candidate = text[start : end + 1]
    try:
        data = json.loads(candidate)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _build_fallback_thinking(prompt_package: Dict[str, Any]) -> str:
    prior_thinking = str(prompt_package.get("thinking") or "").strip()
    intent = str(prompt_package.get("intent") or "").strip()
    topic = str(prompt_package.get("topic") or "").strip()
    subtasks = prompt_package.get("subtasks") or []
    subtasks = [str(step).strip() for step in subtasks if str(step).strip()][:4]

    parts: List[str] = []
    if prior_thinking:
        parts.append(f"Classifier plan: {prior_thinking}")
    if intent:
        parts.append(f"Muc tieu: {intent}.")
    if topic:
        parts.append(f"Chu de xu ly: {topic}.")
    if subtasks:
        parts.append("Huong xu ly: " + " -> ".join(subtasks) + ".")
    return " ".join(parts).strip()


def _normalize_generation(raw: str, prompt_package: Dict[str, Any]) -> Dict[str, str]:
    data = _extract_json(raw)
    if data:
        answer = str(data.get("answer") or "").strip()
        thinking = str(data.get("thinking") or "").strip()
        if answer:
            return {
                "thinking": thinking or _build_fallback_thinking(prompt_package),
                "answer": answer,
            }

    return {
        "thinking": _build_fallback_thinking(prompt_package),
        "answer": (raw or "").strip(),
    }


class ExecutorAgent:
    """Generate answers using the executor model."""

    def __init__(self):
        cfg = get_config()
        self.timeout: int = cfg.get("agents.executor.timeout", 60)
        self.max_retries: int = cfg.get("agents.executor.max_retries", 2)
        self._executor_llm = None
        self._optimizer_llm = None

    def _get_llm(self, category: str, force_executor: bool = False):
        """Always use the executor model for generation."""
        if self._executor_llm is None:
            from src.utils.llm import load_llm

            self._executor_llm = load_llm("executor")
        return self._executor_llm

    def execute(self, prompt_package: Dict, force_executor: bool = False) -> Dict:
        """
        Execute generation for the given prompt package.

        `force_executor` is kept for workflow compatibility; execution still uses executor.
        """
        category = prompt_package.get("category", "general")
        base_system_prompt = prompt_package.get("system_prompt", "")
        user_prompt = prompt_package.get("user_prompt", "")
        few_shot = prompt_package.get("few_shot", [])
        history = prompt_package.get("history", [])
        classifier_thinking = str(prompt_package.get("thinking") or "").strip()
        classifier_context = (
            f"Classifier visible thinking:\n{classifier_thinking}\n\n"
            if classifier_thinking
            else ""
        )
        system_prompt = (
            f"{base_system_prompt}\n\n{classifier_context}{STRUCTURED_OUTPUT_INSTRUCTION}"
        ).strip()

        messages: List[Dict] = []
        messages.extend(few_shot)
        messages.extend(history)

        llm = self._get_llm(category, force_executor)
        model_name = Path(llm.model_path).name

        attempt = 0
        last_error = None
        start = time.time()

        while attempt <= self.max_retries:
            try:
                logger.info(
                    f"Dang sinh cau tra loi [{model_name}] lan {attempt + 1}/{self.max_retries + 1}"
                )
                raw = llm.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    history=messages if messages else None,
                )
                generated = _normalize_generation(raw, prompt_package)
                elapsed = time.time() - start
                logger.info(f"Sinh cau tra loi xong sau {elapsed:.2f}s")

                return {
                    "thinking": generated["thinking"],
                    "answer": generated["answer"],
                    "model_used": model_name,
                    "category": category,
                    "generation_time": round(elapsed, 3),
                    "retries": attempt,
                    "success": True,
                }
            except Exception as e:
                last_error = e
                logger.warning(f"Lan sinh {attempt + 1} that bai: {e}")
                attempt += 1
                time.sleep(1.0 * attempt)

        logger.warning("Model executor that bai, chuyen sang model optimizer du phong")
        try:
            if self._optimizer_llm is None:
                from src.utils.llm import load_llm

                self._optimizer_llm = load_llm("optimizer")
            raw = self._optimizer_llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                history=messages if messages else None,
            )
            generated = _normalize_generation(raw, prompt_package)
            return {
                "thinking": generated["thinking"],
                "answer": generated["answer"],
                "model_used": "optimizer-du-phong",
                "category": category,
                "generation_time": round(time.time() - start, 3),
                "retries": attempt,
                "success": True,
            }
        except Exception as e:
            last_error = e

        return {
            "thinking": _build_fallback_thinking(prompt_package),
            "answer": f"[Sinh cau tra loi that bai sau {attempt} lan: {last_error}]",
            "model_used": model_name,
            "category": category,
            "generation_time": round(time.time() - start, 3),
            "retries": attempt,
            "success": False,
        }
