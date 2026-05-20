"""
Verifier agent.

Checks answer quality against the user query, retrieved context, and the visible
reasoning summaries produced by classifier and executor.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Optional

from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger("verifier")

VERIFY_SYSTEM = """You are Edu-Agent's strict answer verifier.

Evaluate the candidate answer for an educational chatbot.
Use the user query, the classifier thinking, the executor thinking, and the RAG context as the standard.

Score:
1. completeness: answers the actual user intent, not a related question
2. accuracy: agrees with RAG context and does not invent unsupported facts
3. clarity: structured, understandable, and in the appropriate language
4. logic: the visible thinking is coherent, follows the classifier plan, and leads to the final answer

Fail the answer when it:
- ignores important retrieved context
- contradicts the context
- fabricates citations, facts, formulas, or code behavior
- is too vague, too short, repetitive, or off-topic
- answers an out-of-scope request as if it were allowed
- contains thinking that is inconsistent with the final answer
- contains thinking that breaks the intended plan from the classifier

Respond only with valid JSON:
{
  "completeness": <float>,
  "accuracy": <float>,
  "clarity": <float>,
  "logic": <float>,
  "overall": <float>,
  "issues": [<string>, ...],
  "pass": <true|false>
}"""

VERIFY_USER = """Truy van: {query}

Classifier thinking: {classifier_thinking}

Executor thinking: {executor_thinking}

Cau tra loi: {answer}

Ngu canh RAG va nguon neu co: {context}

Hay danh gia chat luong cau tra loi va ca tinh hop ly cua thinking. Chi tra ve JSON."""


class VerifierAgent:
    """
    Verifies generated answers and visible reasoning summaries.
    Can request regeneration if quality is too low.
    """

    def __init__(self):
        cfg = get_config()
        self.enable_fact_check: bool = cfg.get("agents.verifier.enable_fact_check", True)
        self.enable_hallucination_check: bool = cfg.get(
            "agents.verifier.enable_hallucination_check", True
        )
        self.score_threshold: float = cfg.get("agents.verifier.score_threshold", 0.6)
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from src.utils.llm import load_llm

            self._llm = load_llm("verifier")
        return self._llm

    def verify(
        self,
        query: str,
        answer: str,
        context: Optional[str] = None,
        classifier_thinking: Optional[str] = None,
        executor_thinking: Optional[str] = None,
        use_llm: bool = True,
    ) -> Dict:
        """
        Verify an answer against the query, optional RAG context, and thinking traces.

        Returns:
            {
                "completeness": float,
                "accuracy": float,
                "clarity": float,
                "logic": float,
                "overall": float,
                "issues": list[str],
                "pass": bool,
                "method": str,
            }
        """
        logger.info(f"Dang kiem duyet cau tra loi (do dai={len(answer)})")

        heuristic_result = self._heuristic_check(
            query=query,
            answer=answer,
            classifier_thinking=classifier_thinking,
            executor_thinking=executor_thinking,
        )
        if not heuristic_result["pass"]:
            return {**heuristic_result, "method": "heuristic"}

        if use_llm:
            try:
                llm_result = self._llm_verify(
                    query=query,
                    answer=answer,
                    context=context,
                    classifier_thinking=classifier_thinking,
                    executor_thinking=executor_thinking,
                )
                llm_result["method"] = "llm"
                return llm_result
            except Exception as e:
                logger.warning(f"Kiem duyet bang LLM that bai: {e}, dung ket qua heuristic")

        return {**heuristic_result, "method": "heuristic"}

    def _heuristic_check(
        self,
        query: str,
        answer: str,
        classifier_thinking: Optional[str],
        executor_thinking: Optional[str],
    ) -> Dict:
        """Quick rule-based checks before the deeper LLM verification."""
        issues = []
        logic_score = 1.0

        if not answer or len(answer.strip()) < 10:
            issues.append("Cau tra loi qua ngan hoac rong")
            return self._build_result(0.0, 0.0, 0.0, 0.0, issues)

        failure_phrases = [
            "i cannot",
            "i don't know",
            "as an ai",
            "i'm not able",
            "[execution failed",
            "i apologize, but i cannot",
        ]
        answer_lower = answer.lower()
        for phrase in failure_phrases:
            if phrase in answer_lower:
                issues.append(f"Cau tra loi chua cum tu choi: '{phrase}'")

        sentences = re.split(r"[.!?]+", answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        if len(sentences) > 3:
            unique = len(set(sentences))
            if unique / len(sentences) < 0.5:
                issues.append("Cau tra loi bi lap dang ke")

        if len(answer.split()) < 15 and "?" in query:
            issues.append("Cau tra loi co the qua ngan so voi cau hoi")

        if classifier_thinking and len(classifier_thinking.strip()) < 12:
            issues.append("Classifier thinking qua ngan de kiem tra logic")
            logic_score -= 0.2

        if not executor_thinking or len(executor_thinking.strip()) < 12:
            issues.append("Executor thinking thieu hoac qua ngan")
            logic_score -= 0.35

        overall = max(0.3, 1.0 - len(issues) * 0.2)
        logic_score = max(0.0, min(logic_score, overall))

        return self._build_result(
            completeness=overall,
            accuracy=overall,
            clarity=overall,
            logic=logic_score,
            issues=issues,
        )

    def _llm_verify(
        self,
        query: str,
        answer: str,
        context: Optional[str],
        classifier_thinking: Optional[str],
        executor_thinking: Optional[str],
    ) -> Dict:
        """Use LLM for deeper verification."""
        llm = self._get_llm()
        prompt = VERIFY_USER.format(
            query=query,
            classifier_thinking=(classifier_thinking or "Khong co"),
            executor_thinking=(executor_thinking or "Khong co"),
            answer=answer[:1000],
            context=(context[:500] if context else "Khong co ngu canh"),
        )
        raw = llm.generate(
            prompt=prompt,
            system_prompt=VERIFY_SYSTEM,
            max_new_tokens=320,
            temperature=0.1,
        )

        json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return {
                "completeness": float(data.get("completeness", 0.7)),
                "accuracy": float(data.get("accuracy", 0.7)),
                "clarity": float(data.get("clarity", 0.7)),
                "logic": float(data.get("logic", 0.7)),
                "overall": float(data.get("overall", 0.7)),
                "issues": data.get("issues", []),
                "pass": bool(data.get("pass", True)),
            }

        return self._build_result(0.7, 0.7, 0.7, 0.7, [])

    def _build_result(
        self,
        completeness: float,
        accuracy: float,
        clarity: float,
        logic: float,
        issues: list,
    ) -> Dict:
        overall = (completeness + accuracy + clarity + logic) / 4
        return {
            "completeness": round(completeness, 3),
            "accuracy": round(accuracy, 3),
            "clarity": round(clarity, 3),
            "logic": round(logic, 3),
            "overall": round(overall, 3),
            "issues": issues,
            "pass": overall >= self.score_threshold,
        }
