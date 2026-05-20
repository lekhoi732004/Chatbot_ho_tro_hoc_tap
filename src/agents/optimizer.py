"""
Agent optimizer.
Nhận truy vấn, phân loại tác vụ, ngữ cảnh RAG và memory để tạo prompt tối ưu
cho executor.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger("optimizer")

# System prompt theo từng category. Giữ tiếng Anh theo yêu cầu.
SYSTEM_PROMPTS: Dict[str, str] = {
    "factual_qa": (
        "You are Edu-Agent, a bilingual Vietnamese-English educational assistant. "
        "Answer factual questions using RAG_CONTEXT as the primary source. "
        "If context is missing or weak, state the limitation before using general knowledge. "
        "Be concise, cite source names from context when useful, and do not invent facts."
    ),
    "explanation": (
        "You are Edu-Agent, a patient educator. Explain concepts clearly in the user's language. "
        "Use the retrieved context and conversation memory when available. "
        "Use short sections, concrete examples, and only the necessary visible steps."
    ),
    "problem_solving": (
        "You are Edu-Agent, a systematic problem-solving tutor. "
        "Use the classifier plan to solve the task. Present concise solution steps and final answer. "
        "Check assumptions and mention missing information instead of guessing."
    ),
    "summarization": (
        "You are Edu-Agent, an academic summarizer. Summarize only the provided text/context. "
        "Keep key definitions, formulas, procedures, and caveats. "
        "Do not add unsupported outside content."
    ),
    "translation": (
        "You are Edu-Agent, a Vietnamese-English academic translator. "
        "Preserve technical meaning, notation, tone, and formatting. "
        "If a term has multiple translations, choose the academic one and briefly note ambiguity."
    ),
    "code_generation": (
        "You are Edu-Agent, a software engineering tutor. "
        "Write correct, minimal, readable code that matches the user's constraints. "
        "Explain usage and important decisions briefly. Do not include unrelated refactors."
    ),
    "math": (
        "You are Edu-Agent, a mathematics tutor. "
        "Define variables, show the essential derivation, and present the final result clearly. "
        "Avoid unsupported shortcuts and check units or conditions when relevant."
    ),
    "writing": (
        "You are Edu-Agent, an academic writing coach. "
        "Produce well-structured writing in the requested style and language. "
        "Keep the content aligned with the user's purpose and available context."
    ),
    "general": (
        "You are Edu-Agent, a helpful educational assistant. "
        "Answer only in the supported educational/professional learning scope. "
        "Use context and memory when available. Be accurate, concise, and transparent about uncertainty."
    ),
}

LANGUAGE_RULE = (
    "Language rule: If the user writes Vietnamese, including Vietnamese without diacritics, "
    "answer in natural Vietnamese with full diacritics. Do not answer in unaccented Vietnamese. "
    "Keep English only when the user explicitly asks for English, translation output, code symbols, "
    "or established technical terms."
)

# Ví dụ few-shot theo category.
FEW_SHOT_EXAMPLES: Dict[str, List[Dict]] = {
    "math": [
        {
            "user": "Tính tích phân ∫(2x + 3)dx",
            "assistant": (
                "**Bước 1:** Tách tích phân\n∫(2x + 3)dx = ∫2x dx + ∫3 dx\n\n"
                "**Bước 2:** Tính từng phần\n∫2x dx = x^2 + C1\n∫3 dx = 3x + C2\n\n"
                "**Kết quả:** ∫(2x + 3)dx = x^2 + 3x + C"
            ),
        }
    ],
    "code_generation": [
        {
            "user": "Viết hàm Python đảo ngược chuỗi",
            "assistant": (
                "```python\ndef reverse_string(s: str) -> str:\n"
                '    """Đảo ngược chuỗi và trả về kết quả."""\n'
                "    return s[::-1]\n\n"
                "# Ví dụ sử dụng\nprint(reverse_string('hello'))  # Kết quả: 'olleh'\n```"
            ),
        }
    ],
    "summarization": [
        {
            "user": "Tóm tắt: Vòng tuần hoàn nước gồm bay hơi, ngưng tụ, mưa rơi và thu gom.",
            "assistant": (
                "**Tóm tắt:** Vòng tuần hoàn nước là quá trình liên tục với bốn giai đoạn: "
                "(1) Bay hơi - nước biến thành hơi do nhiệt; "
                "(2) Ngưng tụ - hơi nước lạnh đi và tạo thành mây; "
                "(3) Mưa rơi - nước rơi xuống dưới dạng mưa hoặc tuyết; "
                "(4) Thu gom - nước tập trung ở sông, hồ, biển để bắt đầu chu trình mới."
            ),
        }
    ],
}

# Mẫu prompt CoT.
COT_TEMPLATES: Dict[str, str] = {
    "problem_solving": (
        "Giải quyết tác vụ bằng các bước ngắn gọn, dễ quan sát.\n\n"
        "Câu hỏi: {query}\n\nTrả lời:"
    ),
    "math": (
        "Giải bài toán này. Chỉ trình bày suy luận cần thiết và kết quả cuối cùng.\n"
        "**Bài toán:** {query}\n\n"
        "**Lời giải:**"
    ),
    "explanation": (
        "Giải thích rõ ràng cho người học:\n"
        "**Chủ đề:** {query}\n\n"
        "**Giải thích:**"
    ),
    "default": "{query}",
}


class OptimizerAgent:
    """
    Tạo gói prompt tối ưu cho executor.
    Bổ sung system prompt, ví dụ few-shot và khung CoT khi cần.
    """

    def __init__(self):
        from src.utils.config_loader import get_config

        cfg = get_config()
        self.max_iterations = cfg.get("agents.optimizer.max_iterations", 3)
        self.enable_cot = cfg.get("agents.optimizer.enable_cot", True)
        self.enable_few_shot = cfg.get("agents.optimizer.enable_few_shot", True)

    def optimize(
        self,
        query: str,
        category: str,
        context: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        classification: Optional[Dict] = None,
    ) -> Dict:
        """
        Tạo gói prompt tối ưu cho executor.

        Args:
            query: Câu hỏi gốc của người dùng.
            category: Nhóm tác vụ do classifier trả về.
            context: Chuỗi ngữ cảnh RAG.
            history: Lịch sử hội thoại dạng list dict {"role", "content"}.
        """
        logger.info(f"Đang tối ưu prompt cho category: {category}")
        classification = classification or {}

        system_prompt = SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["general"])
        system_prompt = f"{system_prompt}\n\n{LANGUAGE_RULE}"

        if self.enable_cot and category in COT_TEMPLATES:
            user_prompt = COT_TEMPLATES[category].format(query=query)
        else:
            user_prompt = COT_TEMPLATES["default"].format(query=query)

        planning_context = self._build_planning_context(classification, history or [])
        if planning_context:
            user_prompt = planning_context + "\n\n" + user_prompt

        if context:
            context_header = (
                "RAG_CONTEXT là nguồn căn cứ chính. Ưu tiên ngữ cảnh này hơn kiến thức chung. "
                "Khi câu trả lời dùng tài liệu đã truy xuất, hãy nêu tên nguồn hoặc file liên quan. "
                "Nếu ngữ cảnh chưa đủ, hãy nói rõ đang thiếu thông tin gì trước khi trả lời.\n\n"
                f"RAG_CONTEXT:\n{context}\n\n"
            )
            user_prompt = context_header + user_prompt

        user_prompt = (
            "YÊU_CẦU_NGÔN_NGỮ: Nếu câu hỏi là tiếng Việt không dấu, hãy hiểu như tiếng Việt "
            "và trả lời bằng tiếng Việt có dấu đầy đủ.\n\n"
            + user_prompt
        )

        few_shot = []
        if self.enable_few_shot and category in FEW_SHOT_EXAMPLES:
            examples = FEW_SHOT_EXAMPLES[category]
            for ex in examples[:2]:
                few_shot.append({"role": "user", "content": ex["user"]})
                few_shot.append({"role": "assistant", "content": ex["assistant"]})

        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "few_shot": few_shot,
            "history": history or [],
            "category": category,
            "thinking": classification.get("thinking", ""),
            "intent": classification.get("intent", ""),
            "topic": classification.get("topic", ""),
            "subtasks": classification.get("subtasks", []),
        }

    def get_system_prompt(self, category: str) -> str:
        return SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["general"])

    def _build_planning_context(self, classification: Dict, history: List[Dict]) -> str:
        parts = []

        intent = classification.get("intent")
        topic = classification.get("topic")
        subtasks = classification.get("subtasks") or []

        if intent or topic or subtasks:
            lines = ["KẾ_HOẠCH_CLASSIFIER:"]
            if intent:
                lines.append(f"- Ý định: {intent}")
            if topic:
                lines.append(f"- Chủ đề: {topic}")
            if subtasks:
                lines.append("- Tác vụ con:")
                lines.extend(f"  {idx}. {step}" for idx, step in enumerate(subtasks, 1))
            parts.append("\n".join(lines))

        if history:
            memory_lines = []
            for msg in history[-8:]:
                role = msg.get("role", "user")
                content = str(msg.get("content", "")).strip()
                if content:
                    memory_lines.append(f"{role}: {content[:800]}")
            if memory_lines:
                parts.append("NGỮ_CẢNH_MEMORY:\n" + "\n".join(memory_lines))

        return "\n\n".join(parts)
