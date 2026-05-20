"""
Memory Judge.

Scores every message in a session across 6 dimensions and decides
which messages are worth keeping in the active context window.

Scoring dimensions (each 0.0 – 1.0):
  1. recency       — how recent the message is (exponential decay)
  2. informativeness — length + lexical richness
  3. relevance     — semantic similarity to the latest user query
  4. uniqueness    — how different it is from nearby messages (dedup)
  5. role_weight   — assistant answers weighted higher than filler
  6. user_signal   — explicit feedback, reactions, follow-up depth

Final memory_score = weighted sum of the 6 sub-scores.

Pruning strategies:
  - "threshold"  : keep all messages with score >= min_score
  - "top_k"      : keep the top-K messages regardless of threshold
  - "budget"     : keep as many as fit within a token budget
  - "hybrid"     : top_k + threshold (always keep recent N, then score the rest)

The judge can operate:
  - "heuristic"  : fast formula, no LLM required (default)
  - "llm"        : dùng model optimizer để chấm mức quan trọng (chậm nhưng sâu hơn)
  - "hybrid"     : heuristic pre-filter → LLM re-score borderline messages
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.memory.chat_history import PersistedMessage, PersistedSession
from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger("memory_judge")


# ── Score weights ─────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "recency":        0.25,
    "informativeness": 0.20,
    "relevance":      0.25,
    "uniqueness":     0.15,
    "role_weight":    0.10,
    "user_signal":    0.05,
}

# Role base weights (before other scoring)
ROLE_BASE = {
    "system":    0.9,   # system prompts almost always kept
    "assistant": 0.6,
    "user":      0.5,
}


# ── Sub-scorers ───────────────────────────────────────────────────────────────

def score_recency(
    msg: PersistedMessage,
    now: float,
    half_life_hours: float = 24.0,
) -> float:
    """
    Exponential decay: score = exp(-λ * age_hours)
    where λ = ln(2) / half_life_hours.
    Recent messages get ~1.0; messages older than half_life get ~0.5.
    """
    age_hours = (now - msg.timestamp) / 3600.0
    lam = math.log(2) / max(half_life_hours, 0.1)
    return math.exp(-lam * age_hours)


def score_informativeness(msg: PersistedMessage) -> float:
    """
    Combines:
      - normalised word count (longer = more informative, up to a ceiling)
      - type-token ratio (lexical diversity)
      - presence of structured content (code, lists, numbers)
    """
    text = msg.content.strip()
    if not text:
        return 0.0

    words = text.split()
    word_count = len(words)

    # Length score — sigmoid centred at 80 words
    length_score = 1 / (1 + math.exp(-0.05 * (word_count - 80)))

    # Type-token ratio (unique words / total words), capped sample
    sample = words[:200]
    ttr = len(set(w.lower() for w in sample)) / max(len(sample), 1)

    # Structured content bonus
    has_code    = 0.2 if "```" in text or "    " in text else 0.0
    has_list    = 0.1 if re.search(r"(?m)^[\-\*\d]\.", text) else 0.0
    has_numbers = 0.1 if re.search(r"\d+", text) else 0.0

    raw = (length_score * 0.4) + (ttr * 0.4) + has_code + has_list + has_numbers
    return min(raw, 1.0)


def score_relevance(
    msg: PersistedMessage,
    latest_query: str,
    use_embedding: bool = False,
) -> float:
    """
    Measures how relevant the message is to the current query.

    If use_embedding=True and the embedding model is loaded, uses cosine similarity.
    Otherwise falls back to keyword overlap (Jaccard).
    """
    if not latest_query.strip():
        return 0.5

    if use_embedding:
        try:
            from src.rag.embedder import embed_texts
            import numpy as np
            vecs = embed_texts([msg.content[:512], latest_query[:512]], show_progress=False)
            # cosine similarity (already normalised)
            sim = float(np.dot(vecs[0], vecs[1]))
            return max(0.0, min(sim, 1.0))
        except Exception:
            pass  # fall through to keyword

    # Jaccard keyword overlap
    def tokens(t: str) -> set:
        return {w.lower() for w in re.findall(r"\w+", t) if len(w) > 2}

    q_tok = tokens(latest_query)
    m_tok = tokens(msg.content)
    if not q_tok or not m_tok:
        return 0.3
    inter = len(q_tok & m_tok)
    union = len(q_tok | m_tok)
    return inter / union if union else 0.0


def score_uniqueness(
    msg: PersistedMessage,
    all_messages: List[PersistedMessage],
    window: int = 4,
) -> float:
    """
    Penalise messages that are very similar to their neighbours.
    Uses character n-gram overlap against the surrounding ±window messages.
    Score of 1.0 = completely unique; 0.0 = near-duplicate.
    """
    idx = next((i for i, m in enumerate(all_messages) if m.msg_id == msg.msg_id), -1)
    if idx < 0:
        return 0.8

    neighbours = (
        all_messages[max(0, idx - window) : idx] +
        all_messages[idx + 1 : idx + window + 1]
    )
    if not neighbours:
        return 1.0

    def ngrams(text: str, n: int = 3) -> set:
        t = text.lower()
        return {t[i:i+n] for i in range(len(t) - n + 1)}

    msg_ng = ngrams(msg.content)
    if not msg_ng:
        return 0.5

    max_overlap = 0.0
    for nb in neighbours:
        nb_ng = ngrams(nb.content)
        if not nb_ng:
            continue
        overlap = len(msg_ng & nb_ng) / len(msg_ng | nb_ng)
        max_overlap = max(max_overlap, overlap)

    return 1.0 - max_overlap


def score_role_weight(msg: PersistedMessage) -> float:
    """Base score determined by role."""
    return ROLE_BASE.get(msg.role, 0.5)


def score_user_signal(msg: PersistedMessage) -> float:
    """
    Detect explicit positive signals in message metadata or content.
    Positive: user says 'thanks', asks follow-up, message has high category weight.
    Negative: very short filler, repeated question.
    """
    base = 0.5
    content_lower = msg.content.lower()

    # Positive signals
    positive = ["thank", "cảm ơn", "great", "perfect", "exactly", "helpful",
                "explain more", "tell me more", "follow up", "continue"]
    negative = ["ok", "okay", "sure", "yes", "no", "got it", "i see", "alright"]
    filler   = len(msg.content.strip()) < 12

    if filler:
        return 0.1
    if any(p in content_lower for p in positive):
        return min(base + 0.3, 1.0)
    if any(n == content_lower.strip() for n in negative):
        return 0.2
    return base


# ── Score record ──────────────────────────────────────────────────────────────

@dataclass
class MessageScore:
    msg_id: str
    session_id: str
    memory_score: float
    keep: bool
    reason: str
    sub_scores: Dict[str, float] = field(default_factory=dict)


# ── Memory Judge ──────────────────────────────────────────────────────────────

class MemoryJudge:
    """
    Scores all messages in a session and marks which to keep.

    Usage:
        judge = MemoryJudge()
        scores = judge.score_session(session, latest_query="...")
        judge.apply_scores(session, scores)  # sets msg.keep and msg.memory_score
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        strategy: str = "hybrid",
        min_score: float = 0.35,
        top_k: int = 12,
        token_budget: int = 3000,
        always_keep_last_n: int = 4,
        use_embedding_relevance: bool = False,
        llm_mode: str = "heuristic",
        recency_half_life_hours: float = 24.0,
    ):
        cfg = get_config()
        self.weights = weights or cfg.get("memory.judge_weights") or DEFAULT_WEIGHTS
        self.strategy               = strategy
        self.min_score              = min_score
        self.top_k                  = top_k
        self.token_budget           = token_budget
        self.always_keep_last_n     = always_keep_last_n
        self.use_embedding_relevance = use_embedding_relevance
        self.llm_mode               = llm_mode
        self.recency_half_life_hours = recency_half_life_hours

    # ── Public API ────────────────────────────────────────────────────────────

    def score_session(
        self,
        session: PersistedSession,
        latest_query: str = "",
        now: Optional[float] = None,
    ) -> List[MessageScore]:
        """
        Score every message in the session.

        Args:
            session: The session to score.
            latest_query: The most recent user query (used for relevance scoring).
            now: Override current time (for testing).

        Returns:
            List of MessageScore, one per message.
        """
        now = now or time.time()
        messages = session.messages
        scores: List[MessageScore] = []

        for msg in messages:
            sub = self._compute_sub_scores(msg, messages, latest_query, now)
            final = self._weighted_sum(sub)
            scores.append(MessageScore(
                msg_id=msg.msg_id,
                session_id=msg.session_id,
                memory_score=round(final, 4),
                keep=True,          # keep decision applied in apply_scores
                reason=self._reason(sub, final),
                sub_scores=sub,
            ))

        # LLM re-score borderline messages
        if self.llm_mode in ("llm", "hybrid"):
            scores = self._llm_rescore(messages, scores, latest_query)

        # Apply keep/prune decision
        scores = self._apply_strategy(messages, scores)

        logger.info(
            f"Scored {len(scores)} messages: "
            f"{sum(1 for s in scores if s.keep)} keep, "
            f"{sum(1 for s in scores if not s.keep)} prune"
        )
        return scores

    def apply_scores(
        self,
        session: PersistedSession,
        scores: List[MessageScore],
    ) -> None:
        """Write scores back to session messages in-place."""
        score_map = {s.msg_id: s for s in scores}
        for msg in session.messages:
            if msg.msg_id in score_map:
                s = score_map[msg.msg_id]
                msg.memory_score = s.memory_score
                msg.keep         = s.keep
                msg.score_reason = s.reason
                msg.scores_detail = s.sub_scores

    # ── Sub-score computation ─────────────────────────────────────────────────

    def _compute_sub_scores(
        self,
        msg: PersistedMessage,
        all_messages: List[PersistedMessage],
        latest_query: str,
        now: float,
    ) -> Dict[str, float]:
        return {
            "recency":        score_recency(msg, now, self.recency_half_life_hours),
            "informativeness": score_informativeness(msg),
            "relevance":      score_relevance(msg, latest_query, self.use_embedding_relevance),
            "uniqueness":     score_uniqueness(msg, all_messages),
            "role_weight":    score_role_weight(msg),
            "user_signal":    score_user_signal(msg),
        }

    def _weighted_sum(self, sub: Dict[str, float]) -> float:
        total_w = sum(self.weights.values())
        score = sum(sub.get(k, 0) * w for k, w in self.weights.items())
        return score / max(total_w, 1.0)

    def _reason(self, sub: Dict[str, float], final: float) -> str:
        dominant = max(sub, key=sub.get)
        weak     = min(sub, key=sub.get)
        return (
            f"score={final:.2f} | "
            f"strong={dominant}({sub[dominant]:.2f}) | "
            f"weak={weak}({sub[weak]:.2f})"
        )

    # ── Pruning strategies ────────────────────────────────────────────────────

    def _apply_strategy(
        self,
        messages: List[PersistedMessage],
        scores: List[MessageScore],
    ) -> List[MessageScore]:
        """Mark keep/prune on scores according to chosen strategy."""
        score_map = {s.msg_id: s for s in scores}
        n = len(messages)

        # Always-keep guard: most recent N messages always survive
        protected_ids = {
            messages[i].msg_id
            for i in range(max(0, n - self.always_keep_last_n), n)
        }
        # System messages always kept
        system_ids = {m.msg_id for m in messages if m.role == "system"}
        always_keep = protected_ids | system_ids

        if self.strategy == "threshold":
            for s in scores:
                s.keep = s.msg_id in always_keep or s.memory_score >= self.min_score

        elif self.strategy == "top_k":
            sorted_ids = sorted(
                [s.msg_id for s in scores],
                key=lambda mid: score_map[mid].memory_score,
                reverse=True,
            )
            keep_ids = set(sorted_ids[:self.top_k]) | always_keep
            for s in scores:
                s.keep = s.msg_id in keep_ids

        elif self.strategy == "budget":
            # Greedy: fill token budget in order of score, always keep protected
            remaining = self.token_budget
            keep_ids  = set(always_keep)
            # Pre-consume budget for protected messages
            for mid in always_keep:
                if mid in score_map:
                    remaining -= _token_est(score_map[mid])

            sorted_scores = sorted(
                [s for s in scores if s.msg_id not in always_keep],
                key=lambda s: s.memory_score,
                reverse=True,
            )
            for s in sorted_scores:
                cost = _token_est(s)
                if remaining - cost >= 0:
                    keep_ids.add(s.msg_id)
                    remaining -= cost
            for s in scores:
                s.keep = s.msg_id in keep_ids

        elif self.strategy == "hybrid":
            # Keep top-K + anything above threshold, always protect recent
            sorted_ids = sorted(
                [s.msg_id for s in scores],
                key=lambda mid: score_map[mid].memory_score,
                reverse=True,
            )
            keep_ids = (
                set(sorted_ids[:self.top_k])
                | {s.msg_id for s in scores if s.memory_score >= self.min_score}
                | always_keep
            )
            for s in scores:
                s.keep = s.msg_id in keep_ids
        else:
            # Default: keep all
            for s in scores:
                s.keep = True

        return scores

    # ── LLM re-scoring ────────────────────────────────────────────────────────

    def _llm_rescore(
        self,
        messages: List[PersistedMessage],
        scores: List[MessageScore],
        latest_query: str,
        border_margin: float = 0.15,
    ) -> List[MessageScore]:
        """
        Re-score messages whose heuristic score falls in the borderline zone
        [min_score - margin, min_score + margin] bằng model optimizer.
        """
        borderline = [
            s for s in scores
            if abs(s.memory_score - self.min_score) <= border_margin
        ]

        if not borderline:
            return scores

        try:
            from src.utils.llm import load_llm
            import json as _json
            llm = load_llm("optimizer")

            msg_map = {m.msg_id: m for m in messages}
            for s in borderline:
                msg = msg_map.get(s.msg_id)
                if not msg:
                    continue

                prompt = _LLM_SCORE_PROMPT.format(
                    query=latest_query[:300],
                    role=msg.role,
                    content=msg.content[:400],
                )
                raw = llm.generate(prompt, max_new_tokens=80, temperature=0.1)
                try:
                    import re as _re
                    m = _re.search(r"\{.*?\}", raw, _re.DOTALL)
                    if m:
                        data = _json.loads(m.group())
                        llm_score = float(data.get("score", s.memory_score))
                        llm_score = max(0.0, min(1.0, llm_score))
                        # Blend: 40% LLM, 60% heuristic
                        s.memory_score = round(0.6 * s.memory_score + 0.4 * llm_score, 4)
                        s.reason += f" | llm_score={llm_score:.2f}"
                        s.sub_scores["llm"] = llm_score
                except Exception as parse_err:
                    logger.debug(f"LLM score parse failed: {parse_err}")

        except Exception as e:
            logger.warning(f"LLM re-scoring skipped: {e}")

        return scores


def _token_est(s: MessageScore) -> int:
    """Rough token estimate: ~1.3 tokens per word."""
    return int(len(s.reason.split()) * 1.3) + 20  # rough; real content not in MessageScore


_LLM_SCORE_PROMPT = """Bạn là bộ đánh giá mức liên quan của memory trong Edu-Agent.
Hãy chấm mức quan trọng của tin nhắn hội thoại này đối với việc trả lời truy vấn hiện tại.
Chấm từ 0.0 (nội dung thừa, không liên quan) đến 1.0 (ngữ cảnh rất quan trọng cần giữ lại).

Truy vấn hiện tại: {query}

Tin nhắn:
  Vai trò: {role}
  Nội dung: {content}

Chỉ trả về JSON: {{"score": <float 0-1>, "reason": "<lý do ngắn>"}}"""


# ── Module-level singleton ────────────────────────────────────────────────────

_judge: Optional[MemoryJudge] = None


def get_memory_judge() -> MemoryJudge:
    global _judge
    if _judge is None:
        cfg = get_config()
        _judge = MemoryJudge(
            strategy=cfg.get("memory.judge_strategy", "hybrid"),
            min_score=cfg.get("memory.judge_min_score", 0.35),
            top_k=cfg.get("memory.judge_top_k", 12),
            token_budget=cfg.get("memory.judge_token_budget", 3000),
            always_keep_last_n=cfg.get("memory.always_keep_last_n", 4),
            use_embedding_relevance=cfg.get("memory.judge_use_embedding", False),
            llm_mode=cfg.get("memory.judge_llm_mode", "heuristic"),
            recency_half_life_hours=cfg.get("memory.recency_half_life_hours", 24.0),
        )
    return _judge
