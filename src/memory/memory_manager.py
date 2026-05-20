"""
Memory manager (v2) — unified conversation memory with:

  - In-memory session cache (fast access)
  - Persistent storage via ChatHistoryStore (JSON or SQLite)
  - MemoryJudge scoring + pruning on every batch
  - LLM-based summarisation for old message batches
  - Backward-compatible public API
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.memory.chat_history import PersistedMessage, PersistedSession, get_history_store
from src.memory.memory_judge import get_memory_judge
from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger("memory")


@dataclass
class Message:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


@dataclass
class Session:
    session_id: str
    messages: List[Message] = field(default_factory=list)
    summary: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


class MemoryManager:
    def __init__(self):
        cfg = get_config()
        self.max_history: int     = cfg.get("memory.max_history", 20)
        self.enable_summary: bool = cfg.get("memory.enable_summary", True)
        self.summary_trigger: int = cfg.get("memory.summary_trigger", 10)
        self.judge_every: int     = cfg.get("memory.judge_every", 5)
        self.persist: bool        = cfg.get("memory.persist", True)

        self._sessions: Dict[str, PersistedSession] = {}
        self._store = get_history_store() if self.persist else None
        self._judge = get_memory_judge()
        self._msg_counters: Dict[str, int] = {}

    def create_session(self, session_id: Optional[str] = None) -> str:
        if session_id is None:
            session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:12]
        if self.persist and self._store and self._store.session_exists(session_id):
            loaded = self._store.load_session(session_id)
            if loaded:
                self._sessions[session_id] = loaded
                self._msg_counters[session_id] = len(loaded.messages)
                logger.info(f"Restored session: {session_id} ({len(loaded.messages)} msgs)")
                return session_id
        session = PersistedSession(session_id=session_id, created_at=time.time(), last_active=time.time())
        self._sessions[session_id] = session
        self._msg_counters[session_id] = 0
        if self.persist and self._store:
            ensure_session_space = getattr(self._store, "ensure_session_space", None)
            if callable(ensure_session_space):
                ensure_session_space(session_id)
            self._store.save_session(session)
        logger.info(f"Created session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[PersistedSession]:
        if session_id not in self._sessions:
            self._load_or_create(session_id)
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._msg_counters.pop(session_id, None)
        if self.persist and self._store:
            self._store.delete_session(session_id)
        logger.info(f"Deleted session: {session_id}")

    def list_sessions(self) -> List[str]:
        if self.persist and self._store:
            return self._store.list_sessions()
        return list(self._sessions.keys())

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
        category: str = "",
    ) -> PersistedMessage:
        session = self._get_or_create(session_id)
        idx = self._msg_counters.get(session_id, 0)
        msg = PersistedMessage(
            msg_id=f"msg_{idx:04d}",
            session_id=session_id,
            role=role,
            content=content,
            timestamp=time.time(),
            category=category,
            metadata=metadata or {},
            is_file=False,
        )
        session.messages.append(msg)
        session.last_active = time.time()
        session.total_turns += 1
        self._msg_counters[session_id] = idx + 1

        if len(session.messages) % self.judge_every == 0:
            self._run_judge(session, self._latest_user_query(session))

        if self.enable_summary and len(session.messages) >= self.summary_trigger \
                and len(session.messages) % self.summary_trigger == 0:
            self._summarize(session)

        if self.persist and self._store:
            self._store.save_session(session)
        return msg

    def add_file_message(
        self,
        session_id: str,
        file_name: str,
        file_type: str,
        text_content: str,
        metadata: Optional[Dict] = None,
    ) -> PersistedMessage:
        """
        Add an uploaded file to session memory.
        
        Args:
            session_id: Session ID
            file_name: Original filename
            file_type: File type (pdf, docx, image, text, etc.)
            text_content: Extracted text content from file
            metadata: Optional metadata (chunks_created, source_type, etc.)
            
        Returns:
            PersistedMessage with is_file=True
        """
        session = self._get_or_create(session_id)
        idx = self._msg_counters.get(session_id, 0)
        
        # Create a file message with high memory score (files are important context)
        msg = PersistedMessage(
            msg_id=f"msg_{idx:04d}",
            session_id=session_id,
            role="file",  # Special role for files
            content=text_content,
            timestamp=time.time(),
            category="document_context",
            metadata=metadata or {},
            is_file=True,
            file_name=file_name,
            file_type=file_type,
        )
        # File messages get high initial score
        msg.memory_score = 0.95
        msg.keep = True
        msg.score_reason = "Uploaded file - high context value"
        
        session.messages.append(msg)
        session.last_active = time.time()
        session.total_turns += 1
        self._msg_counters[session_id] = idx + 1
        
        if self.persist and self._store:
            self._store.save_session(session)
        
        logger.info(f"Added file message: {file_name} ({file_type}) to session {session_id}")
        return msg

    def get_file_context(self, session_id: str) -> str:
        """
        Return uploaded file references for the session.
        
        File content is retrieved through the shared vector database, not injected
        directly into the prompt.
        """
        session = self._get_or_create(session_id)
        file_messages = [m for m in session.kept_messages if m.is_file]
        
        if not file_messages:
            return ""
        
        parts = ["[FILE ĐÃ TẢI LÊN]"]
        for msg in file_messages:
            parts.append(f"- {msg.file_name} ({msg.file_type.upper()}), đã index vào hệ truy vấn chung")
        
        return "\n".join(parts)

    def get_history(
        self,
        session_id: str,
        last_n: Optional[int] = None,
        as_dicts: bool = True,
        kept_only: bool = True,
    ) -> List:
        session = self._get_or_create(session_id)
        messages = session.kept_messages if kept_only else session.messages
        messages = [message for message in messages if not message.is_file]
        if last_n is not None:
            messages = messages[-last_n:]
        if as_dicts:
            return [{"role": m.role, "content": m.content} for m in messages]
        return messages

    def get_scored_history(self, session_id: str) -> List[PersistedMessage]:
        """Return all messages with scores (including pruned)."""
        return self._get_or_create(session_id).messages

    def get_context(self, session_id: str) -> str:
        """
        Get full context for LLM including files, summary, and recent messages.
        """
        session = self._get_or_create(session_id)
        parts = []
        
        # Add file context first (highest priority)
        file_context = self.get_file_context(session_id)
        if file_context:
            parts.append(file_context)
        
        # Add conversation summary
        if session.summary:
            parts.append(f"\n[Conversation Summary]\n{session.summary}")
        
        # Add recent messages (non-file)
        recent_msgs = [m for m in session.kept_messages[-6:] if not m.is_file]
        if recent_msgs:
            parts.append("\n[Recent Conversation]")
            for m in recent_msgs:
                parts.append(f"{m.role.upper()}: {m.content}")
        
        return "\n\n".join(parts)

    def clear_history(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session:
            session.messages = []
            session.summary = None
            if self.persist and self._store:
                self._store.save_session(session)

    def run_judge(self, session_id: str, latest_query: str = "") -> List[Dict]:
        """Manually trigger the memory judge. Returns score report."""
        session = self._get_or_create(session_id)
        self._run_judge(session, latest_query)
        if self.persist and self._store:
            self._store.save_session(session)
        return [
            {
                "msg_id": m.msg_id,
                "role": m.role,
                "content": m.content[:80],
                "memory_score": m.memory_score,
                "keep": m.keep,
                "reason": m.score_reason,
                "sub_scores": m.scores_detail,
            }
            for m in session.messages
        ]

    def _get_or_create(self, session_id: str) -> PersistedSession:
        if session_id not in self._sessions:
            self._load_or_create(session_id)
        return self._sessions[session_id]

    def _load_or_create(self, session_id: str) -> None:
        if self.persist and self._store and self._store.session_exists(session_id):
            loaded = self._store.load_session(session_id)
            if loaded:
                self._sessions[session_id] = loaded
                self._msg_counters[session_id] = len(loaded.messages)
                return
        self.create_session(session_id)

    def _latest_user_query(self, session: PersistedSession) -> str:
        for msg in reversed(session.messages):
            if msg.role == "user":
                return msg.content
        return ""

    def _run_judge(self, session: PersistedSession, latest_query: str) -> None:
        try:
            scores = self._judge.score_session(session, latest_query=latest_query)
            self._judge.apply_scores(session, scores)
        except Exception as e:
            logger.warning(f"Judge failed: {e}")

    def _summarize(self, session: PersistedSession) -> None:
        try:
            from src.utils.llm import load_llm
            llm = load_llm("optimizer")
            to_summarise = session.kept_messages[:-4]
            if not to_summarise:
                return
            text = "\n".join(f"{m.role}: {m.content}" for m in to_summarise)
            session.summary = llm.generate(
                (
                    "Bạn là bộ tóm tắt memory của Edu-Agent. Hãy tóm tắt cuộc hội thoại "
                    "để phục vụ hỏi đáp giáo dục về sau. Giữ lại mục tiêu của người dùng, "
                    "tên môn học/chủ đề, ràng buộc quan trọng và câu hỏi chưa giải quyết. "
                    "Viết trong 3-5 câu ngắn gọn.\n\n"
                    f"Cuộc hội thoại:\n{text}"
                ),
                max_new_tokens=256,
                temperature=0.3,
            )
            for msg in to_summarise:
                msg.memory_score = min(msg.memory_score, 0.2)
                msg.keep = False
                msg.score_reason += " | summarised"
            logger.info(f"Summarised session {session.session_id}")
        except Exception as e:
            logger.warning(f"Summarisation failed: {e}")

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)


_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager
