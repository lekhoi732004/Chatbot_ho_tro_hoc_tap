"""
Chat history store.

Persists conversation sessions to disk so they survive process restarts.
Supports two backends:
  - "json"   : one JSON file per session in a history/ directory (default, zero deps)
  - "sqlite" : single SQLite database (faster for large histories)

Each persisted message carries a memory_score (set by MemoryJudge) indicating
how valuable it is to keep in the active context window.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger("chat_history")


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class PersistedMessage:
    """A single conversation message with scoring metadata."""
    msg_id: str                     # unique within session  (e.g. "msg_0001")
    session_id: str
    role: str                       # "user" | "assistant" | "system" | "file"
    content: str
    timestamp: float

    # Scoring fields (populated by MemoryJudge)
    memory_score: float = 0.5       # 0.0 = discard, 1.0 = always keep
    keep: bool = True               # False = pruned by judge
    score_reason: str = ""          # human-readable reason from judge
    scores_detail: Dict = field(default_factory=dict)  # sub-scores

    # Context fields
    category: str = ""              # task category from classifier
    metadata: Dict = field(default_factory=dict)  # extra metadata
    
    # File-specific fields
    is_file: bool = False           # True if this is a file upload
    file_name: str = ""             # name of uploaded file
    file_type: str = ""             # pdf, docx, image, text, etc.


@dataclass
class PersistedSession:
    """Full session with messages and aggregate stats."""
    session_id: str
    created_at: float
    last_active: float
    messages: List[PersistedMessage] = field(default_factory=list)
    summary: Optional[str] = None
    total_turns: int = 0            # total messages ever (including pruned)
    metadata: Dict = field(default_factory=dict)

    @property
    def kept_messages(self) -> List[PersistedMessage]:
        return [m for m in self.messages if m.keep]

    @property
    def pruned_messages(self) -> List[PersistedMessage]:
        return [m for m in self.messages if not m.keep]

    def as_history_dicts(self, kept_only: bool = True) -> List[Dict]:
        """Return messages as [{"role": ..., "content": ...}] for LLM input."""
        msgs = self.kept_messages if kept_only else self.messages
        return [{"role": m.role, "content": m.content} for m in msgs]


# ── JSON backend ──────────────────────────────────────────────────────────────

class JsonHistoryStore:
    """
    One directory per session under history_dir/.
    Session data file: <history_dir>/<session_id>/session.json

    Legacy compatibility:
    - older flat files at <history_dir>/<session_id>.json can still be read
    """

    def __init__(self, history_dir: str):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_session_id(session_id: str) -> str:
        return session_id.replace("/", "_").replace("\\", "_")

    def _session_dir(self, session_id: str) -> Path:
        return self.history_dir / self._safe_session_id(session_id)

    def _session_file(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "session.json"

    def _legacy_file(self, session_id: str) -> Path:
        return self.history_dir / f"{self._safe_session_id(session_id)}.json"

    def ensure_session_space(self, session_id: str) -> Path:
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _read_session_file(self, path: Path) -> Optional[PersistedSession]:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        messages = [PersistedMessage(**m) for m in data.get("messages", [])]
        return PersistedSession(
            session_id=data["session_id"],
            created_at=data["created_at"],
            last_active=data["last_active"],
            total_turns=data.get("total_turns", len(messages)),
            summary=data.get("summary"),
            metadata=data.get("metadata", {}),
            messages=messages,
        )

    def save_session(self, session: PersistedSession) -> None:
        data = {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "last_active": session.last_active,
            "total_turns": session.total_turns,
            "summary": session.summary,
            "metadata": session.metadata,
            "messages": [asdict(m) for m in session.messages],
        }
        self.ensure_session_space(session.session_id)
        path = self._session_file(session.session_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_session(self, session_id: str) -> Optional[PersistedSession]:
        current_path = self._session_file(session_id)
        if current_path.exists():
            return self._read_session_file(current_path)

        legacy_path = self._legacy_file(session_id)
        if legacy_path.exists():
            session = self._read_session_file(legacy_path)
            if session:
                self.save_session(session)
            return session
        return None

    def list_sessions(self) -> List[str]:
        session_ids = {
            p.name
            for p in self.history_dir.iterdir()
            if p.is_dir() and (p / "session.json").exists()
        }
        session_ids.update(
            p.stem
            for p in self.history_dir.glob("*.json")
            if p.name != ".gitkeep"
        )
        return sorted(session_ids)

    def delete_session(self, session_id: str) -> bool:
        deleted = False
        session_dir = self._session_dir(session_id)
        if session_dir.exists() and session_dir.is_dir():
            shutil.rmtree(session_dir)
            deleted = True

        legacy_path = self._legacy_file(session_id)
        if legacy_path.exists():
            legacy_path.unlink()
            deleted = True

        return deleted

    def session_exists(self, session_id: str) -> bool:
        return self._session_file(session_id).exists() or self._legacy_file(session_id).exists()

    def iter_sessions(self) -> Iterator[PersistedSession]:
        for sid in self.list_sessions():
            s = self.load_session(sid)
            if s:
                yield s


# ── SQLite backend ────────────────────────────────────────────────────────────

class SqliteHistoryStore:
    """
    All sessions in a single SQLite database.
    Better for deployments with many sessions.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(Path(db_path).parent, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id   TEXT PRIMARY KEY,
                    created_at   REAL,
                    last_active  REAL,
                    total_turns  INTEGER DEFAULT 0,
                    summary      TEXT,
                    metadata     TEXT DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS messages (
                    msg_id       TEXT,
                    session_id   TEXT,
                    role         TEXT,
                    content      TEXT,
                    timestamp    REAL,
                    memory_score REAL DEFAULT 0.5,
                    keep         INTEGER DEFAULT 1,
                    score_reason TEXT DEFAULT '',
                    scores_detail TEXT DEFAULT '{}',
                    category     TEXT DEFAULT '',
                    metadata     TEXT DEFAULT '{}',
                    PRIMARY KEY (session_id, msg_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id);
                CREATE INDEX IF NOT EXISTS idx_msg_keep    ON messages(session_id, keep);
            """)

    def save_session(self, session: PersistedSession) -> None:
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sessions
                  (session_id, created_at, last_active, total_turns, summary, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session.session_id, session.created_at, session.last_active,
                session.total_turns, session.summary,
                json.dumps(session.metadata),
            ))
            # Upsert messages
            for m in session.messages:
                conn.execute("""
                    INSERT OR REPLACE INTO messages
                      (msg_id, session_id, role, content, timestamp,
                       memory_score, keep, score_reason, scores_detail, category, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    m.msg_id, m.session_id, m.role, m.content, m.timestamp,
                    m.memory_score, int(m.keep), m.score_reason,
                    json.dumps(m.scores_detail), m.category,
                    json.dumps(m.metadata),
                ))

    def load_session(self, session_id: str) -> Optional[PersistedSession]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if not row:
                return None

            msg_rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp",
                (session_id,),
            ).fetchall()

        messages = [
            PersistedMessage(
                msg_id=r["msg_id"], session_id=r["session_id"],
                role=r["role"], content=r["content"], timestamp=r["timestamp"],
                memory_score=r["memory_score"], keep=bool(r["keep"]),
                score_reason=r["score_reason"],
                scores_detail=json.loads(r["scores_detail"] or "{}"),
                category=r["category"],
                metadata=json.loads(r["metadata"] or "{}"),
            )
            for r in msg_rows
        ]
        return PersistedSession(
            session_id=row["session_id"],
            created_at=row["created_at"],
            last_active=row["last_active"],
            total_turns=row["total_turns"],
            summary=row["summary"],
            metadata=json.loads(row["metadata"] or "{}"),
            messages=messages,
        )

    def list_sessions(self) -> List[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT session_id FROM sessions ORDER BY last_active DESC"
            ).fetchall()
        return [r["session_id"] for r in rows]

    def delete_session(self, session_id: str) -> bool:
        with self._conn() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            n = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            ).rowcount
        return n > 0

    def session_exists(self, session_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return row is not None

    def iter_sessions(self) -> Iterator[PersistedSession]:
        for sid in self.list_sessions():
            s = self.load_session(sid)
            if s:
                yield s


# ── Factory ───────────────────────────────────────────────────────────────────

def get_history_store(
    backend: Optional[str] = None,
    path: Optional[str] = None,
):
    """
    Return the configured history store.

    Args:
        backend: "json" or "sqlite". Reads from config if None.
        path: Storage path. Reads from config if None.

    Returns:
        JsonHistoryStore or SqliteHistoryStore.
    """
    cfg = get_config()
    backend = backend or cfg.get("memory.history_backend", "json")
    path    = path    or cfg.get("memory.history_path", "data/chat_history")

    if backend == "sqlite":
        db_path = path if path.endswith(".db") else path + "/history.db"
        logger.info(f"Using SQLite history store: {db_path}")
        return SqliteHistoryStore(db_path)
    else:
        logger.info(f"Using JSON history store: {path}")
        return JsonHistoryStore(path)
