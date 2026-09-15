"""Conversation session store: interface + SQLite implementation (Stage 7).

Same pattern as `registry.py`'s `DocumentRegistry` - kept behind an interface
so SQLite can be swapped for PostgreSQL later, per DIRECTION.md. Shares the
same SQLite file as the document registry (`settings.sqlite_path`); a
separate table needs no separate file.

Deliberately minimal: no session listing, deletion, or expiry. A `session_id`
is created implicitly the first time a caller passes one to
`POST /internal/query` - there is no dedicated session-management endpoint
yet (that's a Stage 9/10 concern once the public API and chat UI need it).
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from .models import Turn

_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_turns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    user_query      TEXT NOT NULL,
    rewritten_query TEXT NOT NULL,
    answer_text     TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session_turns_session_id ON session_turns (session_id);
"""


class SessionStore(ABC):
    @abstractmethod
    def get_turns(self, session_id: str) -> list[Turn]:
        """All turns for a session, oldest first. Empty list for an unknown
        or brand-new session_id - not an error."""
        ...

    @abstractmethod
    def add_turn(self, turn: Turn) -> None: ...


class SQLiteSessionStore(SessionStore):
    def __init__(self, db_path: str):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(path)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_turns(self, session_id: str) -> list[Turn]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM session_turns WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [_row_to_turn(r) for r in rows]

    def add_turn(self, turn: Turn) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_turns
                    (session_id, user_query, rewritten_query, answer_text, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    turn.session_id,
                    turn.user_query,
                    turn.rewritten_query,
                    turn.answer_text,
                    turn.created_at.isoformat(),
                ),
            )


def _row_to_turn(row: sqlite3.Row) -> Turn:
    return Turn(
        session_id=row["session_id"],
        user_query=row["user_query"],
        rewritten_query=row["rewritten_query"],
        answer_text=row["answer_text"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )
