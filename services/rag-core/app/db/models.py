"""Document registry and conversation-session data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    UNPROCESSABLE = "unprocessable"


@dataclass
class DocumentRecord:
    doc_id: str
    filename: str
    source_path: str
    content_hash: str
    mime_type: str
    status: DocumentStatus = DocumentStatus.PENDING
    failure_reason: str | None = None
    page_count: int | None = None
    version: str | None = None
    supersedes: str | None = None
    chunk_count: int = 0
    ingested_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class Turn:
    """One question/answer exchange in a multi-turn session (Stage 7).

    `user_query` is exactly what the caller sent; `rewritten_query` is what
    was actually searched on and answered (identical to `user_query` when
    this was the session's first turn, or when rewriting made no change).
    Kept separate so the history fed back into future rewrites shows the raw
    conversation as the user experienced it, not the rewritten form.
    """

    session_id: str
    user_query: str
    rewritten_query: str
    answer_text: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
