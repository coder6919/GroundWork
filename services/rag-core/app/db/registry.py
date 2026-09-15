"""Document registry: interface + SQLite implementation.

Kept behind `DocumentRegistry` so Stage 1's SQLite store can be swapped for
PostgreSQL later (per DIRECTION.md) without touching the ingestion pipeline.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from .models import DocumentRecord, DocumentStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    source_path     TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    mime_type       TEXT NOT NULL,
    status          TEXT NOT NULL,
    failure_reason  TEXT,
    page_count      INTEGER,
    version         TEXT,
    supersedes      TEXT,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    ingested_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents (content_hash);
"""


class DocumentRegistry(ABC):
    @abstractmethod
    def get(self, doc_id: str) -> DocumentRecord | None: ...

    @abstractmethod
    def get_by_hash(self, content_hash: str) -> DocumentRecord | None: ...

    @abstractmethod
    def upsert(self, record: DocumentRecord) -> None: ...

    @abstractmethod
    def list(self) -> list[DocumentRecord]: ...

    @abstractmethod
    def clear(self) -> None:
        """Wipe every record. Used when the Qdrant collection backing this
        registry was dropped and recreated (Stage 5's hybrid schema
        migration) - existing rows would otherwise lie about what's actually
        in Qdrant and cause re-ingestion to be skipped."""
        ...


class SQLiteDocumentRegistry(DocumentRegistry):
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

    def get(self, doc_id: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return _row_to_record(row) if row else None

    def get_by_hash(self, content_hash: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE content_hash = ? ORDER BY ingested_at DESC LIMIT 1",
                (content_hash,),
            ).fetchone()
        return _row_to_record(row) if row else None

    def upsert(self, record: DocumentRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents
                    (doc_id, filename, source_path, content_hash, mime_type, status,
                     failure_reason, page_count, version, supersedes, chunk_count, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    filename=excluded.filename,
                    source_path=excluded.source_path,
                    content_hash=excluded.content_hash,
                    mime_type=excluded.mime_type,
                    status=excluded.status,
                    failure_reason=excluded.failure_reason,
                    page_count=excluded.page_count,
                    version=excluded.version,
                    supersedes=excluded.supersedes,
                    chunk_count=excluded.chunk_count,
                    ingested_at=excluded.ingested_at
                """,
                (
                    record.doc_id,
                    record.filename,
                    record.source_path,
                    record.content_hash,
                    record.mime_type,
                    record.status.value,
                    record.failure_reason,
                    record.page_count,
                    record.version,
                    record.supersedes,
                    record.chunk_count,
                    record.ingested_at.isoformat(),
                ),
            )

    def list(self) -> list[DocumentRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY ingested_at DESC").fetchall()
        return [_row_to_record(r) for r in rows]

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM documents")


def _row_to_record(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        doc_id=row["doc_id"],
        filename=row["filename"],
        source_path=row["source_path"],
        content_hash=row["content_hash"],
        mime_type=row["mime_type"],
        status=DocumentStatus(row["status"]),
        failure_reason=row["failure_reason"],
        page_count=row["page_count"],
        version=row["version"],
        supersedes=row["supersedes"],
        chunk_count=row["chunk_count"],
        ingested_at=datetime.fromisoformat(row["ingested_at"]),
    )
