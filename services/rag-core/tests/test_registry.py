from __future__ import annotations

from datetime import UTC
from pathlib import Path

import pytest

from app.db.models import DocumentRecord, DocumentStatus
from app.db.registry import SQLiteDocumentRegistry


@pytest.fixture
def registry(tmp_path: Path) -> SQLiteDocumentRegistry:
    return SQLiteDocumentRegistry(str(tmp_path / "sub" / "registry.db"))


def _record(**overrides) -> DocumentRecord:
    defaults = {
        "doc_id": "doc-1",
        "filename": "a.md",
        "source_path": "/data/a.md",
        "content_hash": "hash-1",
        "mime_type": "text/markdown",
        "status": DocumentStatus.READY,
        "chunk_count": 3,
    }
    defaults.update(overrides)
    return DocumentRecord(**defaults)


def test_creates_parent_directory_for_db_file(tmp_path: Path):
    db_path = tmp_path / "nested" / "dir" / "registry.db"
    SQLiteDocumentRegistry(str(db_path))
    assert db_path.exists()


def test_get_returns_none_for_unknown_doc(registry: SQLiteDocumentRegistry):
    assert registry.get("nope") is None


def test_upsert_then_get_roundtrips(registry: SQLiteDocumentRegistry):
    registry.upsert(_record())
    got = registry.get("doc-1")
    assert got is not None
    assert got.filename == "a.md"
    assert got.status == DocumentStatus.READY
    assert got.chunk_count == 3


def test_upsert_same_doc_id_updates_in_place(registry: SQLiteDocumentRegistry):
    registry.upsert(_record(status=DocumentStatus.PROCESSING, chunk_count=0))
    registry.upsert(_record(status=DocumentStatus.READY, chunk_count=5))
    got = registry.get("doc-1")
    assert got is not None
    assert got.status == DocumentStatus.READY
    assert got.chunk_count == 5
    assert len(registry.list()) == 1


def test_get_by_hash_finds_the_record(registry: SQLiteDocumentRegistry):
    registry.upsert(_record(doc_id="doc-1", content_hash="shared-hash"))
    got = registry.get_by_hash("shared-hash")
    assert got is not None
    assert got.doc_id == "doc-1"


def test_get_by_hash_returns_none_when_absent(registry: SQLiteDocumentRegistry):
    assert registry.get_by_hash("nowhere") is None


def test_list_orders_newest_first(registry: SQLiteDocumentRegistry):
    from datetime import datetime, timedelta

    now = datetime.now(UTC)
    registry.upsert(_record(doc_id="old", ingested_at=now - timedelta(days=1)))
    registry.upsert(_record(doc_id="new", ingested_at=now))
    ids = [r.doc_id for r in registry.list()]
    assert ids == ["new", "old"]


def test_failed_record_persists_failure_reason(registry: SQLiteDocumentRegistry):
    registry.upsert(_record(status=DocumentStatus.FAILED, failure_reason="bad encoding", chunk_count=0))
    got = registry.get("doc-1")
    assert got is not None
    assert got.status == DocumentStatus.FAILED
    assert got.failure_reason == "bad encoding"


def test_clear_removes_every_record(registry: SQLiteDocumentRegistry):
    registry.upsert(_record(doc_id="doc-1"))
    registry.upsert(_record(doc_id="doc-2", content_hash="hash-2"))

    registry.clear()

    assert registry.list() == []
    assert registry.get("doc-1") is None
