from __future__ import annotations

from datetime import UTC, datetime

from app import main
from app.db.models import DocumentRecord, DocumentStatus


def _record(**overrides) -> DocumentRecord:
    defaults = {
        "doc_id": "abc123",
        "filename": "refund-policy.md",
        "source_path": "/data/storage/abc123.md",
        "content_hash": "h",
        "mime_type": "text/markdown",
        "status": DocumentStatus.READY,
        "chunk_count": 6,
        "ingested_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    defaults.update(overrides)
    return DocumentRecord(**defaults)


def test_returns_404_for_unknown_doc_id(client, monkeypatch):
    monkeypatch.setattr(main, "get_registry", lambda: type("R", (), {"get": staticmethod(lambda doc_id: None)})())

    r = client.get("/internal/sources/nope", headers={"X-Internal-Secret": "test-secret"})

    assert r.status_code == 404


def test_returns_record_metadata_for_a_known_doc_id(client, monkeypatch):
    record = _record()
    monkeypatch.setattr(
        main,
        "get_registry",
        lambda: type("R", (), {"get": staticmethod(lambda doc_id: record if doc_id == "abc123" else None)})(),
    )

    r = client.get("/internal/sources/abc123", headers={"X-Internal-Secret": "test-secret"})

    assert r.status_code == 200
    body = r.json()
    assert body["doc_id"] == "abc123"
    assert body["filename"] == "refund-policy.md"
    assert body["chunk_count"] == 6
    assert body["status"] == "ready"
    assert body["ingested_at"] == "2026-01-01T00:00:00+00:00"


def test_requires_the_shared_secret(client):
    r = client.get("/internal/sources/abc123")
    assert r.status_code == 401
