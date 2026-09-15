"""Route-level tests for POST /internal/ingest/upload (Stage 9) - the only
ingest route a public API layer is allowed to call, since it takes file
bytes rather than a caller-controlled server-side path.
"""

from __future__ import annotations

from app import main
from app.config import Settings
from app.db.models import DocumentStatus
from app.ingestion.pipeline import FileResult


def _install_fakes(monkeypatch, tmp_path, *, ingest_enabled=True):
    settings = Settings(
        rag_core_shared_secret="test-secret",
        storage_dir=str(tmp_path / "storage"),
        ingest_enabled=ingest_enabled,
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "get_registry", lambda: object())
    monkeypatch.setattr(main, "get_embedder", lambda: object())
    return settings


def test_uploads_a_file_ingests_it_and_deletes_the_temp_copy(client, monkeypatch, tmp_path):
    _install_fakes(monkeypatch, tmp_path)
    captured = {}

    def fake_ingest_path(root, **kwargs):
        captured["root"] = root
        captured["exists_during_call"] = root.exists()
        return [FileResult(doc_id="d1", filename=root.name, status=DocumentStatus.READY, chunk_count=2)]

    monkeypatch.setattr(main, "ingest_path", fake_ingest_path)

    r = client.post(
        "/internal/ingest/upload",
        headers={"X-Internal-Secret": "test-secret"},
        files={"file": ("policy.md", b"# Hello\n\nSome content.", "text/markdown")},
    )

    assert r.status_code == 200
    body = r.json()
    assert body["root"] == "policy.md"
    assert body["ready"] == 1
    assert captured["exists_during_call"] is True
    assert not captured["root"].exists()  # cleaned up after ingest_path returns


def test_rejects_unsupported_file_types(client, monkeypatch, tmp_path):
    _install_fakes(monkeypatch, tmp_path)

    r = client.post(
        "/internal/ingest/upload",
        headers={"X-Internal-Secret": "test-secret"},
        files={"file": ("payload.exe", b"MZ...", "application/octet-stream")},
    )

    assert r.status_code == 415


def test_sanitizes_a_path_traversal_filename(client, monkeypatch, tmp_path):
    _install_fakes(monkeypatch, tmp_path)
    captured = {}

    def fake_ingest_path(root, **kwargs):
        captured["root"] = root
        return [FileResult(doc_id="d1", filename=root.name, status=DocumentStatus.READY, chunk_count=1)]

    monkeypatch.setattr(main, "ingest_path", fake_ingest_path)

    r = client.post(
        "/internal/ingest/upload",
        headers={"X-Internal-Secret": "test-secret"},
        files={"file": ("../../etc/evil.md", b"# hi", "text/markdown")},
    )

    assert r.status_code == 200
    # only ever written under storage_dir/_uploads - never escapes it
    upload_dir = (tmp_path / "storage" / "_uploads").resolve()
    assert captured["root"].resolve().parent == upload_dir
    assert captured["root"].name.endswith("_evil.md")


def test_returns_403_when_ingestion_is_disabled(client, monkeypatch, tmp_path):
    _install_fakes(monkeypatch, tmp_path, ingest_enabled=False)

    r = client.post(
        "/internal/ingest/upload",
        headers={"X-Internal-Secret": "test-secret"},
        files={"file": ("policy.md", b"# Hello", "text/markdown")},
    )

    assert r.status_code == 403


def test_requires_the_shared_secret(client, monkeypatch, tmp_path):
    _install_fakes(monkeypatch, tmp_path)

    r = client.post(
        "/internal/ingest/upload",
        files={"file": ("policy.md", b"# Hello", "text/markdown")},
    )

    assert r.status_code == 401
