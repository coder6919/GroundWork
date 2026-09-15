from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.db.registry import SQLiteDocumentRegistry
from app.ingestion.pipeline import _stable_doc_id, ingest_path


class _FakeEmbedder:
    def embed(self, texts, input_type):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeQdrantClient:
    def __init__(self, existing_vectors=None):
        self._exists = existing_vectors is not None
        self._vectors = existing_vectors
        self.upserted_points = []
        self.set_payload_calls = []
        self.deleted_collections = []

    def collection_exists(self, name):
        return self._exists

    def get_collection(self, name):
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=self._vectors)))

    def delete_collection(self, name):
        self.deleted_collections.append(name)
        self._exists = False

    def create_collection(self, collection_name, vectors_config, sparse_vectors_config=None):
        self._exists = True
        self._vectors = vectors_config

    def delete(self, collection_name, points_selector, **kwargs):
        pass

    def upsert(self, collection_name, points, wait=True):
        self.upserted_points.extend(points)

    def set_payload(self, collection_name, payload, points, **kwargs):
        self.set_payload_calls.append((payload, points))

    def count(self, collection_name, exact=True):
        return SimpleNamespace(count=len(self.upserted_points))


@pytest.fixture(autouse=True)
def _patch_qdrant(monkeypatch):
    fake = _FakeQdrantClient()
    monkeypatch.setattr("app.ingestion.pipeline.make_client", lambda timeout=30.0: fake)
    return fake


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_supersedes_marks_the_old_documents_points_and_records_it_on_the_new_doc(
    tmp_path: Path, _patch_qdrant
):
    _write(tmp_path / "policy-v1.md", "# Policy\n\nOriginal wording of the vacation policy.")
    _write(tmp_path / "policy-v2.md", "# Policy\n\nUpdated wording of the vacation policy.")
    settings = Settings(sqlite_path=str(tmp_path / "registry.db"), storage_dir=str(tmp_path / "storage"))
    registry = SQLiteDocumentRegistry(settings.sqlite_path)

    old_doc_id = _stable_doc_id(Path("policy-v1.md"))

    ingest_path(
        tmp_path,
        settings=settings,
        registry=registry,
        embedder=_FakeEmbedder(),
        supersedes={"policy-v2.md": "policy-v1.md"},
        versions={"policy-v1.md": "v1", "policy-v2.md": "v2"},
    )

    new_doc_id = _stable_doc_id(Path("policy-v2.md"))
    new_record = registry.get(new_doc_id)
    assert new_record is not None
    assert new_record.version == "v2"
    assert new_record.supersedes == old_doc_id

    old_record = registry.get(old_doc_id)
    assert old_record is not None
    assert old_record.version == "v1"

    [(payload, selector)] = _patch_qdrant.set_payload_calls
    assert payload == {"superseded": True}
    assert selector.must[0].match.value == old_doc_id

    new_points = [p for p in _patch_qdrant.upserted_points if p.payload["doc_id"] == new_doc_id]
    assert new_points
    assert all(p.payload["doc_version"] == "v2" for p in new_points)
    assert all(p.payload["supersedes_doc_id"] == old_doc_id for p in new_points)
    assert all(p.payload["superseded"] is False for p in new_points)


def test_ingest_without_supersedes_leaves_superseded_flag_false_and_no_payload_calls(
    tmp_path: Path, _patch_qdrant
):
    _write(tmp_path / "standalone.md", "# Doc\n\nJust a normal document.")
    settings = Settings(sqlite_path=str(tmp_path / "registry.db"), storage_dir=str(tmp_path / "storage"))
    registry = SQLiteDocumentRegistry(settings.sqlite_path)

    ingest_path(tmp_path, settings=settings, registry=registry, embedder=_FakeEmbedder())

    assert _patch_qdrant.set_payload_calls == []
    assert all(p.payload["superseded"] is False for p in _patch_qdrant.upserted_points)
    assert all(p.payload["doc_version"] is None for p in _patch_qdrant.upserted_points)


def test_schema_migration_clears_stale_registry_rows_before_reingesting(
    tmp_path: Path, monkeypatch
):
    """Regression test: a real live run found that dropping/recreating the
    Qdrant collection (old single-vector -> hybrid schema) left the registry
    claiming files were already READY when their Qdrant points no longer
    existed, so re-ingestion was silently skipped. The collection must clear
    the registry whenever it has to recreate itself."""
    _write(tmp_path / "doc.md", "# Doc\n\nOriginal content.")
    settings = Settings(sqlite_path=str(tmp_path / "registry.db"), storage_dir=str(tmp_path / "storage"))
    registry = SQLiteDocumentRegistry(settings.sqlite_path)

    fresh_client = _FakeQdrantClient(existing_vectors=None)
    monkeypatch.setattr("app.ingestion.pipeline.make_client", lambda timeout=30.0: fresh_client)
    first = ingest_path(tmp_path, settings=settings, registry=registry, embedder=_FakeEmbedder())
    assert first[0].skipped_duplicate is False
    assert len(registry.list()) == 1

    # Simulate this collection still being on the old (pre-Stage-5) schema.
    outdated_client = _FakeQdrantClient(existing_vectors=object())
    monkeypatch.setattr("app.ingestion.pipeline.make_client", lambda timeout=30.0: outdated_client)
    second = ingest_path(tmp_path, settings=settings, registry=registry, embedder=_FakeEmbedder())

    assert outdated_client.deleted_collections == [settings.qdrant_collection]
    assert second[0].skipped_duplicate is False
    assert len(registry.list()) == 1


def test_pointing_at_a_different_empty_collection_also_clears_a_stale_registry(
    tmp_path: Path, monkeypatch
):
    """Regression test: live-caught pointing rag-core at a brand-new Qdrant
    Cloud cluster while the local SQLite registry still had rows from an
    earlier ingest into local Qdrant. ensure_hybrid_collection only reports
    "recreated" when it drops an outdated SAME collection - a collection
    that's simply new on this target (never existed here) returns False, so
    the old schema-migration-only check missed this case entirely:
    POST /internal/ingest reported every file "skipped_duplicate": true
    against a collection that stayed at 0 points. Any 0-point collection
    must never be trusted to match a non-empty registry, regardless of why
    they diverged."""
    _write(tmp_path / "doc.md", "# Doc\n\nOriginal content.")
    settings = Settings(sqlite_path=str(tmp_path / "registry.db"), storage_dir=str(tmp_path / "storage"))
    registry = SQLiteDocumentRegistry(settings.sqlite_path)

    first_client = _FakeQdrantClient(existing_vectors=None)
    monkeypatch.setattr("app.ingestion.pipeline.make_client", lambda timeout=30.0: first_client)
    first = ingest_path(tmp_path, settings=settings, registry=registry, embedder=_FakeEmbedder())
    assert first[0].skipped_duplicate is False
    assert len(registry.list()) == 1

    # A different Qdrant target (e.g. a fresh cloud cluster) - the collection
    # doesn't exist there yet either, so ensure_hybrid_collection reports
    # False (not "recreated"), same as any brand-new collection.
    new_target_client = _FakeQdrantClient(existing_vectors=None)
    monkeypatch.setattr("app.ingestion.pipeline.make_client", lambda timeout=30.0: new_target_client)
    second = ingest_path(tmp_path, settings=settings, registry=registry, embedder=_FakeEmbedder())

    assert second[0].skipped_duplicate is False
    assert len(new_target_client.upserted_points) > 0
    assert len(registry.list()) == 1


def test_a_populated_collection_on_a_new_target_is_left_alone(tmp_path: Path, monkeypatch):
    """The new empty-collection check must not fire when the target
    collection is new to ensure_hybrid_collection's eyes but already has
    points (e.g. a second rag-core instance pointed at a Qdrant that
    already has real data) - only a genuinely empty collection is
    grounds for distrusting the registry."""
    _write(tmp_path / "doc.md", "# Doc\n\nOriginal content.")
    settings = Settings(sqlite_path=str(tmp_path / "registry.db"), storage_dir=str(tmp_path / "storage"))
    registry = SQLiteDocumentRegistry(settings.sqlite_path)

    first_client = _FakeQdrantClient(existing_vectors=None)
    monkeypatch.setattr("app.ingestion.pipeline.make_client", lambda timeout=30.0: first_client)
    ingest_path(tmp_path, settings=settings, registry=registry, embedder=_FakeEmbedder())
    assert len(registry.list()) == 1

    already_populated_client = _FakeQdrantClient(existing_vectors={"dense": object()})
    already_populated_client.upserted_points = list(first_client.upserted_points)
    monkeypatch.setattr("app.ingestion.pipeline.make_client", lambda timeout=30.0: already_populated_client)
    second = ingest_path(tmp_path, settings=settings, registry=registry, embedder=_FakeEmbedder())

    assert second[0].skipped_duplicate is True
    assert len(registry.list()) == 1
