from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.naive.pipeline import naive_ingest_path


class _FakeEmbedder:
    def embed(self, texts, input_type):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeQdrantClient:
    def __init__(self):
        self.upserted = []
        self.created_collections = []

    def collection_exists(self, name):
        return False

    def create_collection(self, collection_name, vectors_config):
        self.created_collections.append(collection_name)

    def upsert(self, collection_name, points, wait=True):
        self.upserted.append((collection_name, points))


@pytest.fixture(autouse=True)
def _patch_qdrant_client(monkeypatch):
    fake = _FakeQdrantClient()
    monkeypatch.setattr("app.naive.pipeline.make_client", lambda timeout=30.0: fake)
    return fake


def test_ingests_all_supported_files_into_the_naive_collection(tmp_path: Path, _patch_qdrant_client):
    (tmp_path / "a.md").write_text("# Title\n\nSome content here that is long enough.")
    (tmp_path / "b.txt").write_text("Plain text content for the second file.")

    settings = Settings(qdrant_collection_naive="knowledge_base_naive")
    results = naive_ingest_path(tmp_path, settings=settings, embedder=_FakeEmbedder())

    assert {r.filename for r in results} == {"a.md", "b.txt"}
    assert all(r.error is None for r in results)
    assert all(r.chunk_count > 0 for r in results)
    assert len(_patch_qdrant_client.upserted) == 2
    assert all(coll == "knowledge_base_naive" for coll, _ in _patch_qdrant_client.upserted)


def test_a_broken_file_does_not_abort_the_batch(tmp_path: Path, _patch_qdrant_client):
    (tmp_path / "good.md").write_text("Good content that will chunk fine.")
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe\x00\x01\x02\xff\xff")

    settings = Settings()

    class _ExplodingEmbedder:
        def embed(self, texts, input_type):
            if any("bad" in t for t in texts):
                raise RuntimeError("embedding failed")
            return [[0.1] for _ in texts]

    results = naive_ingest_path(tmp_path, settings=settings, embedder=_ExplodingEmbedder())

    by_name = {r.filename: r for r in results}
    assert by_name["good.md"].error is None
    assert by_name["good.md"].chunk_count > 0


def test_empty_file_reports_no_extractable_text(tmp_path: Path, _patch_qdrant_client):
    (tmp_path / "empty.txt").write_text("   \n\n  ")

    settings = Settings()
    results = naive_ingest_path(tmp_path, settings=settings, embedder=_FakeEmbedder())

    assert results[0].chunk_count == 0
    assert results[0].error == "no extractable text"
