"""End-to-end ingestion: real Voyage embeddings + real Qdrant.

Gated behind RUN_INTEGRATION=1 because it spends (a few cents of) real API
credit. Uses a disposable, uniquely-named Qdrant collection that is dropped in
teardown, so it never touches the `knowledge_base` collection other stages use.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from app.config import get_settings
from app.db.models import DocumentStatus
from app.db.registry import SQLiteDocumentRegistry
from app.ingestion.pipeline import ingest_path
from app.providers import get_embedding_provider

pytestmark = pytest.mark.integration
_RUN = os.getenv("RUN_INTEGRATION") == "1"
_REASON = "set RUN_INTEGRATION=1 with Qdrant reachable and VOYAGE_API_KEY set"

GOOD_MD = """# Time Off Policy

## Accrual

Full-time employees accrue 1.25 days of paid time off per month.

## Carryover

Up to 5 unused days may carry over into the next calendar year.
"""

GOOD_TXT = "Payroll runs on the last business day of each month.\n\nDirect deposit changes need five business days notice.\n"


@pytest.fixture
def settings(tmp_path: Path):
    base = get_settings()
    unique_collection = f"test_ingest_{uuid.uuid4().hex[:8]}"
    s = base.model_copy(
        update={
            "qdrant_collection": unique_collection,
            "storage_dir": str(tmp_path / "storage"),
        }
    )
    yield s
    from app.clients.qdrant import make_client

    client = make_client(timeout=10.0)
    if client.collection_exists(unique_collection):
        client.delete_collection(unique_collection)


@pytest.fixture
def registry(tmp_path: Path) -> SQLiteDocumentRegistry:
    return SQLiteDocumentRegistry(str(tmp_path / "registry.db"))


@pytest.fixture
def embedder(settings):
    return get_embedding_provider(settings)


@pytest.mark.skipif(not _RUN, reason=_REASON)
def test_ingest_directory_end_to_end(tmp_path, settings, registry, embedder):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "vacation.md").write_text(GOOD_MD, encoding="utf-8")
    (corpus / "payroll.txt").write_text(GOOD_TXT, encoding="utf-8")
    # binary garbage under a supported extension - must fail in isolation, not abort the batch
    (corpus / "broken.txt").write_bytes(b"\xff\xfe\x00\x01\x02\xff\xff")

    results = ingest_path(corpus, settings=settings, registry=registry, embedder=embedder)
    by_name = {r.filename: r for r in results}

    assert by_name["vacation.md"].status == DocumentStatus.READY
    assert by_name["vacation.md"].chunk_count > 0
    assert by_name["payroll.txt"].status == DocumentStatus.READY

    # a genuinely undecodable file is isolated as a failure, not a crash
    assert "broken.txt" in by_name

    from app.clients.qdrant import make_client

    client = make_client(timeout=10.0)
    count = client.count(settings.qdrant_collection, exact=True).count
    assert count == by_name["vacation.md"].chunk_count + by_name["payroll.txt"].chunk_count

    point = client.scroll(settings.qdrant_collection, limit=1, with_payload=True)[0][0]
    payload = point.payload
    assert payload["tenant_id"] == "default"
    assert payload["chunk_text"]
    assert "doc_id" in payload


@pytest.mark.skipif(not _RUN, reason=_REASON)
def test_reingesting_unchanged_file_skips_reembedding(tmp_path, settings, registry, embedder):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "vacation.md").write_text(GOOD_MD, encoding="utf-8")

    first = ingest_path(corpus, settings=settings, registry=registry, embedder=embedder)
    second = ingest_path(corpus, settings=settings, registry=registry, embedder=embedder)

    assert first[0].skipped_duplicate is False
    assert second[0].skipped_duplicate is True
    assert second[0].status == DocumentStatus.READY


@pytest.mark.skipif(not _RUN, reason=_REASON)
def test_duplicate_content_under_a_new_filename_is_skipped(tmp_path, settings, registry, embedder):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "original.md").write_text(GOOD_MD, encoding="utf-8")

    ingest_path(corpus, settings=settings, registry=registry, embedder=embedder)

    (corpus / "copy-of-original.md").write_text(GOOD_MD, encoding="utf-8")
    results = ingest_path(corpus, settings=settings, registry=registry, embedder=embedder)
    by_name = {r.filename: r for r in results}

    assert by_name["copy-of-original.md"].skipped_duplicate is True
