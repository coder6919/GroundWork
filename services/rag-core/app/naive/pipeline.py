"""Naive ingestion: same file loaders as Stage 1, naive fixed-length chunker,
a separate Qdrant collection. No registry, no dedup, no versioning - a "quick
RAG demo" wouldn't have any of that either.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from ..clients.qdrant import ensure_collection, make_client, upsert_points
from ..config import Settings
from ..ingestion.loaders import detect_mime_type, load_text
from ..ingestion.pipeline import discover_files
from ..logging import get_logger
from ..providers.base import EmbeddingProvider
from .chunking import naive_chunk

log = get_logger("naive.pipeline")


@dataclass
class NaiveIngestResult:
    filename: str
    chunk_count: int
    error: str | None = None


def naive_ingest_path(
    root: Path,
    *,
    settings: Settings,
    embedder: EmbeddingProvider,
) -> list[NaiveIngestResult]:
    client = make_client(timeout=30.0)
    ensure_collection(client, settings.qdrant_collection_naive, settings.embedding_dimensions)

    base = root if root.is_dir() else root.parent
    results: list[NaiveIngestResult] = []

    for path in discover_files(root):
        relative = path.relative_to(base)
        try:
            results.append(_ingest_one(path, relative, settings, embedder, client))
        except Exception as exc:  # noqa: BLE001 - isolate this file, keep the batch going
            log.warning("naive_ingest_file_failed", filename=str(relative), error=str(exc))
            results.append(NaiveIngestResult(str(relative), 0, error=str(exc)))

    return results


def _ingest_one(
    path: Path,
    relative: Path,
    settings: Settings,
    embedder: EmbeddingProvider,
    client: QdrantClient,
) -> NaiveIngestResult:
    detect_mime_type(path)  # validates the extension; raises for unsupported types
    chunks = naive_chunk(load_text(path))
    if not chunks:
        return NaiveIngestResult(str(relative), 0, error="no extractable text")

    vectors = embedder.embed(chunks, input_type="document")
    doc_id = uuid.uuid5(uuid.NAMESPACE_URL, f"naive:{relative}".replace("\\", "/")).hex
    now = datetime.now(UTC).isoformat()
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{i}")),
            vector=vector,
            payload={
                "doc_id": doc_id,
                "source_filename": str(relative),
                "chunk_text": chunk,
                "chunk_index": i,
                "ingested_at": now,
            },
        )
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    upsert_points(client, settings.qdrant_collection_naive, points)
    log.info("naive_ingest_file_done", filename=str(relative), chunks=len(chunks))
    return NaiveIngestResult(str(relative), len(chunks))
