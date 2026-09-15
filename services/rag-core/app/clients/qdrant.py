"""Qdrant connectivity.

Qdrant is a self-hosted, no-cost dependency - nothing here calls a paid API.
Stage 0 only checked reachability. Stage 1 adds collection creation and
point upsert/delete for the ingestion pipeline.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    Modifier,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

from ..config import get_settings
from ..logging import get_logger

log = get_logger("qdrant")

# Named vectors for the hybrid (dense + sparse BM25) collection - Stage 5.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def make_client(timeout: float = 5.0) -> QdrantClient:
    s = get_settings()
    return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key or None, timeout=timeout)


def ensure_collection(client: QdrantClient, name: str, dimensions: int) -> None:
    """Idempotent: creates the collection only if it does not already exist.

    Single unnamed dense vector - used by the Stage 3 naive baseline, which
    deliberately stays vector-only (no hybrid, no rerank; see `ensure_hybrid_collection`
    for the improved pipeline's collection)."""
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE),
        )
        log.info("qdrant_collection_created", collection=name, dimensions=dimensions)


def ensure_hybrid_collection(client: QdrantClient, name: str, dimensions: int) -> bool:
    """Idempotent: creates a named dense+sparse collection for hybrid retrieval.
    Returns True if an existing collection had to be dropped and recreated
    (the caller must then treat any registry rows for it as stale - their
    Qdrant points no longer exist).

    Stage 5 replaced the improved pipeline's single unnamed dense vector with
    named vectors so dense and BM25-sparse search can be fused server-side
    (RRF). Qdrant can't add a vector name to an existing collection, and a
    collection built under the old schema only ever held dev/sample data (no
    durable production store exists yet), so it is dropped and recreated
    rather than requiring a manual migration step.
    """
    if client.collection_exists(name):
        vectors = client.get_collection(name).config.params.vectors
        if isinstance(vectors, dict) and DENSE_VECTOR_NAME in vectors:
            return False  # already the current hybrid schema
        log.warning("qdrant_collection_schema_outdated_recreating", collection=name)
        client.delete_collection(name)
        recreated = True
    else:
        recreated = False

    client.create_collection(
        collection_name=name,
        vectors_config={DENSE_VECTOR_NAME: VectorParams(size=dimensions, distance=Distance.COSINE)},
        sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)},
    )
    log.info("qdrant_hybrid_collection_created", collection=name, dimensions=dimensions)
    return recreated


def mark_superseded(client: QdrantClient, collection: str, doc_id: str) -> None:
    """Flag every point of an old document version as superseded rather than
    deleting it, so it stays available for audit/debugging but is excluded
    from retrieval by default (see `retrieval.search`)."""
    client.set_payload(
        collection_name=collection,
        payload={"superseded": True},
        points=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
    )


def delete_document_points(client: QdrantClient, collection: str, doc_id: str) -> None:
    """Remove all points for a doc_id - called before re-upserting so a re-ingest
    never leaves stale chunks from a previous version of the same document."""
    client.delete(
        collection_name=collection,
        points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
    )


def upsert_points(client: QdrantClient, collection: str, points: list[PointStruct]) -> None:
    if points:
        client.upsert(collection_name=collection, points=points, wait=True)


def check_qdrant() -> dict[str, Any]:
    s = get_settings()
    try:
        names = [c.name for c in make_client().get_collections().collections]
        return {
            "reachable": True,
            "url": s.qdrant_url,
            "collections": names,
            "target_collection": s.qdrant_collection,
            "target_collection_exists": s.qdrant_collection in names,
        }
    except Exception as exc:  # noqa: BLE001 - report, never crash the probe
        log.warning("qdrant_unreachable", url=s.qdrant_url, error=str(exc))
        return {"reachable": False, "url": s.qdrant_url, "error": str(exc)}
