"""Naive retrieval: same vector search mechanics as app/retrieval, but against
the naive collection, returning plain chunk strings - the naive payload has no
heading path or content-type metadata to return."""

from __future__ import annotations

from qdrant_client import QdrantClient

from ..clients.qdrant import make_client as make_qdrant_client
from ..config import Settings
from ..providers.base import EmbeddingProvider


def naive_search(
    query: str,
    *,
    settings: Settings,
    embedder: EmbeddingProvider,
    top_k: int = 5,
    client: QdrantClient | None = None,
) -> list[str]:
    vector = embedder.embed([query], input_type="query")[0]
    client = client or make_qdrant_client(timeout=15.0)
    result = client.query_points(
        settings.qdrant_collection_naive, query=vector, limit=top_k, with_payload=True
    )
    return [(point.payload or {}).get("chunk_text", "") for point in result.points]
