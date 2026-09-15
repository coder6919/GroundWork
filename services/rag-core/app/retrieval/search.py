"""Hybrid retrieval (Stage 5): dense (Voyage) + sparse (local BM25) search,
fused server-side via Qdrant's RRF, then reranked by Cohere down to the final
top-N. Optional metadata filters and supersede exclusion apply to both legs
of the fusion before reranking ever sees the candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint
from qdrant_client.models import (
    DatetimeRange,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector,
)

from ..clients.qdrant import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from ..clients.qdrant import make_client as make_qdrant_client
from ..config import Settings
from ..providers.base import EmbeddingProvider
from ..providers.rerank.base import RerankProvider
from . import sparse


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    source_filename: str
    heading_path: str
    chunk_text: str
    content_type: str
    score: float  # fused hybrid retrieval score (pre-rerank)
    rerank_score: float | None = None  # set once a reranker has scored this chunk


def search(
    query: str,
    *,
    settings: Settings,
    embedder: EmbeddingProvider,
    reranker: RerankProvider | None = None,
    top_k: int | None = None,
    client: QdrantClient | None = None,
    source_filename: str | None = None,
    doc_version: str | None = None,
    ingested_after: str | None = None,
    ingested_before: str | None = None,
    include_superseded: bool = False,
) -> list[RetrievedChunk]:
    """Embed the query (dense + sparse), fetch a `retrieve_top_k` candidate pool
    via Qdrant's server-side RRF fusion, then rerank down to `top_k` (default
    `rerank_top_n`) with Cohere. Reranking is skipped - the fused order is kept
    as-is - when no reranker is supplied, so callers/tests that don't need the
    paid call can omit it."""
    dense_vector = embedder.embed([query], input_type="query")[0]
    sparse_vector = sparse.embed_query(query)
    client = client or make_qdrant_client(timeout=15.0)
    query_filter = _build_filter(
        source_filename=source_filename,
        doc_version=doc_version,
        ingested_after=ingested_after,
        ingested_before=ingested_before,
        include_superseded=include_superseded,
    )

    result = client.query_points(
        settings.qdrant_collection,
        prefetch=[
            Prefetch(
                query=dense_vector,
                using=DENSE_VECTOR_NAME,
                filter=query_filter,
                limit=settings.retrieve_top_k,
            ),
            Prefetch(
                query=SparseVector(indices=sparse_vector.indices, values=sparse_vector.values),
                using=SPARSE_VECTOR_NAME,
                filter=query_filter,
                limit=settings.retrieve_top_k,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=settings.retrieve_top_k,
        with_payload=True,
    )
    candidates = [_to_chunk(point) for point in result.points]

    final_top_n = top_k or settings.rerank_top_n
    if not reranker or not candidates:
        return candidates[:final_top_n]

    ranked = reranker.rerank(query, [c.chunk_text for c in candidates], top_n=final_top_n)
    return [
        _with_rerank_score(candidates[r.index], r.score)
        for r in ranked
        if 0 <= r.index < len(candidates)
    ]


def _with_rerank_score(chunk: RetrievedChunk, score: float) -> RetrievedChunk:
    chunk.rerank_score = score
    return chunk


def _build_filter(
    *,
    source_filename: str | None,
    doc_version: str | None,
    ingested_after: str | None,
    ingested_before: str | None,
    include_superseded: bool,
) -> Filter | None:
    must: list[FieldCondition] = []
    must_not: list[FieldCondition] = []

    if source_filename:
        must.append(FieldCondition(key="source_filename", match=MatchValue(value=source_filename)))
    if doc_version:
        must.append(FieldCondition(key="doc_version", match=MatchValue(value=doc_version)))
    if ingested_after or ingested_before:
        must.append(
            FieldCondition(
                key="ingested_at",
                range=DatetimeRange(
                    gte=datetime.fromisoformat(ingested_after) if ingested_after else None,
                    lte=datetime.fromisoformat(ingested_before) if ingested_before else None,
                ),
            )
        )
    if not include_superseded:
        must_not.append(FieldCondition(key="superseded", match=MatchValue(value=True)))

    if not must and not must_not:
        return None
    return Filter(must=must or None, must_not=must_not or None)


def _to_chunk(point: ScoredPoint) -> RetrievedChunk:
    payload = point.payload or {}
    return RetrievedChunk(
        chunk_id=payload.get("chunk_id", str(point.id)),
        doc_id=payload.get("doc_id", ""),
        source_filename=payload.get("source_filename", ""),
        heading_path=payload.get("heading_path", ""),
        chunk_text=payload.get("chunk_text", ""),
        content_type=payload.get("content_type", "prose"),
        score=point.score,
    )
