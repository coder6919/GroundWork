"""Local BM25 sparse embeddings for hybrid retrieval (Stage 5).

`fastembed`'s `Qdrant/bm25` model computes term-frequency-saturated sparse
vectors entirely locally (tokenization + BM25's TF component) - no neural
weights, no external API call, no rate limits. IDF weighting is applied by
Qdrant itself at query time (`Modifier.IDF` on the collection's sparse vector
config), which is why document- and query-side vectors look different here
(`embed` vs `query_embed`) - this is fastembed's documented pairing for BM25
with Qdrant, not an inconsistency.

The model asset is a tiny (~KB) tokenizer/stopword file fetched from Hugging
Face on first use; the Dockerfile pre-warms it at build time so the running
container never needs network for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from fastembed import SparseTextEmbedding

MODEL_NAME = "Qdrant/bm25"


@dataclass
class SparseVectorData:
    indices: list[int]
    values: list[float]


@lru_cache
def _model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=MODEL_NAME)


def embed_documents(texts: list[str]) -> list[SparseVectorData]:
    """Index-time encoding for a batch of chunk texts."""
    if not texts:
        return []
    return [
        SparseVectorData(indices=e.indices.tolist(), values=e.values.tolist())
        for e in _model().embed(texts)
    ]


def embed_query(text: str) -> SparseVectorData:
    """Query-time encoding - asymmetric to `embed_documents` per fastembed's BM25 design."""
    embedding = next(iter(_model().query_embed([text])))
    return SparseVectorData(indices=embedding.indices.tolist(), values=embedding.values.tolist())
