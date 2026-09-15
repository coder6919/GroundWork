"""Rerank provider factory - the only place that reads `RERANK_PROVIDER` and
picks an implementation, mirroring `providers.get_embedding_provider`."""

from __future__ import annotations

from ...config import Settings
from .base import RerankProvider


def get_rerank_provider(settings: Settings) -> RerankProvider:
    if settings.rerank_provider == "cohere":
        from .cohere import CohereReranker

        return CohereReranker(api_key=settings.cohere_api_key, model=settings.rerank_model)
    raise ValueError(f"Unknown RERANK_PROVIDER: {settings.rerank_provider!r}")
