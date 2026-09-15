"""Provider factory - the only place that reads `EMBEDDING_PROVIDER` and picks
an implementation. Adding a provider means adding one branch here."""

from __future__ import annotations

from ..config import Settings
from .base import EmbeddingProvider


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "voyage":
        from .voyage import VoyageEmbeddingProvider

        return VoyageEmbeddingProvider(
            api_key=settings.voyage_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            max_batch_tokens=settings.voyage_max_batch_tokens,
            batch_delay_seconds=settings.voyage_batch_delay_seconds,
        )
    if settings.embedding_provider == "openai":
        raise NotImplementedError("EMBEDDING_PROVIDER=openai is not implemented yet")
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider!r}")
