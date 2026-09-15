"""Cohere Rerank provider (Stage 5). Paid API - every call is logged via
`log_external_call` per the cost-control standing rule."""

from __future__ import annotations

import cohere

from ...logging import get_logger, log_external_call
from .base import RerankProvider, RerankResult

log = get_logger("providers.rerank.cohere")


class CohereReranker(RerankProvider):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("COHERE_API_KEY is required to use RERANK_PROVIDER=cohere")
        self.model = model
        self._client = cohere.ClientV2(api_key=api_key)

    def rerank(self, query: str, documents: list[str], *, top_n: int) -> list[RerankResult]:
        if not documents:
            return []

        response = self._client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)),
        )

        log_external_call(
            "cohere",
            "rerank",
            model=self.model,
            candidate_count=len(documents),
            top_n=top_n,
        )

        return [RerankResult(index=r.index, score=r.relevance_score) for r in response.results]
