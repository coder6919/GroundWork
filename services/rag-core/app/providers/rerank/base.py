"""Rerank provider interface. Concrete providers live alongside this and are
selected by `RERANK_PROVIDER` - swapping providers never touches retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RerankResult:
    index: int  # position of this document in the input list passed to rerank()
    score: float


class RerankProvider(ABC):
    model: str

    @abstractmethod
    def rerank(self, query: str, documents: list[str], *, top_n: int) -> list[RerankResult]:
        """Score `documents` against `query`, returning up to `top_n` results
        ordered best-first. `index` refers back into the input `documents` list."""
        ...
