"""Embedding provider interface. Concrete providers live alongside this and are
selected by `EMBEDDING_PROVIDER` - swapping providers never touches the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

InputType = Literal["document", "query"]


class EmbeddingProvider(ABC):
    model: str
    dimensions: int

    @abstractmethod
    def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        """Embed a batch of texts. `input_type` lets providers that support
        asymmetric document/query encoding (e.g. Voyage) use the right mode."""
        ...
