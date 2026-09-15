"""Voyage AI embeddings. The only place voyageai's SDK is imported."""

from __future__ import annotations

import time

import tiktoken
import voyageai
from voyageai.error import RateLimitError

from ..logging import get_logger, log_external_call
from .base import EmbeddingProvider, InputType

log = get_logger("provider.voyage")

# Voyage doesn't publish its own tokenizer; cl100k_base is a reasonable proxy
# for batch-sizing purposes (we only need to stay safely under the TPM cap,
# not match Voyage's billed count exactly) - confirmed within ~1% of Voyage's
# own reported total_tokens on real corpus content, live.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


class VoyageEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        api_key: str,
        model: str,
        dimensions: int,
        *,
        max_batch_tokens: int = 8000,
        batch_delay_seconds: float = 21.0,
        rate_limit_retries: int = 3,
        rate_limit_backoff_seconds: float = 30.0,
    ):
        if not api_key:
            raise ValueError("VOYAGE_API_KEY is required to use the voyage embedding provider")
        self.model = model
        self.dimensions = dimensions
        self.max_batch_tokens = max_batch_tokens
        self.batch_delay_seconds = batch_delay_seconds
        self.rate_limit_retries = rate_limit_retries
        self.rate_limit_backoff_seconds = rate_limit_backoff_seconds
        # The SDK's own max_retries backs off in (sub-)second increments, which
        # isn't long enough to clear a real 3 RPM / 10K TPM window - rate-limit
        # retries are handled explicitly below instead, with real backoff.
        self._client = voyageai.Client(api_key=api_key, max_retries=0, timeout=30.0)

    def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for i, batch in enumerate(_batch_by_tokens(texts, self.max_batch_tokens)):
            if i > 0:
                # A single document's chunks can legitimately need more than one
                # call under the free tier's 10K TPM cap - pace subsequent calls
                # to also respect its 3 RPM cap. Never sleeps on the common path
                # (single-batch documents, and every query embed).
                time.sleep(self.batch_delay_seconds)
            result = self._embed_batch_with_retry(batch, input_type)
            embeddings.extend(result.embeddings)
        return embeddings

    def _embed_batch_with_retry(self, batch: list[str], input_type: InputType):
        attempt = 0
        while True:
            try:
                result = self._client.embed(
                    batch,
                    model=self.model,
                    input_type=input_type,
                    output_dimension=self.dimensions,
                )
                log_external_call(
                    "voyage",
                    "embed",
                    model=self.model,
                    input_type=input_type,
                    batch_size=len(batch),
                    total_tokens=result.total_tokens,
                    retries=attempt,
                )
                return result
            except RateLimitError:
                if attempt >= self.rate_limit_retries:
                    raise
                # Free-tier RPM/TPM windows are real minute-scale windows, not
                # sub-second ones - a short exponential backoff won't clear
                # them. Linear backoff at a rate-limit-scale delay instead.
                delay = self.rate_limit_backoff_seconds * (attempt + 1)
                log.warning(
                    "voyage_rate_limited_retrying",
                    attempt=attempt + 1,
                    max_retries=self.rate_limit_retries,
                    delay_seconds=delay,
                    batch_size=len(batch),
                )
                time.sleep(delay)
                attempt += 1


def _batch_by_tokens(texts: list[str], max_batch_tokens: int) -> list[list[str]]:
    """Greedily groups texts so each batch's estimated token total stays under
    `max_batch_tokens`. A single text that alone exceeds the budget still gets
    its own batch (never split mid-text) - rare, but correct: an oversized
    atomic chunk (e.g. a huge table) is a known, accepted edge case elsewhere
    in ingestion (see chunking.py), not something to break here."""
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for text in texts:
        text_tokens = _count_tokens(text)
        if current and current_tokens + text_tokens > max_batch_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(text)
        current_tokens += text_tokens
    if current:
        batches.append(current)
    return batches
