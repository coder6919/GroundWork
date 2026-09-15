"""Unit tests for VoyageEmbeddingProvider's token-budget batching (added after
a live rate-limit incident: a single ~9.5K-token document embedded in one call
perpetually failed Voyage's free-tier 10K TPM cap, no matter how long we
waited, because the call itself - not a transient rate-limit window - was too
big). Mocks voyageai.Client entirely - no network."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from voyageai.error import RateLimitError

from app.providers.voyage import VoyageEmbeddingProvider, _batch_by_tokens, _count_tokens


@dataclass
class FakeEmbedResult:
    embeddings: list[list[float]]
    total_tokens: int


@dataclass
class FakeVoyageClient:
    calls: list[list[str]] = field(default_factory=list)
    fail_first_n_calls: int = 0

    def embed(self, texts, **kwargs):
        self.calls.append(list(texts))
        if len(self.calls) <= self.fail_first_n_calls:
            raise RateLimitError("rate limited")
        return FakeEmbedResult(embeddings=[[0.0] for _ in texts], total_tokens=sum(_count_tokens(t) for t in texts))


@pytest.fixture
def provider(monkeypatch):
    p = VoyageEmbeddingProvider(api_key="test-key", model="voyage-4-lite", dimensions=1024)
    fake = FakeVoyageClient()
    monkeypatch.setattr(p, "_client", fake)
    monkeypatch.setattr("app.providers.voyage.time.sleep", lambda _seconds: None)
    return p, fake


def test_batch_by_tokens_keeps_small_texts_in_one_batch():
    texts = ["short text"] * 5
    batches = _batch_by_tokens(texts, max_batch_tokens=8000)
    assert batches == [texts]


def test_batch_by_tokens_splits_when_budget_exceeded():
    long_text = "word " * 4000  # comfortably over half the budget each
    texts = [long_text, long_text, long_text]
    batches = _batch_by_tokens(texts, max_batch_tokens=8000)
    assert len(batches) > 1
    assert sum(len(b) for b in batches) == 3


def test_batch_by_tokens_never_splits_a_single_oversized_text():
    huge_text = "word " * 20000
    batches = _batch_by_tokens([huge_text], max_batch_tokens=100)
    assert batches == [[huge_text]]


def test_embed_returns_empty_list_for_no_texts(provider):
    p, fake = provider
    assert p.embed([], input_type="document") == []
    assert fake.calls == []


def test_embed_small_input_makes_exactly_one_call(provider):
    p, fake = provider
    texts = ["a", "b", "c"]
    result = p.embed(texts, input_type="document")
    assert len(fake.calls) == 1
    assert fake.calls[0] == texts
    assert len(result) == 3


def test_embed_large_input_splits_into_multiple_calls_and_preserves_order(monkeypatch):
    p = VoyageEmbeddingProvider(api_key="test-key", model="voyage-4-lite", dimensions=1024, max_batch_tokens=50)
    fake = FakeVoyageClient()
    monkeypatch.setattr(p, "_client", fake)
    sleep_calls = []
    monkeypatch.setattr("app.providers.voyage.time.sleep", lambda s: sleep_calls.append(s))

    texts = ["word " * 30 for _ in range(5)]  # each ~30-40 tokens, forces multiple batches at budget=50
    p.embed(texts, input_type="document")

    assert len(fake.calls) > 1
    # every text made it into exactly one call, in original order
    flattened = [t for call in fake.calls for t in call]
    assert flattened == texts
    # paced between batches, not before the first
    assert len(sleep_calls) == len(fake.calls) - 1


def test_embed_requires_api_key():
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        VoyageEmbeddingProvider(api_key="", model="voyage-4-lite", dimensions=1024)


def test_embed_retries_on_rate_limit_and_eventually_succeeds(monkeypatch):
    p = VoyageEmbeddingProvider(
        api_key="test-key", model="voyage-4-lite", dimensions=1024, rate_limit_retries=3, rate_limit_backoff_seconds=1
    )
    fake = FakeVoyageClient(fail_first_n_calls=2)
    monkeypatch.setattr(p, "_client", fake)
    backoff_delays = []
    monkeypatch.setattr("app.providers.voyage.time.sleep", lambda s: backoff_delays.append(s))

    result = p.embed(["a", "b"], input_type="document")

    assert len(fake.calls) == 3  # 2 failures + 1 success, same batch resent each time
    assert all(call == ["a", "b"] for call in fake.calls)
    assert len(result) == 2
    # linear backoff: 30s * 1, then 30s * 2 (using the real default scale via seconds param)
    assert backoff_delays == [1.0, 2.0]


def test_embed_raises_after_exhausting_rate_limit_retries(monkeypatch):
    p = VoyageEmbeddingProvider(
        api_key="test-key", model="voyage-4-lite", dimensions=1024, rate_limit_retries=2, rate_limit_backoff_seconds=1
    )
    fake = FakeVoyageClient(fail_first_n_calls=10)  # never succeeds
    monkeypatch.setattr(p, "_client", fake)
    monkeypatch.setattr("app.providers.voyage.time.sleep", lambda _s: None)

    with pytest.raises(RateLimitError):
        p.embed(["a"], input_type="document")

    assert len(fake.calls) == 3  # initial attempt + 2 retries, then gives up
