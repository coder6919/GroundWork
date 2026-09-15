from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.providers.rerank.cohere as cohere_provider
from app.providers.rerank.cohere import CohereReranker


class _FakeCohereClientV2:
    def __init__(self, api_key):
        self.api_key = api_key
        self.calls = []

    def rerank(self, *, model, query, documents, top_n):
        self.calls.append({"model": model, "query": query, "documents": documents, "top_n": top_n})
        return SimpleNamespace(
            results=[
                SimpleNamespace(index=1, relevance_score=0.95),
                SimpleNamespace(index=0, relevance_score=0.42),
            ]
        )


def test_missing_api_key_raises():
    with pytest.raises(ValueError, match="COHERE_API_KEY"):
        CohereReranker(api_key="", model="rerank-v3.5")


def test_rerank_returns_results_ordered_as_the_provider_returned_them(monkeypatch):
    fake_client = _FakeCohereClientV2(api_key="k")
    monkeypatch.setattr(cohere_provider.cohere, "ClientV2", lambda api_key: fake_client)

    reranker = CohereReranker(api_key="k", model="rerank-v3.5")
    results = reranker.rerank("refund policy", ["doc a", "doc b"], top_n=2)

    assert [(r.index, r.score) for r in results] == [(1, 0.95), (0, 0.42)]
    assert fake_client.calls[0]["model"] == "rerank-v3.5"
    assert fake_client.calls[0]["top_n"] == 2


def test_rerank_caps_top_n_to_the_candidate_count(monkeypatch):
    fake_client = _FakeCohereClientV2(api_key="k")
    monkeypatch.setattr(cohere_provider.cohere, "ClientV2", lambda api_key: fake_client)

    reranker = CohereReranker(api_key="k", model="rerank-v3.5")
    reranker.rerank("q", ["only one doc"], top_n=20)

    assert fake_client.calls[0]["top_n"] == 1


def test_rerank_empty_documents_short_circuits_without_calling_the_api(monkeypatch):
    fake_client = _FakeCohereClientV2(api_key="k")
    monkeypatch.setattr(cohere_provider.cohere, "ClientV2", lambda api_key: fake_client)

    reranker = CohereReranker(api_key="k", model="rerank-v3.5")
    assert reranker.rerank("q", [], top_n=5) == []
    assert fake_client.calls == []


def test_rerank_logs_an_external_call(monkeypatch):
    fake_client = _FakeCohereClientV2(api_key="k")
    monkeypatch.setattr(cohere_provider.cohere, "ClientV2", lambda api_key: fake_client)
    logged = []
    monkeypatch.setattr(
        cohere_provider, "log_external_call", lambda *a, **k: logged.append((a, k))
    )

    reranker = CohereReranker(api_key="k", model="rerank-v3.5")
    reranker.rerank("q", ["a", "b"], top_n=2)

    assert logged and logged[0][0][0] == "cohere"
