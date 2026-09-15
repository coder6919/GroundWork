from __future__ import annotations

from types import SimpleNamespace

from app.config import Settings
from app.providers.rerank.base import RerankResult
from app.retrieval.search import search


class _FakeEmbedder:
    def __init__(self, vector):
        self.vector = vector
        self.calls = []

    def embed(self, texts, input_type):
        self.calls.append((tuple(texts), input_type))
        return [self.vector for _ in texts]


class _FakeQdrantClient:
    def __init__(self, points):
        self._points = points
        self.last_call = None

    def query_points(self, collection_name, **kwargs):
        self.last_call = {"collection_name": collection_name, **kwargs}
        return SimpleNamespace(points=self._points)


class _FakeReranker:
    def __init__(self, order):
        self.order = order  # list of RerankResult to return, in call order
        self.calls = []

    def rerank(self, query, documents, *, top_n):
        self.calls.append((query, tuple(documents), top_n))
        return self.order[: len(documents)]


def _point(**payload_overrides):
    payload = {
        "chunk_id": "c1",
        "doc_id": "d1",
        "source_filename": "a.md",
        "heading_path": "A > B",
        "chunk_text": "hello",
        "content_type": "prose",
    }
    payload.update(payload_overrides)
    return SimpleNamespace(id="pt-1", score=0.87, payload=payload)


def test_search_embeds_query_with_query_input_type():
    embedder = _FakeEmbedder([0.1, 0.2, 0.3])
    client = _FakeQdrantClient([_point()])
    settings = Settings(qdrant_collection="knowledge_base", rerank_top_n=5)

    search("what is the policy?", settings=settings, embedder=embedder, client=client)

    assert embedder.calls == [(("what is the policy?",), "query")]


def test_search_prefetches_both_dense_and_sparse_legs_and_fuses_with_rrf():
    embedder = _FakeEmbedder([0.1])
    client = _FakeQdrantClient([])
    settings = Settings(retrieve_top_k=20)

    search("q", settings=settings, embedder=embedder, client=client)

    prefetch = client.last_call["prefetch"]
    assert {p.using for p in prefetch} == {"dense", "sparse"}
    assert all(p.limit == 20 for p in prefetch)
    assert client.last_call["query"].fusion.value == "rrf"
    assert client.last_call["limit"] == 20


def test_search_without_reranker_truncates_to_rerank_top_n_by_default():
    embedder = _FakeEmbedder([0.1])
    points = [_point(chunk_id=f"c{i}") for i in range(10)]
    client = _FakeQdrantClient(points)
    settings = Settings(rerank_top_n=5)

    results = search("q", settings=settings, embedder=embedder, client=client)

    assert [r.chunk_id for r in results] == [f"c{i}" for i in range(5)]


def test_search_explicit_top_k_overrides_default_without_reranker():
    embedder = _FakeEmbedder([0.1])
    points = [_point(chunk_id=f"c{i}") for i in range(10)]
    client = _FakeQdrantClient(points)
    settings = Settings(rerank_top_n=5)

    results = search("q", settings=settings, embedder=embedder, top_k=2, client=client)

    assert [r.chunk_id for r in results] == ["c0", "c1"]


def test_search_maps_payload_and_score_into_retrieved_chunks():
    embedder = _FakeEmbedder([0.1])
    client = _FakeQdrantClient([_point(chunk_id="c1", doc_id="d1")])
    settings = Settings()

    results = search("q", settings=settings, embedder=embedder, client=client)

    assert len(results) == 1
    r = results[0]
    assert r.chunk_id == "c1"
    assert r.doc_id == "d1"
    assert r.source_filename == "a.md"
    assert r.heading_path == "A > B"
    assert r.chunk_text == "hello"
    assert r.score == 0.87
    assert r.rerank_score is None


def test_search_falls_back_to_point_id_when_chunk_id_missing_from_payload():
    embedder = _FakeEmbedder([0.1])
    point = SimpleNamespace(id="fallback-id", score=0.5, payload={})
    client = _FakeQdrantClient([point])
    settings = Settings()

    results = search("q", settings=settings, embedder=embedder, client=client)

    assert results[0].chunk_id == "fallback-id"


def test_search_reranks_and_reorders_candidates_when_reranker_supplied():
    embedder = _FakeEmbedder([0.1])
    points = [_point(chunk_id="c0"), _point(chunk_id="c1"), _point(chunk_id="c2")]
    client = _FakeQdrantClient(points)
    reranker = _FakeReranker(
        [RerankResult(index=2, score=0.99), RerankResult(index=0, score=0.5)]
    )
    settings = Settings(rerank_top_n=5)

    results = search("q", settings=settings, embedder=embedder, reranker=reranker, client=client)

    assert [r.chunk_id for r in results] == ["c2", "c0"]
    assert [r.rerank_score for r in results] == [0.99, 0.5]
    assert reranker.calls[0][2] == 5  # top_n passed through


def test_search_skips_reranker_when_no_candidates():
    embedder = _FakeEmbedder([0.1])
    client = _FakeQdrantClient([])
    reranker = _FakeReranker([])
    settings = Settings()

    results = search("q", settings=settings, embedder=embedder, reranker=reranker, client=client)

    assert results == []
    assert reranker.calls == []


def test_search_excludes_superseded_chunks_by_default():
    embedder = _FakeEmbedder([0.1])
    client = _FakeQdrantClient([])
    settings = Settings()

    search("q", settings=settings, embedder=embedder, client=client)

    prefetch = client.last_call["prefetch"]
    conditions = prefetch[0].filter.must_not
    assert any(c.key == "superseded" and c.match.value is True for c in conditions)


def test_search_include_superseded_true_skips_the_exclusion_filter():
    embedder = _FakeEmbedder([0.1])
    client = _FakeQdrantClient([])
    settings = Settings()

    search("q", settings=settings, embedder=embedder, client=client, include_superseded=True)

    prefetch = client.last_call["prefetch"]
    assert prefetch[0].filter is None


def test_search_applies_source_filename_and_doc_version_filters():
    embedder = _FakeEmbedder([0.1])
    client = _FakeQdrantClient([])
    settings = Settings()

    search(
        "q",
        settings=settings,
        embedder=embedder,
        client=client,
        source_filename="policy.md",
        doc_version="v2",
    )

    conditions = client.last_call["prefetch"][0].filter.must
    by_key = {c.key: c.match.value for c in conditions}
    assert by_key == {"source_filename": "policy.md", "doc_version": "v2"}


def test_search_applies_ingested_date_range_filter():
    embedder = _FakeEmbedder([0.1])
    client = _FakeQdrantClient([])
    settings = Settings()

    search(
        "q",
        settings=settings,
        embedder=embedder,
        client=client,
        ingested_after="2026-01-01T00:00:00+00:00",
        ingested_before="2026-06-01T00:00:00+00:00",
    )

    conditions = client.last_call["prefetch"][0].filter.must
    date_condition = next(c for c in conditions if c.key == "ingested_at")
    assert date_condition.range.gte.isoformat() == "2026-01-01T00:00:00+00:00"
    assert date_condition.range.lte.isoformat() == "2026-06-01T00:00:00+00:00"
