from __future__ import annotations

from app.retrieval.sparse import embed_documents, embed_query


def test_embed_documents_returns_one_sparse_vector_per_text():
    results = embed_documents(["refund policy for international orders", "vacation accrual rules"])

    assert len(results) == 2
    for r in results:
        assert r.indices
        assert len(r.indices) == len(r.values)


def test_embed_documents_empty_list_returns_empty():
    assert embed_documents([]) == []


def test_embed_query_returns_a_single_sparse_vector():
    result = embed_query("what is the refund policy")

    assert result.indices
    assert len(result.indices) == len(result.values)


def test_matching_terms_produce_overlapping_indices():
    [doc_vector] = embed_documents(["the refund policy covers international orders"])
    query_vector = embed_query("refund policy")

    assert set(query_vector.indices) & set(doc_vector.indices)
