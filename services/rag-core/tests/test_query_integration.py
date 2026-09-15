"""End-to-end retrieval + generation against real Voyage, Qdrant, and Claude.

Deliberately a single test, deliberately cheap: reuses whatever is already in
the `knowledge_base` collection (populated by Stage 1's sample-corpus ingest -
no re-ingestion here, no extra embedding spend) and asks one short question
with top_k=3. Generation itself is cost-minimized in app/generation/claude.py
(thinking disabled, effort "low", max_tokens capped).

Stage 5 note: `knowledge_base` switched from a single unnamed vector to named
dense+sparse vectors for hybrid retrieval - `ensure_hybrid_collection` drops
and recreates a collection still on the old schema the first time anything
ingests into it. If this test fails with "expected the Stage 1 sample corpus
to already be ingested", re-run `POST /internal/ingest` against
`corpus/sample/` once first.
"""

from __future__ import annotations

import os

import pytest

from app.config import get_settings
from app.generation.claude import generate_answer
from app.providers import get_embedding_provider
from app.retrieval.search import search

pytestmark = pytest.mark.integration
_RUN = os.getenv("RUN_INTEGRATION") == "1"
_REASON = "set RUN_INTEGRATION=1 with Qdrant/VOYAGE_API_KEY/ANTHROPIC_API_KEY available"


@pytest.mark.skipif(not _RUN, reason=_REASON)
def test_query_against_ingested_sample_corpus_returns_a_cited_answer():
    settings = get_settings()
    embedder = get_embedding_provider(settings)

    chunks = search(
        "How long does an international refund take?",
        settings=settings,
        embedder=embedder,
        top_k=3,
    )
    assert chunks, "expected the Stage 1 sample corpus to already be ingested"

    answer = generate_answer(
        "How long does an international refund take?", chunks, settings=settings
    )

    assert answer.text.strip()
    assert answer.stop_reason in ("end_turn", "max_tokens")
    # At least one citation should trace back to the refund policy doc.
    assert any(c.source_filename == "refund-policy.md" for c in answer.citations)
