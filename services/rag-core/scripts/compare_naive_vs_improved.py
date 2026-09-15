"""Run one question through both the naive baseline and the improved pipeline
and print both answers side by side. This is what Stage 11's README before/
after section is pulled from.

Usage (from services/rag-core/, inside the rag-core container or venv - run as
a module so `app` is importable):
    uv run --frozen python -m scripts.compare_naive_vs_improved "your question"
    uv run --frozen python -m scripts.compare_naive_vs_improved "your question" --corpus /corpus/sample --reingest

Ingestion into both collections is skipped if they already have points, unless
--reingest is passed - re-running this script doesn't re-embed by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.clients.qdrant import make_client
from app.config import get_settings
from app.db.registry import SQLiteDocumentRegistry
from app.generation.claude import generate_answer, make_anthropic_client
from app.ingestion.pipeline import ingest_path
from app.naive.generation import naive_generate
from app.naive.pipeline import naive_ingest_path
from app.naive.search import naive_search
from app.providers import get_embedding_provider
from app.providers.rerank import get_rerank_provider
from app.retrieval.search import search


def _collection_has_points(client, collection: str) -> bool:
    if not client.collection_exists(collection):
        return False
    return client.count(collection, exact=True).count > 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--corpus", default="/corpus/sample")
    parser.add_argument("--reingest", action="store_true", help="Force re-ingest both collections")
    args = parser.parse_args()

    settings = get_settings()
    embedder = get_embedding_provider(settings)
    anthropic_client = make_anthropic_client(settings.anthropic_api_key)
    qdrant = make_client(timeout=30.0)
    corpus_path = Path(args.corpus)

    if args.reingest or not _collection_has_points(qdrant, settings.qdrant_collection):
        print(f"Ingesting {corpus_path} into '{settings.qdrant_collection}' (improved pipeline)...")
        registry = SQLiteDocumentRegistry(settings.sqlite_path)
        ingest_path(corpus_path, settings=settings, registry=registry, embedder=embedder)

    if args.reingest or not _collection_has_points(qdrant, settings.qdrant_collection_naive):
        print(f"Ingesting {corpus_path} into '{settings.qdrant_collection_naive}' (naive pipeline)...")
        naive_ingest_path(corpus_path, settings=settings, embedder=embedder)

    print(f"\n=== Question ===\n{args.question}\n")

    naive_chunks = naive_search(args.question, settings=settings, embedder=embedder, client=qdrant)
    naive_answer = naive_generate(
        args.question, naive_chunks, settings=settings, client=anthropic_client
    )
    print("=== Naive RAG (fixed chunks, no context-only constraint, no citations) ===")
    print(naive_answer)

    reranker = None
    if settings.cohere_api_key:
        reranker = get_rerank_provider(settings)
    else:
        print("(COHERE_API_KEY not set - skipping rerank, using fused hybrid order only)\n")

    improved_chunks = search(
        args.question, settings=settings, embedder=embedder, reranker=reranker, client=qdrant
    )
    improved_answer = generate_answer(
        args.question, improved_chunks, settings=settings, client=anthropic_client
    )
    print("\n=== Improved RAG (structure-aware chunking, context-only + refusal prompt, native citations) ===")
    print(improved_answer.text)
    if improved_answer.citations:
        print("\nCitations:")
        for c in improved_answer.citations:
            print(f'  - {c.source_filename} ({c.heading_path}): "{c.cited_text}"')
    else:
        print("\n(no citations)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
