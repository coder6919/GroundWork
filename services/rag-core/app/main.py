"""FastAPI application for the RAG core service.

Surface:
  GET  /health          - unauthenticated liveness
  GET  /ready            - unauthenticated readiness (reports Qdrant connectivity)
  GET  /internal/ping    - shared-secret protected; proves the api -> rag-core boundary
  POST /internal/ingest  - shared-secret protected; Stage 1 ingestion (text/Markdown)
  POST /internal/query   - shared-secret protected; hybrid retrieval + Cohere
                           rerank (Stage 5) + cited generation (Stage 2) +
                           refusal/confidence gating (Stage 6) +
                           conversation-aware multi-turn (Stage 7, optional
                           `session_id`)
  POST /internal/query/stream  - shared-secret protected; same pipeline as
                           /internal/query, streamed as SSE (Stage 9) - text
                           arrives token-by-token, citations in one final
                           "done" event (no documented per-token citations
                           delta on the streaming API)
  GET  /internal/sources/{doc_id} - shared-secret protected; registry lookup
                           for citation click-through (Stage 9)
  POST /internal/ingest/upload - shared-secret protected; accepts a raw file
                           upload rather than a server-side path (Stage 9) -
                           the only ingest route the public api layer is
                           allowed to call; /internal/ingest's path-based
                           contract stays internal/dev-only
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Literal

import anthropic
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .clients.qdrant import check_qdrant
from .config import Settings, get_settings
from .db.models import DocumentStatus, Turn
from .db.registry import DocumentRegistry, SQLiteDocumentRegistry
from .db.sessions import SessionStore, SQLiteSessionStore
from .generation.claude import generate_answer, generate_answer_stream, make_anthropic_client
from .generation.query_rewrite import rewrite_query
from .ingestion.loaders import SUPPORTED_EXTENSIONS
from .ingestion.pipeline import ingest_path
from .logging import configure_logging, get_logger
from .naive.generation import naive_generate
from .naive.search import naive_search
from .providers import get_embedding_provider
from .providers.base import EmbeddingProvider
from .providers.rerank import get_rerank_provider
from .providers.rerank.base import RerankProvider
from .retrieval.search import search
from .security import require_internal_secret

settings = get_settings()
configure_logging(level=settings.log_level, json_logs=not settings.is_development)
log = get_logger()


@lru_cache
def get_registry() -> DocumentRegistry:
    return SQLiteDocumentRegistry(get_settings().sqlite_path)


@lru_cache
def get_session_store() -> SessionStore:
    return SQLiteSessionStore(get_settings().sqlite_path)


@lru_cache
def get_embedder() -> EmbeddingProvider:
    """Built lazily, only when ingestion actually runs - importing/starting this
    service never requires an embedding provider key (see missing_ai_keys)."""
    return get_embedding_provider(get_settings())


@lru_cache
def get_anthropic_client() -> anthropic.Anthropic:
    """Built lazily, only when a query actually runs."""
    return make_anthropic_client(get_settings().anthropic_api_key)


@lru_cache
def get_reranker() -> RerankProvider:
    """Built lazily, only when a query actually runs - importing/starting this
    service never requires a rerank provider key (see missing_ai_keys)."""
    return get_rerank_provider(get_settings())


@asynccontextmanager
async def lifespan(_app: FastAPI):
    missing = settings.missing_ai_keys()
    if missing:
        log.warning(
            "ai_provider_keys_missing",
            missing=missing,
            note="Keys are needed only when a route that uses them is actually called.",
        )
    log.info(
        "rag_core_starting",
        version=__version__,
        env=settings.env,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        rerank_provider=settings.rerank_provider,
        qdrant_url=settings.qdrant_url,
        ingest_enabled=settings.ingest_enabled,
    )
    yield
    log.info("rag_core_stopping")


app = FastAPI(title="RAG Core", version=__version__, lifespan=lifespan)


@app.get("/health", tags=["probes"])
async def health() -> dict:
    """Unauthenticated liveness probe (used by the container healthcheck)."""
    return {"status": "ok", "service": "rag-core", "version": __version__}


@app.get("/ready", tags=["probes"])
async def ready():
    """Unauthenticated readiness probe. Reports Qdrant connectivity; creates nothing."""
    qdrant = check_qdrant()
    if not qdrant["reachable"]:
        return JSONResponse(status_code=503, content={"status": "degraded", "qdrant": qdrant})
    return {"status": "ok", "qdrant": qdrant}


@app.get("/internal/ping", tags=["internal"], dependencies=[Depends(require_internal_secret)])
async def internal_ping() -> dict:
    """Shared-secret protected. Confirms the api -> rag-core trust boundary end to end."""
    return {"status": "ok", "authenticated": True, "service": "rag-core"}


class IngestRequest(BaseModel):
    path: str
    # Stage 5 versioned-doc supersedes: {new_relative_path: old_relative_path}.
    supersedes: dict[str, str] | None = None
    # Stage 5 metadata filtering: {relative_path: version_label}.
    versions: dict[str, str] | None = None


class FileResultOut(BaseModel):
    doc_id: str
    filename: str
    status: str
    chunk_count: int = 0
    failure_reason: str | None = None
    skipped_duplicate: bool = False


class IngestResponse(BaseModel):
    root: str
    processed: int
    ready: int
    skipped_duplicate: int
    failed: int
    unprocessable: int
    results: list[FileResultOut]


def _ingest_response(root: str, results: list) -> IngestResponse:
    return IngestResponse(
        root=root,
        processed=len(results),
        ready=sum(1 for r in results if r.status == DocumentStatus.READY and not r.skipped_duplicate),
        skipped_duplicate=sum(1 for r in results if r.skipped_duplicate),
        failed=sum(1 for r in results if r.status == DocumentStatus.FAILED),
        unprocessable=sum(1 for r in results if r.status == DocumentStatus.UNPROCESSABLE),
        results=[
            FileResultOut(
                doc_id=r.doc_id,
                filename=r.filename,
                status=r.status.value,
                chunk_count=r.chunk_count,
                failure_reason=r.failure_reason,
                skipped_duplicate=r.skipped_duplicate,
            )
            for r in results
        ],
    )


@app.post("/internal/ingest", tags=["internal"], dependencies=[Depends(require_internal_secret)])
async def internal_ingest(body: IngestRequest) -> IngestResponse:
    """Internal/dev-only: accepts a server-side path. Never call this from a
    public-facing layer - see /internal/ingest/upload for the public-safe
    equivalent (Stage 9)."""
    s = get_settings()
    if not s.ingest_enabled:
        raise HTTPException(403, "ingestion is disabled (INGEST_ENABLED=false)")
    root = Path(body.path)
    if not root.exists():
        raise HTTPException(404, f"path not found: {body.path}")

    # ingest_path is synchronous (file I/O + blocking HTTP to Voyage/Qdrant) - run
    # it off the event loop so it never stalls concurrent requests like /health.
    results = await run_in_threadpool(
        ingest_path,
        root,
        settings=s,
        registry=get_registry(),
        embedder=get_embedder(),
        supersedes=body.supersedes,
        versions=body.versions,
    )
    return _ingest_response(str(root), results)


def _safe_upload_filename(filename: str | None) -> str:
    """Basename only - never trust a caller-supplied path. A public upload
    must not be able to write outside the upload directory."""
    name = Path(filename or "").name
    if not name or name in (".", ".."):
        raise HTTPException(400, "missing or invalid filename")
    return name


@app.post("/internal/ingest/upload", tags=["internal"], dependencies=[Depends(require_internal_secret)])
async def internal_ingest_upload(file: UploadFile = File(...)) -> IngestResponse:  # noqa: B008 - FastAPI's DI pattern
    """The only ingest route safe to expose behind a public API layer: takes
    file bytes, not a server-side path (DIRECTION.md - hosted ingestion must
    never accept an unrestricted, caller-controlled path)."""
    s = get_settings()
    if not s.ingest_enabled:
        raise HTTPException(403, "ingestion is disabled (INGEST_ENABLED=false)")

    filename = _safe_upload_filename(file.filename)
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, f"unsupported file type: {filename}")

    upload_dir = Path(s.storage_dir) / "_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{uuid.uuid4().hex}_{filename}"

    # Streamed to disk in chunks - never buffers the whole upload in memory.
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            out.write(chunk)

    try:
        results = await run_in_threadpool(
            ingest_path, dest, settings=s, registry=get_registry(), embedder=get_embedder()
        )
    finally:
        dest.unlink(missing_ok=True)  # the registry's storage_dir copy is the durable one

    return _ingest_response(filename, results)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = None
    # Stage 5 metadata filters - all optional, default behavior is unchanged.
    source_filename: str | None = None
    doc_version: str | None = None
    ingested_after: str | None = None  # ISO 8601
    ingested_before: str | None = None  # ISO 8601
    include_superseded: bool = False
    # Stage 7 multi-turn: a caller-generated id, created implicitly on first
    # use. Omit for the original stateless single-turn behavior.
    session_id: str | None = None
    # Stage 10 naive-vs-improved demo toggle: "naive" routes to app/naive's
    # fixed-chunk retrieval + uncited flat-context generation instead of the
    # hybrid+rerank+citations pipeline. Naive ignores metadata filters and
    # session/multi-turn (the naive baseline never had either).
    pipeline: Literal["improved", "naive"] = "improved"


class CitationOut(BaseModel):
    chunk_id: str
    doc_id: str
    source_filename: str
    heading_path: str
    cited_text: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    model: str
    stop_reason: str
    retrieved_count: int
    session_id: str | None = None
    # Set only when a session's history actually changed the query text -
    # None on a session's first turn or when rewriting made no difference.
    rewritten_query: str | None = None


async def _prepare_query(body: QueryRequest, s: Settings) -> tuple[str, str | None, list]:
    """Shared by the blocking and streaming query routes: conversation-aware
    rewrite (Stage 7) + hybrid retrieval (Stage 5), identical either way -
    only how the answer is delivered differs."""
    effective_query = body.query
    rewritten_query: str | None = None
    if body.session_id:
        history = await run_in_threadpool(get_session_store().get_turns, body.session_id)
        recent_history = history[-s.conversation_history_turns :]
        effective_query = await run_in_threadpool(
            rewrite_query,
            recent_history,
            body.query,
            settings=s,
            client=get_anthropic_client(),
        )
        if effective_query != body.query:
            rewritten_query = effective_query

    # Synchronous (blocking HTTP to Voyage/Qdrant) - run off the event loop so
    # a slow query never stalls /health.
    chunks = await run_in_threadpool(
        search,
        effective_query,
        settings=s,
        embedder=get_embedder(),
        reranker=get_reranker(),
        top_k=body.top_k,
        source_filename=body.source_filename,
        doc_version=body.doc_version,
        ingested_after=body.ingested_after,
        ingested_before=body.ingested_before,
        include_superseded=body.include_superseded,
    )
    return effective_query, rewritten_query, chunks


async def _run_naive_query(body: QueryRequest, s: Settings) -> QueryResponse:
    """Stage 10's naive-vs-improved toggle, blocking path: no rewrite, no
    metadata filters, no citations - the naive baseline never had any of
    those (see app/naive/*)."""
    naive_chunks = await run_in_threadpool(
        naive_search,
        body.query,
        settings=s,
        embedder=get_embedder(),
        top_k=body.top_k or 5,
    )
    answer_text = (
        await run_in_threadpool(
            naive_generate, body.query, naive_chunks, settings=s, client=get_anthropic_client()
        )
        if naive_chunks
        else "I don't have information on that in the provided documents."
    )
    return QueryResponse(
        answer=answer_text,
        citations=[],
        model=s.generation_model,
        stop_reason="naive_pipeline",
        retrieved_count=len(naive_chunks),
        session_id=body.session_id,
        rewritten_query=None,
    )


@app.post("/internal/query", tags=["internal"], dependencies=[Depends(require_internal_secret)])
async def internal_query(body: QueryRequest) -> QueryResponse:
    s = get_settings()
    if body.pipeline == "naive":
        return await _run_naive_query(body, s)

    effective_query, rewritten_query, chunks = await _prepare_query(body, s)

    answer = await run_in_threadpool(
        generate_answer, effective_query, chunks, settings=s, client=get_anthropic_client()
    )

    if body.session_id:
        await run_in_threadpool(
            get_session_store().add_turn,
            Turn(
                session_id=body.session_id,
                user_query=body.query,
                rewritten_query=effective_query,
                answer_text=answer.text,
            ),
        )

    return QueryResponse(
        answer=answer.text,
        citations=[CitationOut(**asdict(c)) for c in answer.citations],
        model=answer.model,
        stop_reason=answer.stop_reason,
        retrieved_count=len(chunks),
        session_id=body.session_id,
        rewritten_query=rewritten_query,
    )


def _sse_line(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/internal/query/stream", tags=["internal"], dependencies=[Depends(require_internal_secret)])
async def internal_query_stream(body: QueryRequest) -> StreamingResponse:
    """Same pipeline as /internal/query (Stage 9), delivered as SSE: one
    "meta" event first (session/retrieval info), then "token" events as the
    answer streams in, then one final "done" event with the full answer text
    + citations + stop_reason. Score-floor/no-context short-circuits still
    skip the Claude call entirely - "done" arrives immediately with no
    "token" events at all."""
    s = get_settings()

    if body.pipeline == "naive":
        naive_chunks = await run_in_threadpool(
            naive_search, body.query, settings=s, embedder=get_embedder(), top_k=body.top_k or 5
        )

        def naive_event_stream() -> Iterator[str]:
            yield _sse_line(
                "meta",
                {"session_id": body.session_id, "rewritten_query": None, "retrieved_count": len(naive_chunks)},
            )
            # naive_generate has no streaming variant (see app/naive/generation.py) -
            # the whole answer arrives as one "token" event rather than token-by-token.
            # An Anthropic API failure here must still end the SSE stream with a
            # normal "done" event, not an uncaught exception - see the matching
            # comment in generate_answer_stream (app/generation/claude.py) for why
            # an exception after the 200 + streaming headers are already sent
            # aborts the connection instead of becoming a clean HTTP error.
            try:
                answer_text = (
                    naive_generate(body.query, naive_chunks, settings=s, client=get_anthropic_client())
                    if naive_chunks
                    else "I don't have information on that in the provided documents."
                )
            except anthropic.APIError as exc:
                log.error("naive_generation_stream_failed", error=str(exc))
                yield _sse_line(
                    "done",
                    {
                        "text": "Something went wrong generating an answer. Please try again in a moment.",
                        "citations": [],
                        "model": s.generation_model,
                        "stop_reason": "generation_error",
                    },
                )
                return
            yield _sse_line("token", {"text": answer_text})
            yield _sse_line(
                "done",
                {"text": answer_text, "citations": [], "model": s.generation_model, "stop_reason": "naive_pipeline"},
            )

        return StreamingResponse(naive_event_stream(), media_type="text/event-stream")

    effective_query, rewritten_query, chunks = await _prepare_query(body, s)

    def event_stream() -> Iterator[str]:
        yield _sse_line(
            "meta",
            {
                "session_id": body.session_id,
                "rewritten_query": rewritten_query,
                "retrieved_count": len(chunks),
            },
        )
        final_answer_text: str | None = None
        for name, payload in generate_answer_stream(
            effective_query, chunks, settings=s, client=get_anthropic_client()
        ):
            if name == "done":
                final_answer_text = payload["text"]
            yield _sse_line(name, payload)

        if body.session_id and final_answer_text is not None:
            get_session_store().add_turn(
                Turn(
                    session_id=body.session_id,
                    user_query=body.query,
                    rewritten_query=effective_query,
                    answer_text=final_answer_text,
                )
            )

    # event_stream is a plain (sync) generator - Starlette's StreamingResponse
    # iterates it via a thread pool automatically, the same "never block the
    # event loop" principle as every run_in_threadpool call elsewhere here.
    return StreamingResponse(event_stream(), media_type="text/event-stream")


class SourceOut(BaseModel):
    doc_id: str
    filename: str
    mime_type: str
    status: str
    page_count: int | None
    version: str | None
    supersedes: str | None
    chunk_count: int
    ingested_at: str


@app.get("/internal/sources/{doc_id}", tags=["internal"], dependencies=[Depends(require_internal_secret)])
async def internal_get_source(doc_id: str) -> SourceOut:
    """Registry lookup for citation click-through (Stage 9) - metadata only,
    not the original file bytes; serving/previewing those is a Stage 10
    concern if the frontend actually needs it."""
    record = await run_in_threadpool(get_registry().get, doc_id)
    if record is None:
        raise HTTPException(404, f"unknown doc_id: {doc_id}")
    return SourceOut(
        doc_id=record.doc_id,
        filename=record.filename,
        mime_type=record.mime_type,
        status=record.status.value,
        page_count=record.page_count,
        version=record.version,
        supersedes=record.supersedes,
        chunk_count=record.chunk_count,
        ingested_at=record.ingested_at.isoformat(),
    )
