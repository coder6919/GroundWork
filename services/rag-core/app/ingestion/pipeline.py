"""Ingestion orchestration: discover -> hash/dedup -> chunk -> embed -> upsert.

Every file is isolated in its own try/except so one bad file never aborts a
batch (per the build spec's per-file failure isolation requirement).
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

from ..clients.qdrant import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    delete_document_points,
    ensure_hybrid_collection,
    make_client,
    mark_superseded,
    upsert_points,
)
from ..config import Settings
from ..db.models import DocumentRecord, DocumentStatus
from ..db.registry import DocumentRegistry
from ..logging import get_logger
from ..providers.base import EmbeddingProvider
from ..retrieval import sparse
from .chunking import Chunk, chunk_text
from .dedup import content_hash_file
from .loaders import SUPPORTED_EXTENSIONS, detect_mime_type, load_text
from .ocr import ocr_pdf_pages
from .pdf_loader import build_page_markdown, extract_pdf_pages, is_scanned

log = get_logger("ingestion.pipeline")


@dataclass
class FileResult:
    doc_id: str
    filename: str
    status: DocumentStatus
    chunk_count: int = 0
    failure_reason: str | None = None
    skipped_duplicate: bool = False


def discover_files(root: Path) -> list[Path]:
    """A single explicit file is always returned (letting an unsupported type
    surface as a clear per-file failure); a directory walk is pre-filtered to
    known extensions so unrelated files are silently skipped, not "failed"."""
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def _normalize_rel(relative_path: str) -> str:
    """Backslash/forward-slash-insensitive form, so a caller-supplied
    `supersedes`/`versions` key matches regardless of OS path separators."""
    return relative_path.replace("\\", "/")


def _stable_doc_id(relative_path: Path) -> str:
    """Deterministic from the relative path, so re-ingesting the same path
    always updates the same registry row and Qdrant points instead of creating
    duplicates."""
    return uuid.uuid5(uuid.NAMESPACE_URL, _normalize_rel(str(relative_path))).hex


def ingest_path(
    root: Path,
    *,
    settings: Settings,
    registry: DocumentRegistry,
    embedder: EmbeddingProvider,
    supersedes: dict[str, str] | None = None,
    versions: dict[str, str] | None = None,
) -> list[FileResult]:
    """`supersedes` maps a newly-ingested file's relative path to the relative
    path of the older document it replaces (Stage 5): the old document's Qdrant
    points are flagged `superseded` (not deleted - kept for audit) and excluded
    from retrieval by default. `versions` maps a relative path to a caller-
    supplied version label, stored on the registry row and in the payload for
    metadata filtering. Both are optional and keyed by the same relative-path
    strings the caller passed to `POST /internal/ingest`."""
    client = make_client(timeout=30.0)
    if ensure_hybrid_collection(client, settings.qdrant_collection, settings.embedding_dimensions):
        # The collection was dropped and recreated (old single-vector schema) -
        # every registry row now lies about what's actually in Qdrant, so the
        # content-hash dedup check must not be allowed to skip anything.
        log.warning("registry_cleared_after_schema_migration", collection=settings.qdrant_collection)
        registry.clear()

    supersedes_norm = {_normalize_rel(k): _normalize_rel(v) for k, v in (supersedes or {}).items()}
    versions_norm = {_normalize_rel(k): v for k, v in (versions or {}).items()}

    storage_dir = Path(settings.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    base = root if root.is_dir() else root.parent
    results: list[FileResult] = []

    for path in discover_files(root):
        relative = path.relative_to(base)
        rel_key = _normalize_rel(str(relative))
        doc_id = _stable_doc_id(relative)
        doc_version = versions_norm.get(rel_key)
        supersedes_relative = supersedes_norm.get(rel_key)
        try:
            results.append(
                _ingest_file(
                    path,
                    relative,
                    doc_id,
                    settings,
                    registry,
                    embedder,
                    client,
                    storage_dir,
                    doc_version=doc_version,
                    supersedes_relative=supersedes_relative,
                )
            )
        except Exception as exc:  # noqa: BLE001 - isolate this file, keep the batch going
            log.warning("ingest_file_failed", filename=str(relative), error=str(exc))
            registry.upsert(
                DocumentRecord(
                    doc_id=doc_id,
                    filename=str(relative),
                    source_path=str(path),
                    content_hash="",
                    mime_type="",
                    status=DocumentStatus.FAILED,
                    failure_reason=str(exc),
                )
            )
            results.append(
                FileResult(doc_id, str(relative), DocumentStatus.FAILED, failure_reason=str(exc))
            )
    return results


def _load_and_chunk(
    path: Path, mime_type: str, settings: Settings
) -> tuple[list[Chunk], int | None, str | None]:
    """Returns (chunks, page_count, unprocessable_reason). `page_count` is set
    only for PDFs. `unprocessable_reason` is set (chunks empty) when the
    document has no usable extractable text even after any OCR fallback."""
    if mime_type == "application/pdf":
        return _load_and_chunk_pdf(path, settings)

    text = load_text(path)
    chunks = chunk_text(
        text, target_tokens=settings.chunk_target_tokens, overlap_ratio=settings.chunk_overlap_ratio
    )
    return chunks, None, None if chunks else "no extractable text"


def _load_and_chunk_pdf(path: Path, settings: Settings) -> tuple[list[Chunk], int | None, str | None]:
    pages = extract_pdf_pages(path)
    scanned = is_scanned(pages, settings.pdf_scanned_threshold_chars_per_page)

    if scanned:
        if not settings.ocr_enabled:
            return [], len(pages), "scanned/image-only PDF and OCR is disabled"
        log.info("pdf_scanned_detected_running_ocr", pages=len(pages))
        pages = ocr_pdf_pages(path, dpi=settings.pdf_ocr_dpi)
        if is_scanned(pages, settings.pdf_scanned_threshold_chars_per_page):
            return [], len(pages), "OCR did not yield enough extractable text"

    chunks: list[Chunk] = []
    for page_number, markdown in build_page_markdown(pages):
        page_chunks = chunk_text(
            markdown,
            target_tokens=settings.chunk_target_tokens,
            overlap_ratio=settings.chunk_overlap_ratio,
        )
        for chunk in page_chunks:
            chunk.page_number = page_number
        chunks.extend(page_chunks)

    return chunks, len(pages), None if chunks else "no extractable text"


def _ingest_file(
    path: Path,
    relative: Path,
    doc_id: str,
    settings: Settings,
    registry: DocumentRegistry,
    embedder: EmbeddingProvider,
    client: QdrantClient,
    storage_dir: Path,
    *,
    doc_version: str | None = None,
    supersedes_relative: str | None = None,
) -> FileResult:
    mime_type = detect_mime_type(path)
    h = content_hash_file(path)  # streamed - never loads the whole file for hashing

    duplicate = registry.get_by_hash(h)
    if duplicate and duplicate.status == DocumentStatus.READY and duplicate.doc_id != doc_id:
        log.info("ingest_skip_duplicate_content", filename=str(relative), duplicate_of=duplicate.filename)
        return FileResult(doc_id, str(relative), DocumentStatus.READY, skipped_duplicate=True)

    current = registry.get(doc_id)
    if current and current.status == DocumentStatus.READY and current.content_hash == h:
        log.info("ingest_skip_unchanged", filename=str(relative))
        return FileResult(
            doc_id, str(relative), DocumentStatus.READY, current.chunk_count, skipped_duplicate=True
        )

    registry.upsert(
        DocumentRecord(
            doc_id=doc_id,
            filename=str(relative),
            source_path=str(path),
            content_hash=h,
            mime_type=mime_type,
            status=DocumentStatus.PROCESSING,
        )
    )

    chunks, page_count, unprocessable_reason = _load_and_chunk(path, mime_type, settings)
    if unprocessable_reason:
        registry.upsert(
            DocumentRecord(
                doc_id=doc_id,
                filename=str(relative),
                source_path=str(path),
                content_hash=h,
                mime_type=mime_type,
                status=DocumentStatus.UNPROCESSABLE,
                failure_reason=unprocessable_reason,
                page_count=page_count,
            )
        )
        return FileResult(
            doc_id, str(relative), DocumentStatus.UNPROCESSABLE, failure_reason=unprocessable_reason
        )

    dense_vectors = embedder.embed([c.text for c in chunks], input_type="document")
    sparse_vectors = sparse.embed_documents([c.text for c in chunks])

    stored_path = storage_dir / f"{doc_id}{path.suffix.lower()}"
    shutil.copy2(path, stored_path)  # streamed copy - never loads the whole file into memory

    supersedes_doc_id = _stable_doc_id(Path(supersedes_relative)) if supersedes_relative else None

    now = datetime.now(UTC).isoformat()
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{i}")),
            vector={
                DENSE_VECTOR_NAME: dense_vector,
                SPARSE_VECTOR_NAME: SparseVector(indices=sparse_vector.indices, values=sparse_vector.values),
            },
            payload={
                "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{i}")),
                "doc_id": doc_id,
                "source_filename": str(relative),
                "page_start": chunk.page_number,
                "page_end": chunk.page_number,
                "heading_path": chunk.heading_path,
                "chunk_text": chunk.text,
                "token_count": chunk.token_count,
                "content_type": chunk.content_type,
                "ingested_at": now,
                "doc_version": doc_version,
                "supersedes_doc_id": supersedes_doc_id,
                "superseded": False,
                "tenant_id": "default",
                "source_url": None,
            },
        )
        for i, (chunk, dense_vector, sparse_vector) in enumerate(zip(chunks, dense_vectors, sparse_vectors))
    ]

    delete_document_points(client, settings.qdrant_collection, doc_id)
    upsert_points(client, settings.qdrant_collection, points)

    if supersedes_doc_id:
        mark_superseded(client, settings.qdrant_collection, supersedes_doc_id)
        log.info(
            "ingest_marked_superseded", filename=str(relative), supersedes=supersedes_relative
        )

    registry.upsert(
        DocumentRecord(
            doc_id=doc_id,
            filename=str(relative),
            source_path=str(stored_path),
            content_hash=h,
            mime_type=mime_type,
            status=DocumentStatus.READY,
            chunk_count=len(chunks),
            page_count=page_count,
            version=doc_version,
            supersedes=supersedes_doc_id,
        )
    )
    log.info("ingest_file_ready", filename=str(relative), chunks=len(chunks), pages=page_count)
    return FileResult(doc_id, str(relative), DocumentStatus.READY, chunk_count=len(chunks))
