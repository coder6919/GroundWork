from __future__ import annotations

from pathlib import Path

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate

from app.config import Settings
from app.db.models import DocumentStatus
from app.db.registry import SQLiteDocumentRegistry
from app.ingestion.pdf_loader import PdfPage
from app.ingestion.pipeline import _load_and_chunk_pdf, ingest_path


def _make_pdf(path: Path, text: str = "Some real extractable text content.") -> None:
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    doc.build([Paragraph(text, getSampleStyleSheet()["Normal"])])


# ---- _load_and_chunk_pdf routing (extract_pdf_pages / ocr_pdf_pages mocked) ----


def test_non_scanned_pdf_never_calls_ocr(monkeypatch, tmp_path: Path):
    calls = {"ocr": 0}
    monkeypatch.setattr(
        "app.ingestion.pipeline.extract_pdf_pages",
        lambda path: [PdfPage(1, "Plenty of real text here, well above threshold.")],
    )
    monkeypatch.setattr(
        "app.ingestion.pipeline.ocr_pdf_pages", lambda *a, **k: calls.__setitem__("ocr", calls["ocr"] + 1)
    )

    chunks, page_count, reason = _load_and_chunk_pdf(tmp_path / "x.pdf", Settings())

    assert calls["ocr"] == 0
    assert reason is None
    assert page_count == 1
    assert chunks and chunks[0].page_number == 1


def test_scanned_pdf_triggers_ocr_when_enabled(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "app.ingestion.pipeline.extract_pdf_pages", lambda path: [PdfPage(1, ""), PdfPage(2, "")]
    )
    monkeypatch.setattr(
        "app.ingestion.pipeline.ocr_pdf_pages",
        lambda path, dpi: [PdfPage(1, "OCR recovered this text from the scan."), PdfPage(2, "More OCR text.")],
    )

    settings = Settings(ocr_enabled=True)
    chunks, page_count, reason = _load_and_chunk_pdf(tmp_path / "x.pdf", settings)

    assert reason is None
    assert page_count == 2
    assert any("OCR recovered" in c.text for c in chunks)


def test_scanned_pdf_with_ocr_disabled_is_unprocessable(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "app.ingestion.pipeline.extract_pdf_pages", lambda path: [PdfPage(1, ""), PdfPage(2, "")]
    )
    ocr_called = []
    monkeypatch.setattr("app.ingestion.pipeline.ocr_pdf_pages", lambda *a, **k: ocr_called.append(1))

    settings = Settings(ocr_enabled=False)
    chunks, page_count, reason = _load_and_chunk_pdf(tmp_path / "x.pdf", settings)

    assert ocr_called == []
    assert chunks == []
    assert page_count == 2
    assert reason == "scanned/image-only PDF and OCR is disabled"


def test_ocr_still_scanned_after_running_is_unprocessable(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("app.ingestion.pipeline.extract_pdf_pages", lambda path: [PdfPage(1, "")])
    monkeypatch.setattr(
        "app.ingestion.pipeline.ocr_pdf_pages", lambda path, dpi: [PdfPage(1, "")]  # OCR also found nothing
    )

    settings = Settings(ocr_enabled=True)
    chunks, _page_count, reason = _load_and_chunk_pdf(tmp_path / "x.pdf", settings)

    assert chunks == []
    assert reason == "OCR did not yield enough extractable text"


# ---- full ingest_path with a real generated PDF ----


class _FakeEmbedder:
    def embed(self, texts, input_type):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeQdrantClient:
    def __init__(self):
        self.upserted_points = []

    def collection_exists(self, name):
        return False

    def create_collection(self, collection_name, vectors_config, sparse_vectors_config=None):
        pass

    def delete(self, collection_name, points_selector, **kwargs):
        pass

    def upsert(self, collection_name, points, wait=True):
        self.upserted_points.extend(points)

    def set_payload(self, collection_name, payload, points, **kwargs):
        pass


@pytest.fixture(autouse=True)
def _patch_qdrant(monkeypatch):
    fake = _FakeQdrantClient()
    monkeypatch.setattr("app.ingestion.pipeline.make_client", lambda timeout=30.0: fake)
    return fake


def test_ingest_path_populates_page_metadata_for_a_real_pdf(tmp_path: Path, _patch_qdrant):
    _make_pdf(tmp_path / "doc.pdf", "Real extractable text for this PDF ingestion test.")
    settings = Settings(sqlite_path=str(tmp_path / "registry.db"), storage_dir=str(tmp_path / "storage"))
    registry = SQLiteDocumentRegistry(settings.sqlite_path)

    results = ingest_path(tmp_path, settings=settings, registry=registry, embedder=_FakeEmbedder())

    assert len(results) == 1
    assert results[0].status == DocumentStatus.READY
    record = registry.get(results[0].doc_id)
    assert record is not None
    assert record.page_count == 1

    points = _patch_qdrant.upserted_points
    assert points
    assert points[0].payload["page_start"] == 1
    assert points[0].payload["page_end"] == 1
