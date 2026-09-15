"""OCR fallback for scanned/image-only PDFs.

Local Tesseract only - free, no paid API, consistent with the project's
cost-minimal standing rule. Requires the `tesseract-ocr` system package
(installed in the Docker image; not present on a bare host, so this module
is exercised by a live container check, not host-side unit tests).
"""

from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium
import pytesseract

from ..logging import get_logger
from .pdf_loader import PdfPage

log = get_logger("ingestion.ocr")

_DEFAULT_RENDER_DPI = 72  # pypdfium2's `scale` is relative to this


def ocr_pdf_pages(path: Path, dpi: int = 200) -> list[PdfPage]:
    """Rasterize each page and OCR it. OCR'd text has no real structure, so
    every page comes back with an empty `tables_markdown` - table detection
    for scanned documents is a known v1 limitation."""
    scale = dpi / _DEFAULT_RENDER_DPI
    pages: list[PdfPage] = []
    pdf = pdfium.PdfDocument(str(path))
    try:
        for i in range(len(pdf)):
            image = pdf[i].render(scale=scale).to_pil()
            text = pytesseract.image_to_string(image).strip()
            pages.append(PdfPage(page_number=i + 1, text=text))
            log.info("ocr_page_done", page=i + 1, chars=len(text))
    finally:
        pdf.close()
    return pages
