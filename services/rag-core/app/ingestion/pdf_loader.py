"""Text-based PDF extraction: per-page prose + tables, formatted as Markdown
so it flows through the existing structure-aware chunker unchanged.

Uses pdfplumber (MIT) rather than PyMuPDF/pymupdf4llm to keep the dependency
tree permissively licensed - PyMuPDF is AGPL-3.0/commercial dual-licensed,
which doesn't fit a repo meant to stay a clean Apache-2.0 reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

from .boilerplate import strip_boilerplate


@dataclass
class PdfPage:
    page_number: int  # 1-indexed
    text: str  # prose text with table regions excluded (see extract_pdf_pages)
    tables_markdown: list[str] = field(default_factory=list)


def extract_pdf_pages(path: Path) -> list[PdfPage]:
    """Extract every page's prose text (tables excluded, to avoid the table
    content appearing twice) plus each table formatted as its own Markdown block."""
    pages: list[PdfPage] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables()

            text_only_page = page
            for table in tables:
                text_only_page = text_only_page.outside_bbox(table.bbox)
            text = (text_only_page.extract_text() or "").strip()

            tables_markdown = [_table_to_markdown(t.extract()) for t in tables]
            tables_markdown = [t for t in tables_markdown if t]

            pages.append(PdfPage(page_number=i, text=text, tables_markdown=tables_markdown))
    return pages


def _table_to_markdown(rows: list[list[str | None]]) -> str:
    if not rows:
        return ""
    cleaned = [[(cell or "").strip().replace("\n", " ") for cell in row] for row in rows]
    header, *body = cleaned
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(" --- " for _ in header) + "|",
    ]
    for row in body:
        padded = (row + [""] * len(header))[: len(header)]  # defensive: ragged rows
        lines.append("| " + " | ".join(padded) + " |")
    return "\n".join(lines)


def average_chars_per_page(pages: list[PdfPage]) -> float:
    if not pages:
        return 0.0
    return sum(len(p.text) for p in pages) / len(pages)


def is_scanned(pages: list[PdfPage], threshold: float) -> bool:
    """A near-zero average text density signals an image-only/scanned PDF."""
    return average_chars_per_page(pages) < threshold


def build_page_markdown(pages: list[PdfPage]) -> list[tuple[int, str]]:
    """Strip cross-page boilerplate from prose text, then reattach each
    page's tables. Returns (page_number, markdown) pairs, one per page that
    has any content left."""
    cleaned_texts = strip_boilerplate([p.text for p in pages])
    result = []
    for page, text in zip(pages, cleaned_texts):
        markdown = "\n\n".join([text, *page.tables_markdown]).strip()
        if markdown:
            result.append((page.page_number, markdown))
    return result
