from __future__ import annotations

from pathlib import Path

import pytest
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.ingestion.pdf_loader import (
    PdfPage,
    average_chars_per_page,
    build_page_markdown,
    extract_pdf_pages,
    is_scanned,
)


def _make_pdf(path: Path) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    elements = [
        Paragraph("Refund Policy", styles["Title"]),
        Paragraph("International orders are refunded within 30 days.", styles["Normal"]),
        Spacer(1, 12),
        Table(
            [["Region", "Standard", "Express"], ["US", "$5.00", "$15.00"], ["EU", "$12.00", "$28.00"]],
            style=TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]),
        ),
        PageBreak(),
        Paragraph("Domestic Orders", styles["Heading2"]),
        Paragraph("Domestic refunds are processed within 5 business days.", styles["Normal"]),
    ]
    doc.build(elements)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "sample.pdf"
    _make_pdf(path)
    return path


def test_extract_pdf_pages_returns_one_entry_per_page(sample_pdf: Path):
    pages = extract_pdf_pages(sample_pdf)
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2


def test_table_is_extracted_separately_from_prose_text(sample_pdf: Path):
    pages = extract_pdf_pages(sample_pdf)
    page1 = pages[0]
    assert "Refund Policy" in page1.text
    assert "International orders are refunded within 30 days." in page1.text
    # the table's cell values must NOT also appear duplicated in the prose text
    assert "$5.00" not in page1.text
    assert len(page1.tables_markdown) == 1
    assert "| Region | Standard | Express |" in page1.tables_markdown[0]
    assert "| US | $5.00 | $15.00 |" in page1.tables_markdown[0]


def test_page_with_no_table_has_empty_tables_markdown(sample_pdf: Path):
    pages = extract_pdf_pages(sample_pdf)
    assert pages[1].tables_markdown == []


def test_average_chars_per_page_and_is_scanned_on_real_text():
    pages = [PdfPage(1, "a" * 500), PdfPage(2, "b" * 300)]
    assert average_chars_per_page(pages) == 400.0
    assert is_scanned(pages, threshold=20.0) is False


def test_is_scanned_true_for_near_empty_pages():
    pages = [PdfPage(1, ""), PdfPage(2, "  "), PdfPage(3, "x")]
    assert is_scanned(pages, threshold=20.0) is True


def test_is_scanned_handles_empty_document():
    assert average_chars_per_page([]) == 0.0
    assert is_scanned([], threshold=20.0) is True


def test_build_page_markdown_strips_boilerplate_and_reattaches_tables():
    pages = [
        PdfPage(1, "Header\nIntro text.", tables_markdown=[]),
        PdfPage(2, "Header\nMore text.", tables_markdown=["| a | b |\n| --- | --- |\n| 1 | 2 |"]),
        PdfPage(3, "Header\nClosing text.", tables_markdown=[]),
    ]
    result = build_page_markdown(pages)
    assert len(result) == 3
    page_numbers = [pn for pn, _ in result]
    assert page_numbers == [1, 2, 3]
    md_by_page = dict(result)
    assert "Header" not in md_by_page[1]  # repeated across all pages -> stripped
    assert "Intro text." in md_by_page[1]
    assert "| a | b |" in md_by_page[2]
    assert "More text." in md_by_page[2]


def test_build_page_markdown_drops_pages_left_empty_after_stripping():
    pages = [
        PdfPage(1, "Header only"),
        PdfPage(2, "Header only"),
        PdfPage(3, "Header only\nReal content here."),
    ]
    result = build_page_markdown(pages)
    # pages 1 and 2 become empty once the repeated line is stripped
    assert [pn for pn, _ in result] == [3]
