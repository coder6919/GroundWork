"""Generate corpus/sample/benefits-summary.pdf - a small synthetic multi-page
PDF (real text, a real bordered table, a repeated header/footer, and page
numbers) used as the Stage 4 PDF test fixture and live-ingest demo target.

Not fetched from anywhere - authored the same way corpus/sample/*.md were, so
there's no license question. Requires the dev-only `reportlab` dependency:
    uv run --frozen --group dev python -m scripts.generate_sample_pdf
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUTPUT_PATH = Path(__file__).resolve().parents[3] / "corpus" / "sample" / "benefits-summary.pdf"

HEADER_TEXT = "Employee Benefits Summary"
FOOTER_TEXT = "Confidential - Internal Use Only"


def _add_header_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.drawString(1 * inch, 10.5 * inch, HEADER_TEXT)
    canvas.drawString(1 * inch, 0.5 * inch, FOOTER_TEXT)
    canvas.drawRightString(7.5 * inch, 0.5 * inch, f"Page {doc.page} of 3")
    canvas.restoreState()


def build() -> None:
    styles = getSampleStyleSheet()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUTPUT_PATH), pagesize=letter)

    elements = [
        Paragraph("Health &amp; Wellness Benefits", styles["Title"]),
        Paragraph(
            "All full-time employees are eligible for medical, dental, and vision "
            "coverage starting on their first day of employment. Coverage details "
            "and enrollment deadlines are described below.",
            styles["Normal"],
        ),
        Spacer(1, 12),
        Paragraph(
            "Employees may add dependents during open enrollment or within 30 days "
            "of a qualifying life event such as marriage or the birth of a child.",
            styles["Normal"],
        ),
        PageBreak(),
        Paragraph("Coverage Tiers", styles["Heading2"]),
        Paragraph(
            "The table below summarizes the monthly premium contribution by tier.",
            styles["Normal"],
        ),
        Spacer(1, 12),
        Table(
            [
                ["Tier", "Employee Cost", "Employer Cost"],
                ["Employee Only", "$45.00", "$410.00"],
                ["Employee + Spouse", "$210.00", "$620.00"],
                ["Employee + Children", "$180.00", "$590.00"],
                ["Family", "$340.00", "$890.00"],
            ],
            style=TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            ),
        ),
        PageBreak(),
        Paragraph("Enrollment", styles["Heading2"]),
        Paragraph(
            "New hires must enroll within 14 days of their start date. Employees "
            "who miss this window must wait for the next open enrollment period "
            "unless they experience a qualifying life event.",
            styles["Normal"],
        ),
    ]

    doc.build(elements, onFirstPage=_add_header_footer, onLaterPages=_add_header_footer)
    print(f"wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
