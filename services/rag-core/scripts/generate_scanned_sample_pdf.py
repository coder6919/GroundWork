"""Generate corpus/sample/scanned-notice.pdf - a synthetic *image-only* PDF
(text rendered to a bitmap, no real text layer) used to prove the Stage 4
scanned-PDF detection + Tesseract OCR fallback actually works end to end.

Requires the dev-only `reportlab` dependency:
    uv run --frozen --group dev python -m scripts.generate_scanned_sample_pdf
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

OUTPUT_PATH = Path(__file__).resolve().parents[3] / "corpus" / "sample" / "scanned-notice.pdf"

NOTICE_LINES = [
    "Workplace Safety Notice",
    "",
    "All visitors must sign in at the front desk and wear a visible badge",
    "at all times while on the premises. Safety glasses are required in",
    "the manufacturing area. Report any incident to your site supervisor",
    "within 24 hours.",
]


def _render_text_image(width_px: int = 1700, height_px: int = 2200) -> Image.Image:
    image = Image.new("RGB", (width_px, height_px), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 40)
    except OSError:
        font = ImageFont.load_default()
    y = 150
    for line in NOTICE_LINES:
        draw.text((120, y), line, fill="black", font=font)
        y += 70
    return image


def build() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    image = _render_text_image()
    image_path = OUTPUT_PATH.with_suffix(".png")
    image.save(image_path)

    c = canvas.Canvas(str(OUTPUT_PATH), pagesize=letter)
    page_w, page_h = letter
    c.drawImage(str(image_path), 0, 0, width=page_w, height=page_h)
    c.showPage()
    c.save()
    image_path.unlink()
    print(f"wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes) - image-only, no text layer")


if __name__ == "__main__":
    build()
