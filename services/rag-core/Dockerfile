# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:/usr/local/bin:$PATH"
WORKDIR /app
# tesseract-ocr: local OCR fallback for scanned PDFs (Stage 4) - free, no paid
# API. Its own license (Apache-2.0) and pypdfium2/pdfplumber (also permissive)
# were chosen specifically to avoid AGPL alternatives like PyMuPDF.
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./

# --- development image: full deps (incl. dev group), live-reload ---
FROM base AS dev
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project
# Pre-warm fastembed's local BM25 tokenizer asset (Stage 5 hybrid retrieval) at
# build time - it's a tiny (~KB) tokenizer/stopword file from Hugging Face, not
# a paid API, but baking it into the image means the running container never
# needs network for it (same reasoning as installing tesseract-ocr above).
RUN uv run python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25')"
COPY . .
EXPOSE 8000
CMD ["uv", "run", "--frozen", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# --- production image: runtime deps only, run the venv binary directly ---
FROM base AS prod
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev
RUN uv run python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25')"
COPY app ./app
EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
