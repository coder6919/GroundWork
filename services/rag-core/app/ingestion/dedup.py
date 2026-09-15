"""Content-hash dedup. Hashing happens before chunking/embedding so an
already-ingested document never gets re-embedded."""

from __future__ import annotations

import hashlib
from pathlib import Path

_READ_CHUNK_BYTES = 1024 * 1024  # 1 MiB


def content_hash(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def content_hash_file(path: Path) -> str:
    """Hash a file in fixed-size chunks instead of reading it whole into
    memory - the large-document-safe path used by the ingestion pipeline."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(_READ_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()
