"""Trust boundary between the public `api` layer and this internal service.

Every non-probe route depends on `require_internal_secret`. The check fails closed:
if no shared secret is configured, all internal calls are rejected.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from .config import get_settings


def _constant_time_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


async def require_internal_secret(
    x_internal_secret: str | None = Header(default=None),
) -> None:
    expected = get_settings().rag_core_shared_secret
    if not expected:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="internal auth not configured (RAG_CORE_SHARED_SECRET unset)",
        )
    if not x_internal_secret or not _constant_time_eq(x_internal_secret, expected):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing X-Internal-Secret",
        )
