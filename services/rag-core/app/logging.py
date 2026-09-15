"""Structured logging.

`log_external_call` is the single, greppable event for any outbound paid/provider
call. Nothing invokes it in Stage 0 - it exists so that from here on every billable
call is auditable with:  docker compose logs | grep external_call
"""

from __future__ import annotations

import logging

import structlog


def configure_logging(level: str = "info", json_logs: bool = False) -> None:
    logging.basicConfig(format="%(message)s", level=getattr(logging, level.upper(), logging.INFO))
    renderer = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=True)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "rag_core") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


def log_external_call(provider: str, operation: str, **fields: object) -> None:
    """Emit a uniform record for an outbound paid/provider call (audit hook)."""
    get_logger("external_call").info(
        "external_call", provider=provider, operation=operation, **fields
    )
