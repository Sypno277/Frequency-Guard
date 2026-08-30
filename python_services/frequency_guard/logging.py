"""Structured logging for Frequency Guard.

Emits JSON lines to stdout (and optionally a file) with a configurable
verbosity. Log records include timestamp, level, logger name, message, and
any extra key/value context passed by the caller.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

_configured = False


class JsonFormatter(logging.Formatter):
    """Minimal JSON-lines formatter (no third-party dependency)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        extra = getattr(record, "fields", None)
        if isinstance(extra, Mapping):
            payload.update(extra)
        return json.dumps(payload, default=str)


def configure_logging(
    level: str = "INFO",
    log_file: str | None = None,
    force: bool = False,
) -> None:
    """Configure root logging once. Idempotent unless ``force`` is set."""
    global _configured
    if _configured and not force:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger for ``name``.

    Usage:
        log = get_logger(__name__)
        log.info("analysis complete", extra={"fields": {"latency_ms": 42}})
    """
    return logging.getLogger(name)


def with_fields(logger: logging.Logger, **fields: Any) -> logging.LoggerAdapter:
    """Wrap ``logger`` so every emitted record carries ``fields``."""
    return logging.LoggerAdapter(logger, {"fields": fields})
