"""Unit tests for config and logging modules."""

from __future__ import annotations

import json
import logging
from io import StringIO

import pytest

from python_services.frequency_guard.config import Settings, ensure_dirs, load_settings
from python_services.frequency_guard.logging import (
    JsonFormatter,
    configure_logging,
    get_logger,
    with_fields,
)


class TestSettings:
    """Settings defaults, env override, and directory creation."""

    def test_defaults_no_side_effects(self) -> None:
        """Building Settings must not create directories (pure import rule)."""
        settings = Settings()
        assert settings.image_size == 256
        assert settings.host == "0.0.0.0"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FG_IMAGE_SIZE", "192")
        monkeypatch.setenv("FG_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("FG_BATCH_WORKERS", "3")
        settings = load_settings()
        assert settings.image_size == 192
        assert settings.log_level == "DEBUG"
        assert settings.batch_workers == 3

    def test_ensure_dirs_creates_paths(self, tmp_path) -> None:
        base = tmp_path / "fg"
        settings = Settings(model_dir=base / "models", reports_dir=base / "reports", data_dir=base / "data")
        ensure_dirs(settings)
        assert (base / "models").is_dir()
        assert (base / "reports").is_dir()
        assert (base / "data").is_dir()


class TestJsonFormatter:
    """JsonFormatter output shape."""

    def test_json_format_contains_fields(self) -> None:
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter())
        logger = logging.getLogger("fg.test.serializer")
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.info("hello", extra={"fields": {"latency_ms": 12}})
        record = json.loads(stream.getvalue())
        assert record["level"] == "INFO"
        assert record["message"] == "hello"
        assert record["latency_ms"] == 12
        logger.handlers = []


class TestLoggingConfig:
    """configure_logging idempotency and verbosity."""

    def test_configure_logging_idempotent(self) -> None:
        configure_logging(force=True)
        configure_logging(force=False)
        assert logging.getLogger().level in (logging.INFO, logging.DEBUG)

    def test_level_from_string(self) -> None:
        configure_logging("DEBUG", force=True)
        assert logging.getLogger().level == logging.DEBUG


class TestLoggerFactory:
    def test_get_logger_returns_child(self) -> None:
        logger = get_logger("fg.unit")
        assert logger.name == "fg.unit"

    def test_with_fields_adapter(self) -> None:
        logger = get_logger("fg.unit.fields")
        adapter = with_fields(logger, source="test")
        assert adapter.extra == {"fields": {"source": "test"}}
