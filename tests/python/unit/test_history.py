"""Unit tests for the SQLite audit-history store."""

from __future__ import annotations

from pathlib import Path

import pytest

from python_services.frequency_guard.api.history import HistoryRow, HistoryStore


def _insert_row(store: HistoryStore, source: str = "img.png", verdict: str = "ai") -> int:
    return store.insert(
        request_id="req-123",
        source=source,
        verdict=verdict,
        confidence=87.5,
        fake_probability=0.875,
        latency_ms=42.0,
        model_version="2.0.0",
        file_sha256="a" * 64,
        payload={"features": {"spectral_slope": -2.1}},
    )


class TestHistoryStore:
    def test_insert_and_recent(self, tmp_path: Path) -> None:
        store = HistoryStore(tmp_path / "h.sqlite3")
        row_id = _insert_row(store)
        assert row_id > 0

        rows = store.recent(limit=10)
        assert len(rows) == 1
        assert isinstance(rows[0], HistoryRow)
        assert rows[0].request_id == "req-123"
        assert rows[0].verdict == "ai"
        assert rows[0].confidence == pytest.approx(87.5)
        assert rows[0].file_sha256 == "a" * 64

    def test_recent_newest_first(self, tmp_path: Path) -> None:
        store = HistoryStore(tmp_path / "h.sqlite3")
        first = _insert_row(store)
        second = _insert_row(store)
        rows = store.recent(limit=10)
        assert [r.id for r in rows] == [second, first]

    def test_limit_clamped(self, tmp_path: Path) -> None:
        store = HistoryStore(tmp_path / "h.sqlite3")
        for _ in range(3):
            _insert_row(store)
        assert len(store.recent(limit=2)) == 2
        # absurd limits are clamped into a sane range rather than crashing
        assert len(store.recent(limit=10000)) == 3

    def test_stats_aggregates(self, tmp_path: Path) -> None:
        store = HistoryStore(tmp_path / "h.sqlite3")
        _insert_row(store, verdict="ai")
        _insert_row(store, verdict="authentic")
        stats = store.stats()
        assert stats["total"] == 2.0
        assert stats["n_ai"] == 1.0
        assert stats["avg_latency_ms"] == pytest.approx(42.0)

    def test_clear(self, tmp_path: Path) -> None:
        store = HistoryStore(tmp_path / "h.sqlite3")
        _insert_row(store)
        store.clear()
        assert store.recent() == []

    def test_as_dict_roundtrip(self) -> None:
        row = HistoryRow(
            id=1,
            created_at="2026-01-01T00:00:00Z",
            request_id="r",
            source="s",
            verdict="ai",
            confidence=50.0,
            fake_probability=0.5,
            latency_ms=10.0,
            model_version="v",
            file_sha256="f" * 64,
        )
        d = row.as_dict()
        assert d["id"] == 1
        assert d["confidence"] == pytest.approx(50.0)
