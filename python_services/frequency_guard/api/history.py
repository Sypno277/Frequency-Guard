"""SQLite audit-history store (Masterplan §3.1 LOGGING/MONITORING, §5.5).

Persists one row per analysis: timestamp, request id, source filename,
verdict, calibrated confidence, latency, model version, and file hash.
Thread-safe (single connection + lock); powers the dashboard's
History/audit panel via ``GET /api/v1/history``.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    request_id TEXT NOT NULL,
    source TEXT NOT NULL,
    verdict TEXT NOT NULL,
    verdict_state TEXT,
    confidence REAL NOT NULL,
    fake_probability REAL NOT NULL,
    latency_ms REAL NOT NULL,
    model_version TEXT NOT NULL,
    file_sha256 TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_history_created ON analysis_history(created_at DESC);
"""

#: Migration for databases created before ``verdict_state`` existed.
_SCHEMA_MIGRATION = "ALTER TABLE analysis_history ADD COLUMN verdict_state TEXT;"


@dataclass(frozen=True)
class HistoryRow:
    """One persisted analysis record."""

    id: int
    created_at: str
    request_id: str
    source: str
    verdict: str
    confidence: float
    fake_probability: float
    latency_ms: float
    model_version: str
    file_sha256: str
    verdict_state: str | None = None
    severity: str | None = None  # "[DETERMINISTIC AI]" etc. (FrequencyGuard Phase 4)
    narrative: str | None = None  # natural-language summary (Phase 4)

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "request_id": self.request_id,
            "source": self.source,
            "verdict": self.verdict,
            "verdict_state": self.verdict_state,
            "confidence": self.confidence,
            "fake_probability": self.fake_probability,
            "latency_ms": self.latency_ms,
            "model_version": self.model_version,
            "file_sha256": self.file_sha256,
            "severity": self.severity,
            "narrative": self.narrative,
        }


class HistoryStore:
    """Bounded SQLite-backed audit log."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            # Migrate pre-verdict_state databases.
            columns = {r[1] for r in conn.execute("PRAGMA table_info(analysis_history)")}
            if "verdict_state" not in columns:
                with contextlib.suppress(sqlite3.OperationalError):
                    conn.execute(_SCHEMA_MIGRATION)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def insert(
        self,
        request_id: str,
        source: str,
        verdict: str,
        confidence: float,
        fake_probability: float,
        latency_ms: float,
        model_version: str,
        file_sha256: str,
        payload: dict[str, object] | None = None,
        verdict_state: str | None = None,
    ) -> int:
        """Insert one analysis row; returns its id."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO analysis_history
                    (created_at, request_id, source, verdict, verdict_state,
                     confidence, fake_probability, latency_ms, model_version,
                     file_sha256, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    request_id,
                    source[:512],
                    verdict,
                    verdict_state,
                    float(confidence),
                    float(fake_probability),
                    float(latency_ms),
                    model_version,
                    file_sha256,
                    json.dumps(payload or {}),
                ),
            )
            return int(cursor.lastrowid or 0)

    def recent(self, limit: int = 100) -> list[HistoryRow]:
        """Return the most recent rows, newest first."""
        limit = max(1, min(limit, 1000))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, request_id, source, verdict, verdict_state,
                       confidence, fake_probability, latency_ms, model_version,
                       file_sha256, payload_json
                FROM analysis_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        result: list[HistoryRow] = []
        for row in rows:
            payload: dict[str, object] = {}
            try:
                payload = json.loads(row["payload_json"] or "{}")
            except (ValueError, TypeError):
                payload = {}
            severity = payload.get("severity")
            narrative = payload.get("narrative")
            result.append(
                HistoryRow(
                    id=row["id"],
                    created_at=row["created_at"],
                    request_id=row["request_id"],
                    source=row["source"],
                    verdict=row["verdict"],
                    verdict_state=row["verdict_state"],
                    confidence=row["confidence"],
                    fake_probability=row["fake_probability"],
                    latency_ms=row["latency_ms"],
                    model_version=row["model_version"],
                    file_sha256=row["file_sha256"],
                    severity=str(severity) if severity is not None else None,
                    narrative=str(narrative) if narrative is not None else None,
                )
            )
        return result

    def stats(self) -> dict[str, float]:
        """Aggregate counters for the metrics panel."""
        with self._lock, self._connect() as conn:
            row = conn.execute("""
                SELECT COUNT(*) AS total,
                       COALESCE(AVG(latency_ms), 0) AS avg_latency,
                       SUM(CASE WHEN verdict = 'ai' THEN 1 ELSE 0 END) AS n_ai,
                       MIN(created_at) AS first_at,
                       MAX(created_at) AS last_at
                FROM analysis_history
                """).fetchone()
        return {
            "total": float(row["total"]),
            "avg_latency_ms": float(row["avg_latency"]),
            "n_ai": float(row["n_ai"] or 0),
        }

    def clear(self) -> None:
        """Delete all rows (used by tests)."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM analysis_history")
