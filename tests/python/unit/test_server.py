"""Unit tests for FastAPI server endpoints beyond the integration happy paths."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from python_services.frequency_guard.api.server import create_app
from python_services.frequency_guard.config import Settings

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = ROOT / "data" / "demo_dataset"
CHECKPOINT_DIR = ROOT / "checkpoints"


def _demo_image(rel: str) -> bytes:
    return (DEMO_DIR / rel).read_bytes()


@pytest.fixture(scope="module")
def client() -> TestClient:
    if not (CHECKPOINT_DIR / "ensemble.joblib").exists():
        pytest.skip("Run scripts/train_demo.py first to produce model artifacts")
    app = create_app(settings=Settings())
    return TestClient(app)


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 30.0) -> dict:
    """Poll a batch job until completed/failed or timeout."""
    deadline = time.perf_counter() + timeout
    last = {}
    while time.perf_counter() < deadline:
        resp = client.get(f"/api/v1/jobs/{job_id}")
        assert resp.status_code == 200
        last = resp.json()
        if last["status"] in ("completed", "failed"):
            return last
        time.sleep(0.2)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


class TestBatchEndpoints:
    def test_batch_submit_and_poll(self, client: TestClient) -> None:
        files = [
            ("files", ("fake.png", _demo_image("fake/fake_0000.png"), "image/png")),
            ("files", ("real.png", _demo_image("real/real_0000.png"), "image/png")),
        ]
        resp = client.post("/api/v1/batch", files=files)
        assert resp.status_code == 200
        job = resp.json()
        assert job["job_id"]
        assert job["total_images"] == 2

        final = _wait_for_job(client, job["job_id"])
        assert final["completed"] == 2
        assert len(final["results"]) == 2
        assert final["status"] == "completed"
        # exactly one AI and one authentic
        verdicts = [r["is_ai"] for r in final["results"]]
        assert verdicts.count(True) == 1
        assert verdicts.count(False) == 1

    def test_batch_csv_download(self, client: TestClient) -> None:
        files = [("files", ("fake.png", _demo_image("fake/fake_0001.png"), "image/png"))]
        job = client.post("/api/v1/batch", files=files).json()
        final = _wait_for_job(client, job["job_id"])
        assert final["completed"] == 1

        csv_resp = client.get(f"/api/v1/jobs/{job['job_id']}/csv")
        assert csv_resp.status_code == 200
        assert csv_resp.headers["content-type"].startswith("text/csv")
        text = csv_resp.text
        assert "filename" in text
        # the CSV carries the client-supplied filename, not the server path
        assert "fake.png" in text

    def test_batch_unknown_job_404(self, client: TestClient) -> None:
        assert client.get("/api/v1/jobs/nope").status_code == 404

    def test_batch_csv_unknown_job_404(self, client: TestClient) -> None:
        assert client.get("/api/v1/jobs/nope/csv").status_code == 404


class TestPerformanceEndpoint:
    def test_performance_200_when_report_exists(self, client: TestClient) -> None:
        perf = CHECKPOINT_DIR / "performance.json"
        if not perf.exists():
            pytest.skip("No performance.json artifact")
        resp = client.get("/api/v1/model/performance")
        assert resp.status_code == 200
        body = resp.json()
        assert "accuracy" in body
        assert "confusion_matrix" in body

    def test_performance_409_when_no_report(self, tmp_path) -> None:
        settings = Settings(model_dir=tmp_path / "no_models")
        app = create_app(settings=settings)
        c = TestClient(app)
        resp = c.get("/api/v1/model/performance")
        assert resp.status_code == 409


class TestHistoryEndpoint:
    def test_history_limit_bounds(self, client: TestClient) -> None:
        assert client.get("/api/v1/history?limit=1").status_code == 200
        assert client.get("/api/v1/history?limit=500").status_code == 200
        # too big -> 422 from Query(le=500)
        assert client.get("/api/v1/history?limit=501").status_code == 422
        # too small -> 422 from Query(ge=1)
        assert client.get("/api/v1/history?limit=0").status_code == 422


class TestMetricsEndpoint:
    def test_metrics_contains_cache_and_throughput_fields(self, client: TestClient) -> None:
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200
        body = resp.json()
        for key in (
            "uptime_seconds",
            "total_analyzed",
            "images_per_second",
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_p99_ms",
            "peak_rss_mb",
            "cache_entries",
        ):
            assert key in body
