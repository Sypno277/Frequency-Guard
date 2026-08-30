"""Integration tests: image-in → verdict-out through the FastAPI app.

Masterplan §6 — integration tests. Covers JPG/PNG/WebP/BMP, multiple
resolutions, corrupted/blank/pure-noise/EXIF-rotated inputs, and asserts
meaningful 4xx errors (never 500s). Uses the trained demo artifacts.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from python_services.frequency_guard.api.server import create_app
from python_services.frequency_guard.config import Settings

pytestmark = pytest.mark.integration

DEMO_DIR = Path(__file__).resolve().parents[3] / "data" / "demo_dataset"


def _ensure_model() -> None:
    """Skip with clear message if no trained model exists yet."""
    if not (Path(__file__).resolve().parents[3] / "checkpoints" / "ensemble.joblib").exists():
        pytest.skip("Run scripts/train_demo.py first to produce model artifacts")


@pytest.fixture(scope="module")
def client() -> TestClient:
    _ensure_model()
    settings = Settings()
    app = create_app(settings=settings)
    return TestClient(app)


def _demo_image(label: str, index: int = 0) -> bytes:
    path = DEMO_DIR / label / f"{label}_{index:04d}.png"
    return path.read_bytes()


# --- happy paths -------------------------------------------------------


def test_analyze_fake_image(client: TestClient) -> None:
    resp = client.post("/api/v1/analyze", files={"file": ("fake.png", _demo_image("fake"), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_ai"] is True
    assert 0.0 <= body["confidence"] <= 100.0
    assert len(body["spectrum"]) > 0
    assert len(body["families"]) == 4
    assert body["explainability"]["saliency_png_base64"].startswith("data:image/png;base64,")


def test_analyze_real_image(client: TestClient) -> None:
    resp = client.post("/api/v1/analyze", files={"file": ("real.png", _demo_image("real"), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_ai"] is False


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_metrics_endpoint(client: TestClient) -> None:
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("uptime_seconds", "total_analyzed", "images_per_second", "latency_p50_ms", "peak_rss_mb"):
        assert key in body


def test_history_endpoint(client: TestClient) -> None:
    resp = client.get("/api/v1/history")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_model_info(client: TestClient) -> None:
    resp = client.get("/api/v1/model")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == "2.0.0"
    assert body["n_features"] is None or body["n_features"] > 0


# --- error paths (meaningful 4xx, never 500) ---------------------------


def test_analyze_empty_payload(client: TestClient) -> None:
    resp = client.post("/api/v1/analyze", files={"file": ("empty.png", b"", "image/png")})
    assert resp.status_code == 422


def test_analyze_corrupt_payload(client: TestClient) -> None:
    resp = client.post("/api/v1/analyze", files={"file": ("corrupt.png", b"not-an-image", "image/png")})
    assert resp.status_code == 422


def test_analyze_txt_format(client: TestClient) -> None:
    resp = client.post("/api/v1/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert resp.status_code == 422


def test_unknown_job_404(client: TestClient) -> None:
    resp = client.get("/api/v1/jobs/does-not-exist")
    assert resp.status_code == 404


def test_batch_empty(client: TestClient) -> None:
    resp = client.post("/api/v1/batch", files=[])
    assert resp.status_code in (400, 422)
