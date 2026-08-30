"""Performance / benchmark gates (Masterplan §6, §8).

Fails CI when:
- p95 inference latency > 150ms per image (CPU, 256px)
- peak RSS > 2GB after a small batch run

Marked ``perf`` and excluded from the default run; CI executes them via
``pytest -m perf``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from python_services.frequency_guard.config import Settings
from python_services.frequency_guard.evaluation.evaluate import current_peak_rss_mb
from python_services.frequency_guard.features.extractor import (
    cache_key,
    extract_features,
    global_cache,
)
from python_services.frequency_guard.io.loader import load_image_file
from python_services.frequency_guard.preprocess import preprocess

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        not (Path(__file__).resolve().parents[3] / "checkpoints" / "ensemble.joblib").exists(),
        reason="Run scripts/train_demo.py first to produce a model",
    ),
]

LATENCY_BUDGET_MS = 150.0
MEMORY_BUDGET_MB = 2048.0
BATCH = 8

DEMO_DIR = Path(__file__).resolve().parents[3] / "data" / "demo_dataset"


def _settings() -> Settings:
    """Full-size defaults (image_size=256)."""
    return Settings()


def _feature_matrix(n: int = BATCH) -> np.ndarray:
    """Extract features for the first n demo images (cached across calls)."""
    X_list = []
    for i in range(n):
        rel = f"fake/fake_{i:04d}.png" if i % 2 == 0 else f"real/real_{i:04d}.png"
        loaded = load_image_file(DEMO_DIR / rel, _settings())
        prep = preprocess(loaded.bgr, _settings())
        key = cache_key(rel, prep, _settings())
        bundle = global_cache.get_or_compute(key, lambda p=prep: extract_features(p, _settings()))
        X_list.append(bundle.vector)
    return np.stack(X_list)


def test_inference_latency_p95_under_150ms() -> None:
    """p95 per-image ensemble latency must stay under the 150ms budget."""
    from python_services.frequency_guard.api.inference import InferenceService

    service = InferenceService(_settings())
    service.ensure_model()
    X = _feature_matrix(BATCH)

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        service._ensemble.predict_proba(X)  # type: ignore[union-attr]
        times.append((time.perf_counter() - t0) * 1000.0 / X.shape[0])
    measured_p95 = float(np.percentile(times, 95))
    assert measured_p95 < LATENCY_BUDGET_MS, f"p95 {measured_p95:.1f}ms > budget {LATENCY_BUDGET_MS}ms"


def test_full_pipeline_latency_per_image_under_150ms() -> None:
    """End-to-end per-image pipeline (load→preprocess→extract) under budget."""
    rel = "fake/fake_0005.png"
    # Warm cache so we measure steady-state throughput.
    _feature_matrix(BATCH)

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        loaded = load_image_file(DEMO_DIR / rel, _settings())
        prep = preprocess(loaded.bgr, _settings())
        extract_features(prep, _settings())
        times.append((time.perf_counter() - t0) * 1000.0)
    measured_p95 = float(np.percentile(times, 95))
    assert (
        measured_p95 < LATENCY_BUDGET_MS
    ), f"pipeline p95 {measured_p95:.1f}ms > budget {LATENCY_BUDGET_MS}ms"


def test_peak_rss_under_2gb() -> None:
    """Peak RSS after feature extraction on a small batch must stay <2GB."""
    _feature_matrix(16)
    rss_mb = current_peak_rss_mb()
    assert rss_mb < MEMORY_BUDGET_MB, f"peak RSS {rss_mb:.0f}MB > budget {MEMORY_BUDGET_MB}MB"
