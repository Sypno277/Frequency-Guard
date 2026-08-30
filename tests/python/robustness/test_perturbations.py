"""Robustness / adversarial tests (Masterplan §6).

Asserts detection stability under JPEG compression, resizing, cropping,
mild noise, and screenshot-style re-encode. Verdict direction must remain
stable for clearly-labeled demo images; feature vectors may drift but the
verdict should not flip.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from python_services.frequency_guard.config import Settings
from python_services.frequency_guard.features.extractor import extract_features
from python_services.frequency_guard.io.loader import load_image_file
from python_services.frequency_guard.preprocess import preprocess

pytestmark = pytest.mark.robustness

DEMO_DIR = Path(__file__).resolve().parents[3] / "data" / "demo_dataset"

JPEG_QUALITIES = [30, 60, 95]


def _settings() -> Settings:
    # Must match the defaults used to train the model (89-feature vector).
    return Settings()


def _load_bgr(rel: str) -> np.ndarray:
    loaded = load_image_file(DEMO_DIR / rel, Settings())
    return loaded.bgr


def _predict_label(bgr: np.ndarray, model_dir: Path) -> tuple[int, float]:
    """Run preprocess→features→model; returns (predicted_label, fake_prob)."""
    from python_services.frequency_guard.api.inference import InferenceService

    service = InferenceService(Settings(model_dir=model_dir))
    service.ensure_model()
    prep = preprocess(bgr, _settings())
    bundle = extract_features(prep, _settings())
    vec = bundle.vector.reshape(1, -1)
    raw = float(service._ensemble.predict_proba(vec)[0, 1])
    calib = float(np.clip(service._calibrator.predict_proba(np.asarray([raw]))[0], 0, 1))
    return (1 if calib >= service._threshold else 0), calib


def _jpeg_encode(bgr: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    assert decoded is not None
    return decoded


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[3] / "checkpoints" / "ensemble.joblib").exists(),
    reason="Run scripts/train_demo.py first",
)
class TestVerdictStability:
    """Verdict direction must survive degradations on clear-cut samples."""

    @pytest.fixture(autouse=True)
    def _model(self):
        self.model_dir = Path(__file__).resolve().parents[3] / "checkpoints"
        # Warm-load once per class instance
        from python_services.frequency_guard.api.inference import InferenceService

        self._svc = InferenceService(Settings(model_dir=self.model_dir))
        self._svc.ensure_model()

    def test_real_survives_jpeg(self) -> None:
        bgr = _load_bgr("real/real_0000.png")
        base_label, _ = self._predict(bgr)
        for q in JPEG_QUALITIES:
            label, _prob = self._predict(_jpeg_encode(bgr, q))
            assert label == base_label == 0, f"real verdict flipped at JPEG q={q}"

    def test_fake_survives_jpeg(self) -> None:
        bgr = _load_bgr("fake/fake_0000.png")
        base_label, _ = self._predict(bgr)
        for q in JPEG_QUALITIES:
            label, _prob = self._predict(_jpeg_encode(bgr, q))
            assert label == base_label == 1, f"fake verdict flipped at JPEG q={q}"

    def test_verdict_stable_under_resize_updown(self) -> None:
        bgr = _load_bgr("fake/fake_0001.png")
        up = cv2.resize(bgr, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        down = cv2.resize(up, None, fx=2 / 3, fy=2 / 3, interpolation=cv2.INTER_AREA)
        label, _ = self._predict(down)
        assert label == 1

    def test_verdict_stable_under_center_crop(self) -> None:
        bgr = _load_bgr("fake/fake_0002.png")
        h, w = bgr.shape[:2]
        crop = bgr[h // 8 : -h // 8 or None, w // 8 : -w // 8 or None]
        label, _ = self._predict(crop)
        assert label == 1

    def test_mild_noise_does_not_flip_real(self) -> None:
        rng = np.random.default_rng(42)
        bgr = _load_bgr("real/real_0003.png").astype(np.float32)
        noisy = np.clip(bgr + rng.normal(0, 4, bgr.shape), 0, 255).astype(np.uint8)
        label, _ = self._predict(noisy)
        assert label == 0

    def _predict(self, bgr: np.ndarray) -> tuple[int, float]:
        prep = preprocess(bgr, _settings())
        bundle = extract_features(prep, _settings())
        vec = bundle.vector.reshape(1, -1)
        raw = float(self._svc._ensemble.predict_proba(vec)[0, 1])
        calib = float(np.clip(self._svc._calibrator.predict_proba(np.asarray([raw]))[0], 0, 1))
        return (1 if calib >= self._svc._threshold else 0), calib
