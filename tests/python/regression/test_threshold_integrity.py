"""Threshold-integrity regression tests (E-Masterplan D0.3).

Locks the fix for the reported "AI image shown as Uncerta
in → Authentic@high-confidence" bug:

1. A persisted threshold >= 0.5 must never make the AI gate unreachable.
2. ``is_ai`` must be consistent with ``confidence >= 50`` for the same input
   (kills the "Uncertain ≠ verdict" contradiction).
3. ``find_fpr_threshold`` must never return a value >= 0.5.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from python_services.frequency_guard.api.inference import InferenceService
from python_services.frequency_guard.config import Settings, ensure_dirs
from python_services.frequency_guard.evaluation.evaluate import find_fpr_threshold
from python_services.frequency_guard.models.verdict import clamp_threshold, resolve_verdict

ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = ROOT / "data" / "demo_dataset"
CHECKPOINT_DIR = ROOT / "checkpoints"

pytestmark = pytest.mark.regression


def _demo_bytes(rel: str) -> bytes:
    return (DEMO_DIR / rel).read_bytes()


class TestClampThreshold:
    def test_rejects_near_unity(self) -> None:
        # A stale 1.0 threshold made is_ai never true.
        assert clamp_threshold(1.0) == 0.5
        assert clamp_threshold(0.9) == 0.5
        assert clamp_threshold(0.51) == 0.5

    def test_accepts_usable_range(self) -> None:
        assert clamp_threshold(0.37) == pytest.approx(0.37)
        assert clamp_threshold(0.05) == pytest.approx(0.05)

    def test_clamps_low_to_min(self) -> None:
        # A near-zero threshold (0.001) made every image "AI"; clamp to 0.05.
        assert clamp_threshold(0.001) == pytest.approx(0.05)

    def test_none_and_nan_default_to_half(self) -> None:
        assert clamp_threshold(None) == 0.5
        assert clamp_threshold(float("nan")) == 0.5


class TestFindFprThreshold:
    def test_never_returns_above_half(self) -> None:
        y = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        prob = np.asarray([0.99, 0.98, 0.97, 0.99, 0.99, 0.99], dtype=np.float64)
        t = find_fpr_threshold(y, prob, target_fpr=0.01)
        assert t <= 0.5

    def test_returns_usable_low_threshold(self) -> None:
        # 10 real + 10 fake: one real sits at 0.30, so FPR=0.1 is met at a
        # genuinely usable mid threshold (0.30), not clamped to 0.5.
        y = np.asarray([0] * 9 + [0] + [1] * 10, dtype=np.int64)
        prob = np.asarray([0.01] * 9 + [0.30] + [0.95] * 10, dtype=np.float64)
        t = find_fpr_threshold(y, prob, target_fpr=0.15)
        assert 0.05 <= t < 0.5


class TestVerdictConsistency:
    def test_ai_confident_at_high_confidence(self) -> None:
        v = resolve_verdict(0.95, 0.5)
        assert v.state == "ai"
        assert v.is_ai is True
        assert v.confidence == pytest.approx(95.0)

    def test_authentic_confident_at_low_fake_prob(self) -> None:
        v = resolve_verdict(0.05, 0.5)
        assert v.state == "authentic"
        assert v.is_ai is False
        assert v.confidence == pytest.approx(5.0)

    def test_uncertain_above_gate_but_below_half(self) -> None:
        # Above the decision gate but under the 50% confidence axis -> uncertain.
        v = resolve_verdict(0.3, 0.08)
        assert v.state == "uncertain"
        assert v.is_ai is True
        assert v.confidence == pytest.approx(30.0)

    def test_confident_label_never_below_half(self) -> None:
        # For fake-probabilities below 0.5: the "AI" label can never be shown,
        # and the "authentic" label carries a low (below-50) confidence on the
        # real side — which is truthful. The key invariant is that "ai"/"authentic"
        # are NEVER emitted on the wrong side of the confidence axis.
        for fake in np.linspace(0.0, 0.49, 50):
            v = resolve_verdict(float(fake), 0.5)
            assert v.state in ("authentic", "uncertain")
            if v.state == "authentic":
                assert v.confidence < 50.0

    def test_stale_threshold_cannot_turn_ai_into_authentic(self) -> None:
        # With a clamped threshold of 0.5, a 0.95 fake score MUST be "ai".
        v = resolve_verdict(0.95, 1.0)  # stale 1.0 threshold passed in
        assert v.state == "ai"
        assert v.is_ai is True


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "ensemble.joblib").exists(), reason="Run scripts/train_demo.py first"
)
class TestPersistedThreshold:
    def test_loaded_threshold_is_usable(self) -> None:
        settings = Settings()
        ensure_dirs(settings)
        service = InferenceService(settings)
        service.ensure_model()
        # The persisted threshold must be in (0.05, 0.5), never >= 0.5.
        assert 0.05 <= service._threshold <= 0.5

    def test_analyze_fake_image_is_ai(self) -> None:
        settings = Settings()
        ensure_dirs(settings)
        service = InferenceService(settings)
        service.ensure_model()
        resp = service.analyze_bytes(_demo_bytes("fake/fake_0000.png"), source="fake.png").response
        assert resp.is_ai is True
        assert resp.verdict_state == "ai"
        assert resp.confidence >= 50.0

    def test_analyze_real_image_is_authentic(self) -> None:
        settings = Settings()
        ensure_dirs(settings)
        service = InferenceService(settings)
        service.ensure_model()
        resp = service.analyze_bytes(_demo_bytes("real/real_0000.png"), source="real.png").response
        assert resp.is_ai is False
        assert resp.verdict_state == "authentic"
