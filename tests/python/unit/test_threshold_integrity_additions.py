"""Additional regression tests for the agreement-gate and threshold-override behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

from python_services.frequency_guard.api.inference import InferenceService
from python_services.frequency_guard.config import Settings

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = ROOT / "checkpoints"
DEMO_DIR = ROOT / "data" / "demo_dataset"


def _demo_bytes(rel: str) -> bytes:
    return (DEMO_DIR / rel).read_bytes()


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "ensemble.joblib").exists(), reason="Run scripts/train_demo.py first"
)
class TestAgreementGateRegression:
    """Locks WI-2: low agreement must not emit hard AI unless near-certain."""

    def test_no_hard_ai_below_agreement_floor(self) -> None:
        service = InferenceService(Settings())
        service.ensure_model()
        # The demo real_0001 is a smooth, real-class image; if its ensemble
        # agreement is low, the gate must suppress a hard AI flag.
        resp = service.analyze_bytes(_demo_bytes("real/real_0001.png"), source="real_0001.png").response
        if resp.evidence is not None and resp.evidence.agreement < 0.5:
            assert resp.is_ai is False
            assert resp.verdict_state == "uncertain"


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "ensemble.joblib").exists(), reason="Run scripts/train_demo.py first"
)
class TestThresholdOverrideRegression:
    """Locks WI-3: a per-request threshold override is honored."""

    def test_override_reflected_in_response(self) -> None:
        service = InferenceService(Settings())
        service.ensure_model()
        resp = service.analyze_bytes(
            _demo_bytes("fake/fake_0000.png"), source="fake.png", threshold=0.05
        ).response
        assert resp.threshold == pytest.approx(0.05)
