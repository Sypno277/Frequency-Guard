"""Unit tests for the inference service lifecycle and analysis pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from python_services.frequency_guard.api.inference import (
    InferenceOutcome,
    InferenceService,
    ModelNotReadyError,
)
from python_services.frequency_guard.config import Settings
from python_services.frequency_guard.io.loader import LoadError

ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = ROOT / "data" / "demo_dataset"
CHECKPOINT_DIR = ROOT / "checkpoints"


def _demo_bytes(rel: str) -> bytes:
    return (DEMO_DIR / rel).read_bytes()


@pytest.fixture(scope="module")
def service() -> InferenceService:
    if not (CHECKPOINT_DIR / "ensemble.joblib").exists():
        pytest.skip("Run scripts/train_demo.py first")
    settings = Settings()
    svc = InferenceService(settings)
    svc.ensure_model()
    return svc


class TestModelLifecycle:
    def test_not_ready_without_artifacts(self, tmp_path) -> None:
        svc = InferenceService(Settings(model_dir=tmp_path / "empty"))
        assert svc.model_ready is False
        with pytest.raises(ModelNotReadyError):
            svc.ensure_model()

    def test_ensure_model_loads_artifacts(self, service: InferenceService) -> None:
        assert service.model_ready is True
        assert service._ensemble is not None
        assert service._calibrator is not None
        assert service._attributor is not None
        assert 0.0 < service._threshold <= 1.0

    def test_reload_reloads_from_disk(self, service: InferenceService) -> None:
        before = service._threshold
        service.reload()
        assert service.model_ready is True
        assert service._threshold == before

    def test_set_training_metadata(self, service: InferenceService) -> None:
        service.set_training_metadata(
            training_mode="manifest",
            trained_at="2026-01-01T00:00:00Z",
            metrics={"accuracy": 0.99, "f1": 0.98},
            threshold=0.42,
        )
        assert service._training_mode == "manifest"
        assert service._trained_at == "2026-01-01T00:00:00Z"
        assert service._metrics == {"accuracy": 0.99, "f1": 0.98}
        assert service._threshold == 0.42
        # restore demo state for other tests in this module
        service.reload()

    def test_resolve_threshold_prefers_perf_report(self) -> None:
        from python_services.frequency_guard.api.inference import InferenceService as Svc

        # 0.37 is a usable threshold (< 0.5), so it passes through unchanged.
        assert Svc._resolve_threshold(Path("missing.json"), {"threshold": 0.37}) == 0.37

    def test_resolve_threshold_falls_back_to_report_file(self, tmp_path: Path) -> None:
        report = tmp_path / "report.json"
        report.write_text('{"threshold": 0.61}')
        # A persisted threshold >= 0.5 would make is_ai never true (the reported
        # "Authentic@high-confidence" bug); clamp_threshold rejects it.
        result = InferenceService._resolve_threshold(report, None)
        assert result == pytest.approx(0.5)

    def test_resolve_threshold_defaults_half(self, tmp_path: Path) -> None:
        report = tmp_path / "report.json"
        report.write_text("{}")
        assert InferenceService._resolve_threshold(report, None) == 0.5

    def test_resolve_threshold_rejects_near_unity(self) -> None:
        # A stale threshold of 1.0 must never silently survive; clamp to 0.5.
        result = InferenceService._resolve_threshold(Path("missing.json"), {"threshold": 1.0})
        assert result == pytest.approx(0.5)

    def test_resolve_threshold_clamps_low_to_min(self, tmp_path: Path) -> None:
        report = tmp_path / "report.json"
        report.write_text('{"threshold": 0.001}')
        result = InferenceService._resolve_threshold(report, None)
        assert result == pytest.approx(0.05)


class TestAnalyzeBytes:
    def test_fake_image_full_payload(self, service: InferenceService) -> None:
        outcome = service.analyze_bytes(_demo_bytes("fake/fake_0000.png"), source="fake.png")
        assert isinstance(outcome, InferenceOutcome)
        resp = outcome.response
        assert resp.is_ai is True
        assert 0.0 <= resp.fake_probability <= 1.0
        assert 0.0 <= resp.confidence <= 100.0
        assert len(resp.families) == 4
        assert abs(sum(f.probability for f in resp.families) - 1.0) < 1e-3
        assert len(resp.spectrum) > 0
        assert len(resp.azimuthal) > 0
        assert len(resp.wavelet_bands) > 0
        assert resp.explainability.saliency_png_base64.startswith("data:image/png;base64,")
        assert resp.sha256 and len(resp.sha256) == 64
        assert outcome.latency_ms >= 0.0

    def test_real_image_authentic(self, service: InferenceService) -> None:
        resp = service.analyze_bytes(_demo_bytes("real/real_0000.png"), source="real.png").response
        assert resp.is_ai is False

    def test_corrupt_bytes_raise_load_error(self, service: InferenceService) -> None:
        with pytest.raises(LoadError):
            service.analyze_bytes(b"not-an-image", source="bad.png")

    def test_empty_bytes_raise_load_error(self, service: InferenceService) -> None:
        with pytest.raises(LoadError):
            service.analyze_bytes(b"", source="empty.png")

    def test_deterministic_sha256(self, service: InferenceService) -> None:
        blob = _demo_bytes("fake/fake_0002.png")
        r1 = service.analyze_bytes(blob, source="a.png").response.sha256
        r2 = service.analyze_bytes(blob, source="b.png").response.sha256
        assert r1 == r2

    def test_low_agreement_suppresses_hard_positive(self, service: InferenceService, monkeypatch) -> None:
        """A low-agreement mid-confidence call must NOT emit a hard AI flag."""
        from python_services.frequency_guard.api.schemas import EvidenceBreakdown

        def fake_evidence(_bundle, _m):
            return EvidenceBreakdown(contributions=[], member_probabilities={}, agreement=0.1)

        monkeypatch.setattr(service, "_evidence_breakdown", fake_evidence)
        resp = service.analyze_bytes(_demo_bytes("fake/fake_0000.png"), source="fake.png").response
        # fake_prob is well above 0.5 here, so keep is_ai even at low agreement.
        # To test the suppression path, force a mid calibrated probability.
        monkeypatch.setattr(
            service,
            "_calibrated_probability",
            lambda *a, **k: (0.36, np.zeros(4, dtype=np.float64), False),
        )
        resp = service.analyze_bytes(_demo_bytes("fake/fake_0000.png"), source="fake.png").response
        assert resp.is_ai is False
        assert resp.verdict_state == "uncertain"

    def test_low_agreement_keeps_positive_when_certain(self, service: InferenceService, monkeypatch) -> None:
        """A low-agreement but near-certain (>0.9) call keeps is_ai=True."""
        from python_services.frequency_guard.api.schemas import EvidenceBreakdown

        def fake_evidence(_bundle, _m):
            return EvidenceBreakdown(contributions=[], member_probabilities={}, agreement=0.1)

        monkeypatch.setattr(service, "_evidence_breakdown", fake_evidence)
        monkeypatch.setattr(
            service,
            "_calibrated_probability",
            lambda *a, **k: (0.95, np.zeros(4, dtype=np.float64), False),
        )
        resp = service.analyze_bytes(_demo_bytes("fake/fake_0000.png"), source="fake.png").response
        assert resp.is_ai is True
        assert resp.verdict_state == "ai"

    def test_threshold_override_applied_at_inference(self, service: InferenceService) -> None:
        """A per-call threshold override must be used instead of the persisted one."""
        # With the demo model, fake_0000 scores ~1.0. A very high override
        # (0.9, clamped) keeps is_ai=True; use the real image to test the
        # flip: real_0000 scores ~0.0, so a low override would flip it AI.
        resp_override = service.analyze_bytes(
            _demo_bytes("real/real_0000.png"), source="real.png", threshold=0.05
        ).response
        resp_default = service.analyze_bytes(
            _demo_bytes("real/real_0000.png"), source="real.png"
        ).response
        # The persisted threshold is ~0.08; a 0.05 override is more aggressive.
        # We can't assume the real image flips (its probability may be ~0),
        # but the returned threshold must reflect the override, not the model's.
        assert resp_override.threshold == pytest.approx(0.05)
        assert resp_default.threshold == pytest.approx(service._threshold)


class TestWaveletBandsHelper:
    def test_band_levels_and_names(self, service: InferenceService) -> None:
        from python_services.frequency_guard.api.inference import _wavelet_bands
        from python_services.frequency_guard.features.extractor import extract_features
        from python_services.frequency_guard.io.loader import load_image_bytes
        from python_services.frequency_guard.preprocess import preprocess

        loaded = load_image_bytes(_demo_bytes("fake/fake_0000.png"), "f.png", Settings())
        prep = preprocess(loaded.bgr, Settings())
        bundle = extract_features(prep, Settings())
        bands = _wavelet_bands(bundle)
        assert bands
        names = {b.band for b in bands}
        assert names <= {"LH", "HL", "HH"}
        levels = {b.level for b in bands}
        assert min(levels) == 1
