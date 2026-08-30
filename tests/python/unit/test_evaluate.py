"""Unit tests for the evaluation harness (metrics, thresholds, latency, RSS)."""

from __future__ import annotations

import numpy as np
import pytest

from python_services.frequency_guard.config import Settings
from python_services.frequency_guard.evaluation.evaluate import (
    EvaluationReport,
    current_peak_rss_mb,
    evaluate_model,
    find_fpr_threshold,
    measure_latency,
    persist_report,
    resolution_buckets,
)


def _synthetic_probs(n: int = 200, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Well-separated fake probabilities."""
    rng = np.random.default_rng(seed)
    y = np.asarray([0] * (n // 2) + [1] * (n // 2))
    fake = np.clip(rng.normal(0.85, 0.08, n // 2), 0.01, 0.99)
    real = np.clip(rng.normal(0.15, 0.08, n // 2), 0.01, 0.99)
    p = np.empty(n, dtype=np.float64)
    p[y == 1] = fake
    p[y == 0] = real
    return y, p


class TestFindFprThreshold:
    def test_hits_target_fpr(self) -> None:
        y, p = _synthetic_probs()
        t = find_fpr_threshold(y, p, target_fpr=0.02)
        preds = (p >= t).astype(int)
        fpr = float(np.mean(preds[y == 0] == 1))
        assert fpr <= 0.02

    def test_no_real_samples_returns_first_candidate(self) -> None:
        """When no real samples exist, FPR is always 0 so the function
        returns the first (lowest) candidate threshold that satisfies it."""
        t = find_fpr_threshold(np.ones(5), np.linspace(0.2, 0.9, 5), target_fpr=0.02)
        assert t == pytest.approx(0.2)

    def test_logs_working_point(self, caplog) -> None:
        """The chosen threshold and achieved FPR/TPR must be logged for audit."""
        import logging

        y, p = _synthetic_probs(80)
        with caplog.at_level(logging.INFO, logger="python_services.frequency_guard.evaluation.evaluate"):
            find_fpr_threshold(y, p, target_fpr=0.02)
        assert any("threshold_working_point" in r.message for r in caplog.records)
        # The structured fields should include the chosen threshold and rates.
        assert any(
            "threshold" in r.__dict__.get("fields", {}) and "achieved_fpr" in r.__dict__.get("fields", {})
            for r in caplog.records
        )


class TestMeasureLatency:
    def test_percentile_keys(self) -> None:
        fn = lambda X: X * 2.0  # noqa: E731
        out = measure_latency(fn, np.zeros((10, 4)), repeats=2)
        assert set(out.keys()) == {"p50", "p95", "p99"}
        assert all(v >= 0.0 for v in out.values())


class TestPeakRss:
    def test_positive_or_zero_float(self) -> None:
        value = current_peak_rss_mb()
        assert isinstance(value, float)
        assert value >= 0.0


class TestEvaluateModel:
    def test_report_fields_populated(self) -> None:
        settings = Settings()
        y, p = _synthetic_probs()
        report = evaluate_model(
            y_true=y,
            raw_fake_prob=p,
            calibrated_fake_prob=p,
            settings=settings,
            predict_fn=lambda M: (M[:, 0] > 0).astype(int),
            generators=np.asarray(["cam"] * 100 + ["gan"] * 100),
            latency_probe=np.zeros((4, 8)),
        )
        assert isinstance(report, EvaluationReport)
        assert report.accuracy > 0.9
        assert report.f1 > 0.9
        assert report.fpr_at_threshold <= settings.threshold_fpr_target + 1e-6
        assert len(report.confusion_matrix) == 2
        assert "fpr" in report.roc_points and "tpr" in report.roc_points
        assert "precision" in report.pr_points and "recall" in report.pr_points
        assert report.n_samples == 200
        # per-generator breakdown present
        assert set(report.per_generator.keys()) == {"cam", "gan"}
        assert report.per_generator["gan"]["accuracy"] > 0.9

    def test_single_class_roc_auc_guarded(self) -> None:
        """roc_auc must not crash on single-class input (returns 0.5 sentinel)."""
        y = np.ones(20, dtype=int)
        p = np.full(20, 0.9)
        report = evaluate_model(y_true=y, raw_fake_prob=p, calibrated_fake_prob=p, settings=Settings())
        assert report.roc_auc == 0.5

    def test_to_dict_roundtrip(self) -> None:
        y, p = _synthetic_probs(60)
        report = evaluate_model(y_true=y, raw_fake_prob=p, calibrated_fake_prob=p, settings=Settings())
        d = report.to_dict()
        assert d["accuracy"] == report.accuracy
        assert "generated_at" in d


class TestResolutionSlices:
    def test_resolution_buckets_labels(self) -> None:
        widths = [200, 400, 800, 1200]
        heights = [200, 300, 600, 1600]
        assert resolution_buckets(widths, heights) == ["small", "small", "medium", "large"]

    def test_evaluate_populates_per_resolution(self) -> None:
        settings = Settings()
        y, p = _synthetic_probs(80)
        # Buckets key off min(w, h): <512 small, 512..1023 medium, >=1024 large.
        # (1500, 1000) has min-dim 1000 → 'medium', so the large bucket needs a
        # genuinely large image here.
        sizes = [(300, 300)] * 20 + [(800, 600)] * 20 + [(400, 400)] * 20 + [(2000, 1200)] * 20
        report = evaluate_model(
            y_true=y,
            raw_fake_prob=p,
            calibrated_fake_prob=p,
            settings=settings,
            image_sizes=sizes,
        )
        assert set(report.per_resolution.keys()) == {"small", "medium", "large"}
        for entry in report.per_resolution.values():
            assert "n" in entry and "accuracy" in entry and "f1" in entry
        assert report.per_resolution["large"]["n"] == 20


class TestPersistReport:
    def test_writes_benchmark_and_history(self, tmp_path) -> None:
        reports_dir = tmp_path / "reports"
        settings = Settings(reports_dir=reports_dir)
        y, p = _synthetic_probs(80)

        report = evaluate_model(y_true=y, raw_fake_prob=p, calibrated_fake_prob=p, settings=settings)
        path = persist_report(report, settings)

        assert path.exists()
        assert (reports_dir / "metrics_history.json").exists()
        history = __import__("json").loads((reports_dir / "metrics_history.json").read_text())
        assert len(history) == 1
        # slim entries drop curve points but keep headline metrics
        assert "accuracy" in history[0]
        assert "roc_points" not in history[0]

    def test_history_corrupt_file_recovered(self, tmp_path) -> None:
        """A corrupt metrics_history.json must be reset rather than crash."""
        import json as jsonlib

        reports_dir = tmp_path / "reports"
        reports_dir.mkdir(parents=True)
        (reports_dir / "metrics_history.json").write_text("not-json{{{")

        settings = Settings(reports_dir=reports_dir)
        y, p = _synthetic_probs(40)
        report = evaluate_model(y_true=y, raw_fake_prob=p, calibrated_fake_prob=p, settings=settings)
        persist_report(report, settings)
        history = jsonlib.loads((reports_dir / "metrics_history.json").read_text())
        assert len(history) == 1
