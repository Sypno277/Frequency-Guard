"""Unit tests for the ensemble classifier and probability calibrator."""

from __future__ import annotations

import numpy as np
import pytest

from python_services.frequency_guard.models.calibration import (
    ProbabilityCalibrator,
    brier_score,
    expected_calibration_error,
)
from python_services.frequency_guard.models.classifier import (
    EnsembleClassifier,
    EnsembleConfig,
    StackedMetaClassifier,
)


def _synthetic_dataset(
    n_real: int = 60, n_fake: int = 60, dim: int = 40, seed: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Two separable clusters: real centered at -1.5, fake centered at +1.5."""
    rng = np.random.default_rng(seed)
    X = np.vstack(
        [
            rng.normal(-1.5, 1.0, (n_real, dim)),
            rng.normal(1.5, 1.0, (n_fake, dim)),
        ]
    )
    y = np.asarray([0] * n_real + [1] * n_fake, dtype=np.int64)
    return X, y


def _small_config() -> EnsembleConfig:
    return EnsembleConfig(
        rf_trees=50,
        gb_estimators=40,
        lr_max_iter=300,
        random_state=0,
        n_jobs=1,
    )


class TestEnsembleClassifier:
    def test_fit_and_predict_separable(self) -> None:
        X, y = _synthetic_dataset()
        model = EnsembleClassifier(_small_config()).fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

        preds = model.predict(X)
        assert (preds == y).mean() > 0.85

    def test_unfitted_raises(self) -> None:
        model = EnsembleClassifier(_small_config())
        with pytest.raises(RuntimeError):
            model.predict_proba(np.zeros((4, 8)))

    def test_fit_validates_shape(self) -> None:
        model = EnsembleClassifier(_small_config())
        with pytest.raises(ValueError):
            model.fit(np.zeros((10,)), np.zeros(10, dtype=np.int64))

    def test_fit_requires_two_classes(self) -> None:
        model = EnsembleClassifier(_small_config())
        with pytest.raises(ValueError):
            model.fit(np.zeros((10, 4)), np.zeros(10, dtype=np.int64))

    def test_member_probabilities_shape(self) -> None:
        X, y = _synthetic_dataset(n_real=30, n_fake=30)
        model = EnsembleClassifier(_small_config()).fit(X, y)
        assert model.member_probabilities(X).shape == (60, 4, 2)

    def test_save_load_roundtrip(self, tmp_path) -> None:
        X, y = _synthetic_dataset(n_real=30, n_fake=30)
        model = EnsembleClassifier(_small_config())
        model.feature_names = tuple(f"f{i}" for i in range(X.shape[1]))
        model.fit(X, y)

        path = tmp_path / "ensemble.joblib"
        model.save(path)
        loaded = EnsembleClassifier.load(path)
        assert loaded.fitted
        assert loaded.feature_names == model.feature_names
        np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X), atol=1e-6)


class TestCalibrationMetrics:
    def test_ece_in_unit_range(self) -> None:
        y = np.asarray([0, 0, 1, 1])
        p = np.asarray([0.1, 0.2, 0.9, 0.8])
        assert 0.0 <= expected_calibration_error(y, p) <= 1.0

    def test_ece_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            expected_calibration_error(np.zeros(3), np.zeros(4))

    def test_brier_perfect_prediction(self) -> None:
        y = np.asarray([0, 1, 1, 0])
        p = np.asarray([0.0, 1.0, 1.0, 0.0])
        assert brier_score(y, p) == 0.0


class TestStackedMetaClassifier:
    def test_unfitted_raises(self) -> None:
        meta = StackedMetaClassifier()
        with pytest.raises(RuntimeError):
            meta.predict_proba(np.zeros((4, 4, 2)))

    def test_rejects_wrong_member_shape(self) -> None:
        meta = StackedMetaClassifier()
        with pytest.raises(ValueError):
            meta.fit(np.zeros((8, 4)), np.zeros(8, dtype=np.int64))

    def test_requires_two_classes(self) -> None:
        meta = StackedMetaClassifier()
        with pytest.raises(ValueError):
            meta.fit(np.zeros((8, 4, 2)), np.zeros(8, dtype=np.int64))

    def test_fit_and_predict_separable(self) -> None:
        X, y = _synthetic_dataset(n_real=40, n_fake=40)
        model = EnsembleClassifier(_small_config()).fit(X, y)
        member = model.member_probabilities(X)
        # Clear signal: fake samples should map to higher OOF fake probability.
        meta = StackedMetaClassifier().fit(member, y)
        preds = (meta.predict_fake(member) >= 0.5).astype(int)
        assert (preds == y).mean() > 0.85

    def test_save_load_roundtrip(self, tmp_path) -> None:
        X, y = _synthetic_dataset(n_real=30, n_fake=30)
        model = EnsembleClassifier(_small_config()).fit(X, y)
        meta = StackedMetaClassifier().fit(model.member_probabilities(X), y)
        path = tmp_path / "meta.joblib"
        meta.save(path)
        loaded = StackedMetaClassifier.load(path)
        assert loaded.fitted
        np.testing.assert_allclose(
            loaded.predict_proba(model.member_probabilities(X)),
            meta.predict_proba(model.member_probabilities(X)),
            atol=1e-9,
        )


class TestProbabilityCalibrator:
    def test_isotonic_roundtrip(self) -> None:
        rng = np.random.default_rng(1)
        raw = rng.uniform(0.0, 1.0, 200)
        y = (raw > 0.5).astype(np.int64)
        cal = ProbabilityCalibrator("isotonic").fit(raw, y)
        out = cal.predict_proba(raw)
        assert out.shape == raw.shape
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_calibration_improves_ece(self) -> None:
        rng = np.random.default_rng(2)
        raw = np.clip(rng.normal(0.5, 0.2, 400), 0.01, 0.99)
        y = (rng.uniform(0.0, 1.0, 400) < raw).astype(np.int64)
        cal = ProbabilityCalibrator("isotonic").fit(raw, y)
        result = cal.evaluate(raw[:100], y[:100])
        assert result.ece_after <= result.ece_before + 0.05

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError):
            ProbabilityCalibrator("quantile")

    def test_unfitted_raises(self) -> None:
        with pytest.raises(RuntimeError):
            ProbabilityCalibrator().predict_proba(np.asarray([0.5]))

    def test_save_load_roundtrip(self, tmp_path) -> None:
        raw = np.linspace(0.1, 0.9, 100)
        y = (raw > 0.5).astype(np.int64)
        cal = ProbabilityCalibrator("isotonic").fit(raw, y)
        path = tmp_path / "calibrator.joblib"
        cal.save(path)
        loaded = ProbabilityCalibrator.load(path)
        np.testing.assert_allclose(loaded.predict_proba(raw), cal.predict_proba(raw), atol=1e-3)
