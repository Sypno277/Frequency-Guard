"""Unit tests for the training pipeline (manifest → features → CV → ensemble → calibration)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from python_services.frequency_guard.config import Settings
from python_services.frequency_guard.models.classifier import EnsembleConfig
from python_services.frequency_guard.training.train_pipeline import (
    TrainResult,
    _cross_validate,
    _extract_vectors,
    _load_manifest,
    lomo_holdout,
    train_from_manifest,
)


def _write_image(path: Path, fake: bool, seed: int) -> None:
    """Write a small deterministic image (smooth for real, checker for fake)."""
    rng = np.random.default_rng(seed)
    size = 64
    yy, xx = np.mgrid[0:size, 0:size]
    if fake:
        base = 0.35 * np.sin(xx * np.pi / 2) * np.sin(yy * np.pi / 2)
        img = np.clip((base + 0.5) * 255 + rng.normal(0, 1, (size, size)), 0, 255).astype(np.uint8)
    else:
        base = 0.3 * np.sin(xx / 8.0) * np.cos(yy / 10.0)
        img = np.clip((base + 0.5) * 255 + rng.normal(0, 3, (size, size)), 0, 255).astype(np.uint8)
    bgr = np.stack([img, img, img], axis=-1)
    ok, buf = cv2.imencode(".png", bgr)
    assert ok
    path.write_bytes(buf.tobytes())


def _make_manifest(tmp_path: Path, n_real: int = 4, n_fake: int = 4) -> Path:
    """Build a small manifest of synthetic images; returns the CSV path."""
    rows: list[tuple[str, int]] = []
    for i in range(n_real):
        p = tmp_path / f"real_{i}.png"
        _write_image(p, fake=False, seed=100 + i)
        rows.append((str(p), 0))
    for i in range(n_fake):
        p = tmp_path / f"fake_{i}.png"
        _write_image(p, fake=True, seed=200 + i)
        rows.append((str(p), 1))

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows, columns=["image_path", "label"]).to_csv(manifest, index=False)
    return manifest


def _small_settings(tmp_path: Path) -> Settings:
    """Fast settings + a scratch output dir to avoid touching the repo model dir."""
    return Settings(
        image_size=64,
        dct_block_size=8,
        wavelet_name="db4",
        wavelet_levels=2,
        radial_bins=8,
        azimuthal_bins=6,
        tiles_per_side=2,
        n_folds=2,
        model_dir=tmp_path / "out_models",
        reports_dir=tmp_path / "out_reports",
        data_dir=tmp_path / "out_data",
    )


def _small_config() -> EnsembleConfig:
    return EnsembleConfig(
        rf_trees=10,
        gb_estimators=8,
        lr_max_iter=200,
        svm_c=1.0,
        random_state=0,
        n_jobs=1,
    )


class TestLoadManifest:
    def test_loads_valid_manifest(self, tmp_path) -> None:
        manifest = _make_manifest(tmp_path)
        df = _load_manifest(manifest)
        assert len(df) == 8
        assert set(df["label"].unique()) == {0, 1}

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            _load_manifest(tmp_path / "nope.csv")

    def test_missing_columns_raises(self, tmp_path) -> None:
        csv = tmp_path / "bad.csv"
        pd.DataFrame({"foo": [1, 2]}).to_csv(csv, index=False)
        with pytest.raises(ValueError):
            _load_manifest(csv)

    def test_invalid_labels_raises(self, tmp_path) -> None:
        csv = tmp_path / "bad_labels.csv"
        pd.DataFrame({"image_path": ["a.png", "b.png"], "label": [0, 2]}).to_csv(csv, index=False)
        with pytest.raises(ValueError):
            _load_manifest(csv)


class TestExtractVectors:
    def test_returns_vectors_and_labels(self, tmp_path) -> None:
        manifest = _make_manifest(tmp_path, n_real=3, n_fake=3)
        settings = _small_settings(tmp_path)
        df = _load_manifest(manifest)
        X, y, names = _extract_vectors(df, settings)
        assert X.shape == (6, len(names))
        assert y.shape == (6,)
        assert sorted(y.tolist()) == [0, 0, 0, 1, 1, 1]
        assert len(names) == X.shape[1]

    def test_degrade_emits_dual_vectors(self, tmp_path) -> None:
        """With degrade=True, each image yields a pristine + augmented row."""
        manifest = _make_manifest(tmp_path, n_real=3, n_fake=3)
        settings = _small_settings(tmp_path)
        df = _load_manifest(manifest)
        X, y, names = _extract_vectors(df, settings, degrade=True)
        assert X.shape == (12, len(names))
        assert y.shape == (12,)
        assert sorted(y.tolist()) == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        # The pristine and augmented halves must not be identical vectors.
        assert not np.allclose(X[0], X[6])

    def test_degrade_default_is_pristine(self, tmp_path) -> None:
        """Without degrade, behavior is unchanged (single vector per image)."""
        manifest = _make_manifest(tmp_path, n_real=2, n_fake=2)
        settings = _small_settings(tmp_path)
        df = _load_manifest(manifest)
        X, y, names = _extract_vectors(df, settings)
        assert X.shape == (4, len(names))

    def test_cache_key_differs_by_seed(self, tmp_path) -> None:
        """Augmented features must not alias pristine ones in the cache."""
        from python_services.frequency_guard.features.extractor import cache_key
        from python_services.frequency_guard.io.loader import load_image_file
        from python_services.frequency_guard.preprocess import preprocess

        manifest = _make_manifest(tmp_path, n_real=1, n_fake=1)
        settings = _small_settings(tmp_path)
        df = _load_manifest(manifest)
        row = df.iloc[0]
        loaded = load_image_file(row["image_path"], settings)
        prep = preprocess(loaded.bgr, settings)
        pristine_key = cache_key(str(row["image_path"]), prep, settings)
        aug_key = cache_key(f"{row['image_path']}::aug::{123}", prep, settings)
        assert pristine_key != aug_key


class TestCrossValidate:
    def test_returns_metrics_oof_and_member_probs(self, tmp_path) -> None:
        manifest = _make_manifest(tmp_path, n_real=6, n_fake=6)
        settings = _small_settings(tmp_path)
        X, y, _names = _extract_vectors(_load_manifest(manifest), settings)
        metrics, oof, oof_members = _cross_validate(X, y, settings, _small_config())
        assert oof.shape == (12,)
        assert oof_members.shape == (12, 4, 2)
        assert np.allclose(oof_members[:, :, 1].mean(axis=1), oof, atol=1e-9)
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.roc_auc <= 1.0
        assert metrics.n_samples == 12
        assert metrics.n_real == 6
        assert metrics.n_fake == 6


class TestLomoHoldout:
    def _generator_manifest(self, tmp_path: Path) -> Path:
        """Manifest with two generator families: 'gan' and 'diffusion'."""
        rows: list[tuple[str, int, str]] = []
        for i in range(4):
            p1 = tmp_path / f"g{i}.png"
            _write_image(p1, fake=True, seed=300 + i)
            rows.append((str(p1), 1, "gan"))
            p2 = tmp_path / f"d{i}.png"
            _write_image(p2, fake=True, seed=400 + i)
            rows.append((str(p2), 1, "diffusion"))
            p3 = tmp_path / f"r{i}.png"
            _write_image(p3, fake=False, seed=500 + i)
            rows.append((str(p3), 0, "camera"))
        manifest = tmp_path / "manifest_gen.csv"
        pd.DataFrame(rows, columns=["image_path", "label", "generator"]).to_csv(manifest, index=False)
        return manifest

    def test_returns_report_when_generator_column_present(self, tmp_path) -> None:
        manifest = self._generator_manifest(tmp_path)
        settings = _small_settings(tmp_path)
        df = _load_manifest(manifest)
        X, y, _names = _extract_vectors(df, settings)
        report = lomo_holdout(df, X, y, settings, _small_config())
        # 'camera' is the only real class, so holding it out leaves a one-class
        # training fold and is correctly skipped; the two fake families remain.
        assert set(report.keys()) == {"gan", "diffusion"}
        for _key, entry in report.items():
            assert "n" in entry and "accuracy" in entry and "roc_auc" in entry

    def test_returns_empty_without_generator_column(self, tmp_path) -> None:
        manifest = _make_manifest(tmp_path, n_real=4, n_fake=4)
        settings = _small_settings(tmp_path)
        df = _load_manifest(manifest)
        X, y, _names = _extract_vectors(df, settings)
        assert lomo_holdout(df, X, y, settings, _small_config()) == {}


class TestTrainFromManifest:
    def test_trains_and_persists_artifacts(self, tmp_path) -> None:
        manifest = _make_manifest(tmp_path, n_real=5, n_fake=5)
        settings = _small_settings(tmp_path)
        out_dir = tmp_path / "artifacts"

        result = train_from_manifest(manifest, out_dir, settings=settings, ensemble_config=_small_config())

        assert isinstance(result, TrainResult)
        assert result.ensemble.fitted
        assert result.calibrator is not None
        assert result.metrics.n_samples == 10
        assert len(result.feature_names) > 0
        assert result.report_path.exists()
        assert (out_dir / "ensemble.joblib").exists()
        assert (out_dir / "calibrator.joblib").exists()
        assert (out_dir / "meta_learner.joblib").exists()

        # report must be valid JSON with metrics
        import json

        payload = json.loads(result.report_path.read_text(encoding="utf-8"))
        assert payload["metrics"]["accuracy"] >= 0.0
        assert payload["metrics"]["n_samples"] == 10

    def test_trainresult_to_dict_shape(self, tmp_path) -> None:
        manifest = _make_manifest(tmp_path, n_real=4, n_fake=4)
        settings = _small_settings(tmp_path)
        result = train_from_manifest(
            manifest, tmp_path / "artifacts2", settings=settings, ensemble_config=_small_config()
        )
        d = result.to_dict()
        assert d["metrics"]["accuracy"] >= 0.0
        assert d["metrics"]["n_samples"] == 8
        assert "train_report" in d
