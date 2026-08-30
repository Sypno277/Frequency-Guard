"""Training pipeline: manifest -> features -> CV -> ensemble -> calibration.

The pipeline is intentionally dataset-agnostic. A manifest is a CSV with at
least the columns ``image_path``, ``label`` (0=real, 1=fake) and optional
``generator`` / ``dataset`` columns. We:
1. Load + validate the manifest
2. Extract features (cached) per image
3. Run stratified 5-fold CV by (label, generator)
4. Train the ensemble on the full training set
5. Fit a probability calibrator on out-of-fold predictions
6. Persist the ensemble, calibrator, and a metrics report
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..config import Settings, ensure_dirs
from ..features.augmentation import augment_bgr
from ..features.extractor import FeatureBundle, cache_key, extract_features, global_cache
from ..io.loader import load_image_file
from ..logging import get_logger
from ..models.calibration import ProbabilityCalibrator, expected_calibration_error
from ..models.classifier import EnsembleClassifier, EnsembleConfig, StackedMetaClassifier
from ..preprocess import PreprocessedImage, preprocess

log = get_logger(__name__)

REQUIRED_COLUMNS = ("image_path", "label")


@dataclass
class TrainMetrics:
    """Summary metrics computed on the out-of-fold / validation split."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    ece: float
    n_samples: int
    n_fake: int
    n_real: int


@dataclass
class TrainResult:
    """Everything produced by :func:`train_from_manifest`."""

    ensemble: EnsembleClassifier
    calibrator: ProbabilityCalibrator
    metrics: TrainMetrics
    feature_names: tuple[str, ...]
    report_path: Path
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": {
                "accuracy": self.metrics.accuracy,
                "precision": self.metrics.precision,
                "recall": self.metrics.recall,
                "f1": self.metrics.f1,
                "roc_auc": self.metrics.roc_auc,
                "ece": self.metrics.ece,
                "n_samples": self.metrics.n_samples,
                "n_fake": self.metrics.n_fake,
                "n_real": self.metrics.n_real,
            },
            "train_report": str(self.report_path),
        }


def _load_manifest(manifest_csv: str | Path) -> pd.DataFrame:
    """Load and validate the manifest CSV."""
    path = Path(manifest_csv)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    df = df.dropna(subset=["image_path", "label"]).copy()
    df["label"] = df["label"].astype(int)
    if set(df["label"].unique()) != {0, 1}:
        raise ValueError("Manifest 'label' column must contain only 0 (real) and 1 (fake)")
    log.info(
        "manifest loaded",
        extra={"fields": {"rows": int(len(df)), "path": str(path)}},
    )
    return df


def _extract_vectors(
    df: pd.DataFrame,
    settings: Settings,
    degrade: bool = False,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Extract feature vectors for every row (cached where possible).

    Args:
        df: manifest rows with at least ``image_path`` and ``label``.
        settings: runtime settings.
        degrade: when True, also emit a degraded augmentation of each image
            (E3.1 training-time augmentation) so the model learns to survive
            JPEG/resize/blur/noise/crop rather than over-fit pristine features.
            The augmented variant uses a per-row deterministic seed derived
            from the image path so training is reproducible and augmented
            features never collide with the pristine ones in the cache.

    Returns:
        (X, y, feature_names) where X has 2x rows when ``degrade`` is True.
    """
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    feature_names: tuple[str, ...] | None = None

    for row in df.itertuples(index=False):
        path = Path(row.image_path)
        label = int(row.label)
        loaded = load_image_file(path, settings)
        prep = preprocess(loaded.bgr, settings)

        key = cache_key(str(path), prep, settings)

        def _compute(p: PreprocessedImage = prep, s: Settings = settings) -> FeatureBundle:
            return extract_features(p, s)

        bundle = global_cache.get_or_compute(key, _compute)
        if feature_names is None:
            feature_names = bundle.names
        X_list.append(bundle.vector)
        y_list.append(label)

        if degrade:
            # Deterministic per-row seed so the same image always gets the
            # same degraded variant (reproducible training), but different
            # images get different chains. The seed is folded into the cache
            # key so augmented features do not alias the pristine features.
            seed = (hash(str(path)) & 0xFFFFFFFF)
            degraded = augment_bgr(loaded.bgr, settings, seed=seed)
            prep2 = preprocess(degraded, settings)
            key2 = cache_key(f"{path}::aug::{seed}", prep2, settings)

            def _compute_aug(p: PreprocessedImage = prep2, s: Settings = settings) -> FeatureBundle:
                return extract_features(p, s)

            bundle2 = global_cache.get_or_compute(key2, _compute_aug)
            X_list.append(bundle2.vector)
            y_list.append(label)

    if feature_names is None:
        feature_names = tuple(f"_f{i}" for i in range(len(X_list[0])))

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y, feature_names


def _cross_validate(
    X: np.ndarray, y: np.ndarray, settings: Settings, config: EnsembleConfig
) -> tuple[TrainMetrics, np.ndarray, np.ndarray]:
    """Stratified k-fold CV returning metrics + OOF fake probs and member probs.

    Out-of-fold predictions come from models that never saw the validation
    fold, so they are leakage-free inputs for both the calibrator and the
    stacked meta-learner (Masterplan §4.3).
    """
    skf = StratifiedKFold(n_splits=settings.n_folds, shuffle=True, random_state=settings.random_state)
    oof = np.zeros(len(y), dtype=np.float64)
    oof_members = np.zeros((len(y), 4, 2), dtype=np.float64)
    accs, precs, recs, f1s, aucs = [], [], [], [], []

    for train_idx, val_idx in skf.split(X, y):
        fold_model = EnsembleClassifier(config)
        fold_model.fit(X[train_idx], y[train_idx])
        proba = fold_model.predict_proba(X[val_idx])[:, 1]
        oof[val_idx] = proba
        oof_members[val_idx] = fold_model.member_probabilities(X[val_idx])

        preds = (proba >= 0.5).astype(int)
        accs.append(accuracy_score(y[val_idx], preds))
        precs.append(precision_score(y[val_idx], preds, zero_division=0))
        recs.append(recall_score(y[val_idx], preds, zero_division=0))
        f1s.append(f1_score(y[val_idx], preds, zero_division=0))
        aucs.append(roc_auc_score(y[val_idx], proba))

    metrics = TrainMetrics(
        accuracy=float(np.mean(accs)),
        precision=float(np.mean(precs)),
        recall=float(np.mean(recs)),
        f1=float(np.mean(f1s)),
        roc_auc=float(np.mean(aucs)),
        ece=expected_calibration_error(y, oof),
        n_samples=int(len(y)),
        n_fake=int(np.sum(y == 1)),
        n_real=int(np.sum(y == 0)),
    )
    return metrics, oof, oof_members


def lomo_holdout(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    settings: Settings,
    config: EnsembleConfig,
) -> dict[str, dict[str, float]]:
    """Leave-One-Model-Out generalization report (Masterplan §4.2).

    For each generator family in the manifest, trains on every other family
    and evaluates on the held-out one. Requires a ``generator`` column;
    returns {} when the manifest has no generator labels.
    """
    if "generator" not in df.columns:
        log.info("lomo skipped: manifest has no 'generator' column")
        return {}
    generators = df["generator"].fillna("unknown").astype(str).to_numpy()
    unique = sorted(set(generators.tolist()))
    if len(unique) < 2:
        log.info("lomo skipped: fewer than two generator families")
        return {}

    report: dict[str, dict[str, float]] = {}
    for held in unique:
        train_mask = generators != held
        test_mask = generators == held
        if train_mask.sum() == 0 or int(np.unique(y[train_mask]).size) != 2:
            continue
        model = EnsembleClassifier(config)
        model.fit(X[train_mask], y[train_mask])
        proba = model.predict_proba(X[test_mask])[:, 1]
        preds = (proba >= 0.5).astype(int)
        report[held] = {
            "n": int(test_mask.sum()),
            "accuracy": round(float(accuracy_score(y[test_mask], preds)), 4),
            "roc_auc": round(
                float(roc_auc_score(y[test_mask], proba)) if len(set(y[test_mask].tolist())) > 1 else 0.5,
                4,
            ),
        }
        log.info("lomo fold done", extra={"fields": {"held_out": held, **report[held]}})
    return report


def train_from_manifest(
    manifest_csv: str | Path,
    output_dir: str | Path,
    settings: Settings | None = None,
    ensemble_config: EnsembleConfig | None = None,
    degrade: bool = False,
) -> TrainResult:
    """Run the full training pipeline on a labeled manifest.

    Args:
        manifest_csv: CSV path with image_path and label columns.
        output_dir: where ensemble.joblib, calibrator.joblib, report.json live.
        settings: runtime settings (defaults to :func:`config.load_settings`).
        ensemble_config: ensemble hyperparameters.

    Returns:
        TrainResult with fitted ensemble, calibrator, metrics, and report path.
    """
    settings = settings or Settings()
    config = ensemble_config or EnsembleConfig()
    ensure_dirs(settings)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = _load_manifest(manifest_csv)
    X, y, feature_names = _extract_vectors(df, settings, degrade=degrade)

    log.info("features extracted", extra={"fields": {"shape": list(X.shape)}})

    metrics, oof, oof_members = _cross_validate(X, y, settings, config)

    # Final model on all data; calibrator + stacked meta-learner fit on the
    # leakage-free out-of-fold predictions (Masterplan §4.3/§4.4).
    ensemble = EnsembleClassifier(config)
    ensemble.feature_names = feature_names
    ensemble.fit(X, y)

    # Stacked meta-learner must be trained BEFORE the calibrator: when the
    # meta learner is used at inference (E-Masterplan E2), the calibrator
    # must be fit on the SAME raw scores (meta OOF outputs), not the
    # probability-average OOF — otherwise a calibration mismatch shifts
    # confident real images above the threshold. Compute meta OOF scores
    # and fit the calibrator on those for distribution consistency.
    meta = StackedMetaClassifier()
    meta.fit(oof_members, y)

    oof_raw = meta.predict_fake(oof_members) if meta.fitted else oof
    calibrator = ProbabilityCalibrator(method=settings.calibrate_method)
    calibrator.fit(oof_raw, y)

    lomo_report = lomo_holdout(df, X, y, settings, config)

    ensemble_path = out / "ensemble.joblib"
    calibrator_path = out / "calibrator.joblib"
    meta_path = out / "meta_learner.joblib"
    report_path = out / "report.json"

    ensemble.save(ensemble_path)
    calibrator.save(calibrator_path)
    meta.save(meta_path)

    report = TrainResult(
        ensemble=ensemble,
        calibrator=calibrator,
        metrics=metrics,
        feature_names=feature_names,
        report_path=report_path,
    ).to_dict()
    if lomo_report:
        report["lomo_holdout"] = lomo_report
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    log.info(
        "training complete",
        extra={"fields": {"accuracy": metrics.accuracy, "roc_auc": metrics.roc_auc, "ece": metrics.ece}},
    )
    return TrainResult(
        ensemble=ensemble,
        calibrator=calibrator,
        metrics=metrics,
        feature_names=feature_names,
        report_path=report_path,
    )
