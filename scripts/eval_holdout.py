"""Honest held-out evaluation + regression gate (Candidate B, Layer 3 fix).

The repository's reported metrics were in-sample: the benchmark images were
mixed into training, so "100% accuracy / 0.98" proved memorization, not
generalization. This script:

  1. Trains ONLY on the demo train split (``data/split/train_manifest.csv``),
     which by construction excludes every benchmark image.
  2. Evaluates on the benchmark holdout (``data/split/holdout_manifest.csv``)
     — the only real photos + the only genuine AI portrait in the repo.
  3. Enforces the regression gate:
       real-photo FPR  <= 2%
       AI-image recall >= 80%
     Exits non-zero when the gate fails, so CI cannot ship a regressed model.

Artifacts are written to ``checkpoints/`` (ensemble/calibrator/meta) and the
gate report to ``reports/holdout_report.json``.

Run:
    python -m scripts.eval_holdout [--fpr-target 0.02] [--no-save]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from python_services.frequency_guard.config import Settings, ensure_dirs, load_settings
from python_services.frequency_guard.evaluation.evaluate import evaluate_model
from python_services.frequency_guard.features.extractor import (
    FeatureBundle,
    cache_key,
    extract_features,
    global_cache,
)
from python_services.frequency_guard.io.loader import load_image_file
from python_services.frequency_guard.logging import configure_logging, get_logger
from python_services.frequency_guard.models.attribution import FamilyAttributor
from python_services.frequency_guard.models.calibration import ProbabilityCalibrator
from python_services.frequency_guard.models.classifier import (
    EnsembleClassifier,
    EnsembleConfig,
    StackedMetaClassifier,
)
from python_services.frequency_guard.preprocess import PreprocessedImage, preprocess
from python_services.frequency_guard.training.train_pipeline import (
    _cross_validate,
    _extract_vectors,
    _load_manifest,
)

log = get_logger(__name__)

# Default regeneration gate (Masterplan §8 success metrics).
REAL_FPR_MAX = 0.02
AI_RECALL_MIN = 0.80


def _extract_holdout(
    manifest_path: Path,
    settings: Settings,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    """Extract features for the holdout manifest, preserving row metadata."""
    df = _load_manifest(manifest_path)
    rows: list[dict[str, str]] = []
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    for row in df.itertuples(index=False):
        path = Path(row.image_path)
        label = int(row.label)
        loaded = load_image_file(path, settings)
        prep = preprocess(loaded.bgr, settings)
        key = cache_key(str(path), prep, settings)

        def _compute(p: PreprocessedImage = prep, s: Settings = settings) -> FeatureBundle:
            return extract_features(p, s)

        bundle = global_cache.get_or_compute(key, _compute)
        X_list.append(bundle.vector)
        y_list.append(label)
        rows.append(
            {
                "image_path": str(path),
                "label": str(label),
                "image_id": path.stem,
                "state": path.parent.name,
            }
        )
    X = np.stack(X_list)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y, rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Honest held-out evaluation + regression gate")
    parser.add_argument("--fpr-target", type=float, default=0.02, help="Target real-photo FPR")
    parser.add_argument("--no-save", action="store_true", help="Do not write artifacts to checkpoints/")
    args = parser.parse_args()

    settings: Settings = load_settings()
    configure_logging(level=settings.log_level)
    ensure_dirs(settings)

    split_dir = settings.data_dir / "split"
    train_manifest = split_dir / "train_manifest.csv"
    holdout_manifest = split_dir / "holdout_manifest.csv"
    if not train_manifest.exists() or not holdout_manifest.exists():
        print("Split manifest missing. Run `python -m scripts.build_split` first.", file=sys.stderr)
        return 2

    print(f"TRAIN   : {train_manifest}")
    print(f"HOLDOUT : {holdout_manifest}")

    # --- 1. Train on the split (benchmark excluded) ---------------------
    config = EnsembleConfig()
    train_df = _load_manifest(train_manifest)
    X_train, y_train, feature_names = _extract_vectors(train_df, settings, degrade=True)
    print(f"Train features: {X_train.shape}")

    # Cross-validate for calibrator/meta fitting (leakage-free OOF).
    _, oof, oof_members = _cross_validate(X_train, y_train, settings, config)

    ensemble = EnsembleClassifier(config)
    ensemble.feature_names = feature_names
    ensemble.fit(X_train, y_train)

    meta = StackedMetaClassifier()
    meta.fit(oof_members, y_train)
    oof_raw = meta.predict_fake(oof_members) if meta.fitted else oof
    calibrator = ProbabilityCalibrator(method=settings.calibrate_method)
    calibrator.fit(oof_raw, y_train)

    # --- 2. Evaluate on the held-out benchmark --------------------------
    X_hold, y_hold, hold_rows = _extract_holdout(holdout_manifest, settings)
    print(f"Holdout features: {X_hold.shape}")

    raw_fake = ensemble.predict_proba(X_hold)[:, 1]
    calib_fake = calibrator.predict_proba(raw_fake)

    report = evaluate_model(
        y_true=y_hold,
        raw_fake_prob=raw_fake,
        calibrated_fake_prob=calib_fake,
        settings=settings,
        predict_fn=lambda M: ensemble.predict(M),
        latency_probe=X_hold[:8],
    )

    # --- 3. Enforce the regression gate ---------------------------------
    fpr = report.fpr_at_threshold
    recall = report.recall

    gate_passed = fpr <= REAL_FPR_MAX and recall >= AI_RECALL_MIN
    print("\n=== HELD-OUT REGRESSION GATE ===")
    print(f"  real-FPR          : {fpr:.4f}  (target <= {REAL_FPR_MAX})")
    print(f"  AI recall         : {recall:.4f}  (target >= {AI_RECALL_MIN})")
    print(f"  accuracy          : {report.accuracy:.4f}")
    print(f"  roc_auc           : {report.roc_auc:.4f}")
    print(f"  ece               : {report.ece:.4f}")
    print(f"  confusion_matrix  : {report.confusion_matrix}")
    print(f"  gate result       : {'PASS' if gate_passed else 'FAIL'}")
    print("=" * 60)

    # Write the gate report regardless, so the truth is recorded.
    report_dir = settings.reports_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    gate_payload = report.to_dict()
    gate_payload["gate"] = {
        "passed": bool(gate_passed),
        "real_fpr": round(fpr, 4),
        "real_fpr_target": REAL_FPR_MAX,
        "ai_recall": round(recall, 4),
        "ai_recall_min": AI_RECALL_MIN,
        "train_manifest": str(train_manifest),
        "holdout_manifest": str(holdout_manifest),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    gate_path = report_dir / "holdout_report.json"
    gate_path.write_text(json.dumps(gate_payload, indent=2), encoding="utf-8")
    print(f"gate report written: {gate_path}")

    # --- 4. Optionally persist the honest model artifacts ---------------
    if not args.no_save:
        ensemble.save(settings.model_dir / "ensemble.joblib")
        calibrator.save(settings.model_dir / "calibrator.joblib")
        meta.save(settings.model_dir / "meta_learner.joblib")
        attributor = FamilyAttributor()
        attributor.save(settings.model_dir / "attributor.joblib")

        perf_payload = gate_payload
        perf_payload["training_mode"] = "honest_split"
        perf_payload["degradation_augmentation"] = True
        perf_payload["holdout"] = True
        (settings.model_dir / "performance.json").write_text(
            json.dumps(perf_payload, indent=2), encoding="utf-8"
        )
        print(f"artifacts written to: {settings.model_dir}")

    print(f"\nGATE {'PASSED' if gate_passed else 'FAILED'} — {'ok' if gate_passed else 'model cannot ship'}")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
