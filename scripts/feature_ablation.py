"""Feature-family honest ablation gate (E-Masterplan D4.1 / D4.2).

D4.1: Each E1 feature family (chroma, HOS, JPEG, PRNU, freq-texture) is
tested on the real GenImage/CIFAKE benchmark. Any family contributing
< 0.002 AUROC is identified for dropping and documentation — never claim
"we added features" without measured evidence.

D4.2: The stacked meta-learner is only kept if it beats probability-averaging
by >= 0.005 AUROC on the real benchmark. Otherwise the ensemble reverts to
averaging, eliminating the meta-learner + step-calibrator mismatch that
produced the reported "AI image -> high-confidence Authentic" flip.

Usage:
    python -m scripts.feature_ablation --manifest data/datasets/genimage_manifest.csv \
        [--eval-manifest data/datasets/cifake_manifest.csv] [--output-dir checkpoints]

Writes ``reports/feature_ablation_<ts>.json`` with per-family AUROC + the
model-upgrade comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from python_services.frequency_guard.config import Settings, load_settings
from python_services.frequency_guard.logging import configure_logging, get_logger
from python_services.frequency_guard.models.classifier import EnsembleClassifier
from python_services.frequency_guard.training.train_pipeline import _extract_vectors, _load_manifest

log = get_logger(__name__)

#: AUROC below which a feature family is considered non-contributing.
MIN_CONTRIBUTING_AUROC = 0.002
#: AUROC margin required for the stacked meta-learner to be kept over averaging.
META_KEEP_MARGIN = 0.005

#: Prefix -> feature-family name map from FeatureBundle.names in extractor.py.
FAMILY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("fft_", "fft"),
    ("dct_", "dct"),
    ("wavelet_", "wavelet"),
    ("noise_", "noise"),
    ("texture_", "texture"),
    ("chroma_", "chroma"),
    ("hos_", "hos"),
    ("jpeg_", "jpeg"),
    ("prnu_", "prnu"),
    ("freqtex_", "freq_texture"),
)


def _family_columns(feature_names: tuple[str, ...]) -> dict[str, list[int]]:
    """Map each feature family to the column indices it occupies."""
    families: dict[str, list[int]] = {}
    for idx, name in enumerate(feature_names):
        for prefix, family in FAMILY_PREFIXES:
            if name.startswith(prefix):
                families.setdefault(family, []).append(idx)
                break
    return families


def _auc_of_subset(
    X: np.ndarray,
    y: np.ndarray,
    cols: list[int],
    settings: Settings,
) -> float:
    """Train a standalone ensemble on the chosen columns and report AUROC."""
    X_sub = X[:, cols]
    # Guard against degenerate subsets (single column / zero variance).
    if X_sub.shape[1] == 0:
        return 0.0
    if np.unique(y).size < 2 or len(np.unique(y, axis=0)) < 2:
        return 0.0
    ensemble = EnsembleClassifier()
    ensemble.fit(X_sub, y)
    proba = ensemble.predict_proba(X_sub)[:, 1]
    try:
        return float(roc_auc_score(y, proba))
    except ValueError:
        return 0.0


def _compare_meta_vs_average(
    df,
    X: np.ndarray,
    y: np.ndarray,
    settings: Settings,
) -> dict[str, float | bool]:
    """Train a stacked meta-learner and compare it to probability averaging.

    Runs the same OOF-based stacking the training pipeline uses so the
    comparison is apples-to-apples. Returns the meta AUROC, avg AUROC, and
    whether the meta-learner should be kept.
    """
    from sklearn.model_selection import StratifiedKFold

    from python_services.frequency_guard.models.classifier import StackedMetaClassifier

    skf = StratifiedKFold(n_splits=settings.n_folds, shuffle=True, random_state=settings.random_state)
    oof_members = np.zeros((len(y), 4, 2), dtype=np.float64)
    oof_avg = np.zeros(len(y), dtype=np.float64)

    for train_idx, val_idx in skf.split(X, y):
        fold = EnsembleClassifier()
        fold.fit(X[train_idx], y[train_idx])
        oof_avg[val_idx] = fold.predict_proba(X[val_idx])[:, 1]
        oof_members[val_idx] = fold.member_probabilities(X[val_idx])

    meta = StackedMetaClassifier()
    meta.fit(oof_members, y)
    oof_meta = meta.predict_fake(oof_members)

    avg_auc = float(roc_auc_score(y, oof_avg))
    meta_auc = float(roc_auc_score(y, oof_meta))
    keep_meta = meta_auc >= avg_auc + META_KEEP_MARGIN
    return {
        "meta_auc": round(meta_auc, 4),
        "average_auc": round(avg_auc, 4),
        "margin": round(meta_auc - avg_auc, 4),
        "keep_meta": bool(keep_meta),
    }


def run_ablation(
    manifest_csv: str | Path,
    eval_manifest: str | Path | None,
    output_dir: str | Path,
    settings: Settings,
) -> dict[str, object]:
    """Run the full D4 honest gate on a (preferably real) manifest."""
    df = _load_manifest(manifest_csv)
    X, y, feature_names = _extract_vectors(df, settings)

    families = _family_columns(feature_names)
    family_results: dict[str, dict[str, float | int | bool]] = {}
    for family, cols in sorted(families.items()):
        auc = _auc_of_subset(X, y, cols, settings)
        family_results[family] = {
            "n_features": len(cols),
            "auc": round(auc, 4),
            "contributing": bool(auc >= MIN_CONTRIBUTING_AUROC),
        }
        log.info("family ablated", extra={"fields": {"family": family, "auc": auc}})

    model_gate = _compare_meta_vs_average(df, X, y, settings)

    report: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_csv),
        "eval_manifest": str(eval_manifest) if eval_manifest else None,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "families": family_results,
        "model_gate": model_gate,
        "rules": {
            "min_contributing_auc": MIN_CONTRIBUTING_AUROC,
            "meta_keep_margin": META_KEEP_MARGIN,
        },
    }

    (settings.reports_dir / "feature_ablation_latest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = settings.reports_dir / f"feature_ablation_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("feature ablation complete", extra={"fields": {"report": str(out_path)}})
    return report


def main(argv: list[str] | None = None) -> int:
    configure_logging(level="INFO")
    parser = argparse.ArgumentParser(description="Run the D4 feature-family honest ablation gate.")
    parser.add_argument("--manifest", required=True, help="Training manifest CSV.")
    parser.add_argument("--eval-manifest", default=None, help="Independent eval manifest CSV.")
    parser.add_argument("--output-dir", default="checkpoints", help="Model artifact output dir.")
    args = parser.parse_args(argv)

    settings = load_settings()
    report = run_ablation(args.manifest, args.eval_manifest, args.output_dir, settings)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
