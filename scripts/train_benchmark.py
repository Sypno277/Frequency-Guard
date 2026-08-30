"""E-Masterplan E0: full benchmark protocol runner.

Runs the complete evaluation protocol on one or two manifests:

1. Train on the training manifest (stratified CV + calibrator + meta-learner).
2. LOMO holdout (leave-one-generator-out) — needs a ``generator`` column.
3. Cross-dataset evaluation: train manifest → eval manifest (e.g. GenImage
   → CIFAKE). When ``--eval-manifest`` is omitted, a held-out split of the
   training manifest is used instead.

Writes ``reports/benchmark_<ts>.json`` + appends to
``reports/metrics_history.json`` with the E0 drift fields
(``cross_dataset_auc``, ``lomo_worst_auc``).

Usage:
    python -m scripts.train_benchmark --manifest data/manifest.csv \
        [--eval-manifest data/cifake_manifest.csv] [--output-dir checkpoints]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from python_services.frequency_guard.config import Settings, load_settings
from python_services.frequency_guard.evaluation.evaluate import evaluate_model
from python_services.frequency_guard.logging import configure_logging, get_logger
from python_services.frequency_guard.models.calibration import ProbabilityCalibrator
from python_services.frequency_guard.training.train_pipeline import (
    TrainResult,
    _extract_vectors,
    _load_manifest,
    train_from_manifest,
)

log = get_logger(__name__)


def _cross_dataset_eval(
    train_result: TrainResult,
    eval_manifest: str | Path,
    settings: Settings,
) -> dict[str, float]:
    """Evaluate the trained pipeline on an independent eval manifest."""
    df = _load_manifest(eval_manifest)
    X_eval, y_eval, _ = _extract_vectors(df, settings)

    calibrator: ProbabilityCalibrator = train_result.calibrator
    ensemble = train_result.ensemble
    raw = ensemble.predict_proba(X_eval)[:, 1]
    calibrated = np.asarray(
        [float(calibrator.predict_proba(np.asarray([p]))[0]) for p in raw], dtype=np.float64
    )
    report = evaluate_model(
        y_true=y_eval,
        raw_fake_prob=raw,
        calibrated_fake_prob=calibrated,
        settings=settings,
    )
    return {
        "cross_dataset_accuracy": report.accuracy,
        "cross_dataset_auc": report.roc_auc,
        "cross_dataset_f1": report.f1,
        "cross_dataset_ece": report.ece,
        "cross_dataset_n": float(report.n_samples),
    }


def run_benchmark(
    manifest_csv: str | Path,
    eval_manifest: str | Path | None,
    output_dir: str | Path,
    settings: Settings,
) -> dict[str, object]:
    """Run train → LOMO → cross-dataset and persist the combined report."""
    train_result = train_from_manifest(manifest_csv, output_dir, settings=settings)

    report: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_manifest": str(manifest_csv),
        "eval_manifest": str(eval_manifest) if eval_manifest else None,
        "in_distribution": train_result.to_dict()["metrics"],
    }

    # LOMO report comes from the training run when a generator column exists.
    report_path = train_result.report_path
    if Path(report_path).exists():
        stored = json.loads(Path(report_path).read_text(encoding="utf-8"))
        lomo = stored.get("lomo_holdout", {})
        report["lomo_holdout"] = lomo
        if lomo:
            worst = min(v.get("roc_auc", 0.0) for v in lomo.values())
            report["lomo_worst_auc"] = worst

    if eval_manifest:
        report["cross_dataset"] = _cross_dataset_eval(train_result, eval_manifest, settings)

    # Append the E0 drift fields to metrics_history.json.
    history_path = settings.reports_dir / "metrics_history.json"
    history: list[dict[str, object]] = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            history = []
    slim: dict[str, object] = {
        "generated_at": report["generated_at"],
        "in_distribution": report["in_distribution"],
        "lomo_worst_auc": report.get("lomo_worst_auc"),
        "cross_dataset_auc": (report.get("cross_dataset") or {}).get("cross_dataset_auc"),
    }
    history.append(slim)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history[-100:], indent=2), encoding="utf-8")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = settings.reports_dir / f"benchmark_e0_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("benchmark complete", extra={"fields": {"report": str(out_path)}})
    return report


def main() -> None:
    """CLI entry point."""
    configure_logging(level="INFO")
    parser = argparse.ArgumentParser(description="Run the E0 benchmark protocol.")
    parser.add_argument("--manifest", required=True, help="Training manifest CSV.")
    parser.add_argument(
        "--eval-manifest", default=None, help="Independent eval manifest CSV (cross-dataset)."
    )
    parser.add_argument("--output-dir", default="checkpoints", help="Model artifact output dir.")
    args = parser.parse_args()

    settings = load_settings()
    report = run_benchmark(args.manifest, args.eval_manifest, args.output_dir, settings)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
