"""Drift checker (Masterplan §8 monitoring plan).

Diffs two entries in ``reports/metrics_history.json`` and exits non-zero
when any tracked metric regresses beyond its allowed threshold:

    accuracy  : drop > 0.02        (2pp)
    f1        : drop > 0.02
    roc_auc   : drop > 0.01
    ece       : rise > 0.05
    fpr_at_threshold : rise > 0.01
    latency_ms.p95    : rise > 50%

Usage:
    python -m scripts.check_drift                 # latest vs previous
    python -m scripts.check_drift --index -1 -2   # explicit entries
    python -m scripts.check_drift --baseline reports/metrics_history.json[0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from python_services.frequency_guard.config import Settings

# metric -> (direction, max_allowed_regression)
# direction: "lower_is_better" | "higher_is_better"
THRESHOLDS: dict[str, tuple[str, float]] = {
    "accuracy": ("higher_is_better", 0.02),
    "f1": ("higher_is_better", 0.02),
    "roc_auc": ("higher_is_better", 0.01),
    "ece": ("lower_is_better", 0.05),
    "fpr_at_threshold": ("lower_is_better", 0.01),
}


def _latency_p95(entry: dict[str, object]) -> float:
    lat = entry.get("latency_ms")
    if isinstance(lat, dict):
        value = lat.get("p95", 0.0)
        if isinstance(value, int | float):
            return float(value)
    return 0.0


def _metric_value(entry: dict[str, object], metric: str) -> float | None:
    if metric == "latency_p95":
        return _latency_p95(entry)
    raw = entry.get(metric)
    return float(raw) if isinstance(raw, int | float) else None


def check_drift(
    baseline: dict[str, object],
    current: dict[str, object],
) -> list[str]:
    """Return a list of human-readable regression violations (empty = OK)."""
    violations: list[str] = []
    for metric, (direction, allowance) in THRESHOLDS.items():
        base_v = _metric_value(baseline, metric)
        curr_v = _metric_value(current, metric)
        if base_v is None or curr_v is None:
            continue  # absent metrics are skipped, not failures
        delta = curr_v - base_v
        regressed = delta < -allowance if direction == "higher_is_better" else delta > allowance
        if regressed:
            arrow = "dropped" if direction == "higher_is_better" else "rose"
            violations.append(
                f"{metric} {arrow} from {base_v:.4f} to {curr_v:.4f} " f"(allowed regression {allowance})"
            )

    # Latency: relative gate on p95.
    base_lat, curr_lat = _latency_p95(baseline), _latency_p95(current)
    if base_lat > 0 and curr_lat > base_lat * 1.5:
        violations.append(f"latency p95 rose {base_lat:.1f}ms -> {curr_lat:.1f}ms (>+50%)")
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diff two benchmark history entries for regressions.")
    parser.add_argument("--history", type=Path, default=None, help="metrics_history.json path")
    parser.add_argument("--base-index", type=int, default=-2, help="baseline index (default: second-newest)")
    parser.add_argument("--curr-index", type=int, default=-1, help="current index (default: newest)")
    args = parser.parse_args(argv)

    settings = Settings()
    history_path = args.history or (settings.reports_dir / "metrics_history.json")
    if not history_path.exists():
        print(f"No history file at {history_path}; nothing to diff.")
        return 0

    try:
        history: list[dict[str, object]] = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"History file is corrupt: {exc}")
        return 2

    if len(history) < 2:
        print("Fewer than two history entries; nothing to diff.")
        return 0

    baseline = history[args.base_index]
    current = history[args.curr_index]
    violations = check_drift(baseline, current)

    if violations:
        print("REGRESSIONS DETECTED:")
        for v in violations:
            print(f"  - {v}")
        return 1

    print(
        f"OK: no drift between entries {args.base_index} and {args.curr_index} "
        f"of {len(history)} ({accuracy_of(current)})."
    )
    return 0


def accuracy_of(entry: dict[str, object]) -> str:
    acc = entry.get("accuracy")
    return f"current accuracy={acc}" if isinstance(acc, int | float) else "no accuracy recorded"


if __name__ == "__main__":
    sys.exit(main())
