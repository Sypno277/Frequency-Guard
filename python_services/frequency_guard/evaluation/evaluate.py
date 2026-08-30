"""Evaluation harness (Masterplan §3.1 evaluation/evaluate.py, §8).

Held-out metrics: accuracy/precision/recall/F1, ROC + PR curves, confusion
matrix, ECE, per-generator breakdown, latency p50/p95/p99, peak RSS.
Writes reports/benchmark_<ts>.json + appends metrics_history.json.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ..logging import get_logger

log = get_logger(__name__)


def _slice_metrics(
    y_true: np.ndarray, preds: np.ndarray, buckets: np.ndarray | None
) -> dict[str, dict[str, float]]:
    """Accuracy/F1 per bucket (generator family, resolution band, JPEG q...)."""
    out: dict[str, dict[str, float]] = {}
    if buckets is None:
        return out
    buckets = np.asarray(buckets)
    for name in np.unique(buckets):
        mask = buckets == name
        if mask.sum() == 0:
            continue
        out[str(name)] = {
            "n": int(mask.sum()),
            "accuracy": round(float(accuracy_score(y_true[mask], preds[mask])), 4),
            "f1": round(float(f1_score(y_true[mask], preds[mask], zero_division=0)), 4),
        }
    return out


def resolution_buckets(widths, heights, small_px: int = 512) -> list[str]:
    """Bucket labels by image size: 'small' (<small_px), 'medium', 'large'."""
    labels: list[str] = []
    for w, h in zip(np.asarray(widths), np.asarray(heights), strict=True):
        m = min(int(w), int(h))
        labels.append("small" if m < small_px else ("large" if m >= small_px * 2 else "medium"))
    return labels


@dataclass
class EvaluationReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    ece: float
    threshold: float
    fpr_at_threshold: float
    confusion_matrix: list[list[int]]
    roc_points: dict[str, list[float]]
    pr_points: dict[str, list[float]]
    latency_ms: dict[str, float]
    peak_rss_mb: float
    n_samples: int
    per_generator: dict[str, dict[str, float]] = field(default_factory=dict)
    per_resolution: dict[str, dict[str, float]] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "ece": self.ece,
            "threshold": self.threshold,
            "fpr_at_threshold": self.fpr_at_threshold,
            "confusion_matrix": self.confusion_matrix,
            "roc_points": self.roc_points,
            "pr_points": self.pr_points,
            "latency_ms": self.latency_ms,
            "peak_rss_mb": self.peak_rss_mb,
            "n_samples": self.n_samples,
            "per_generator": self.per_generator,
            "per_resolution": self.per_resolution,
            "generated_at": self.generated_at,
        }


def find_fpr_threshold(y_true, fake_prob, target_fpr):
    """Find the smallest threshold whose false-positive rate <= target_fpr.

    The resulting working point (chosen threshold + achieved FPR/TPR) is
    logged so the operating point is auditable rather than silently chosen.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    candidates = np.unique(fake_prob)
    best = float(candidates.max()) if len(candidates) else 0.5
    real_mask = y_true == 0
    fake_mask = y_true == 1
    n_real = int(real_mask.sum())
    for t in np.sort(candidates):
        preds = (fake_prob >= t).astype(int)
        fpr = float(np.mean(preds[real_mask] == 1)) if n_real else 0.0
        if fpr <= target_fpr:
            best = float(t)
            break
    # A threshold >= 0.5 means the AI gate is effectively unreachable (the
    # reported "AI image → high-confidence Authentic" bug). Clamp to the
    # 0.5 decision boundary so a detector can never be silently disabled.
    if best >= 0.5:
        log.warning(
            "threshold_suspicious: FPR-target search returned %.4f; clamping to 0.5",
            best,
        )
        best = 0.5

    preds = (fake_prob >= best).astype(int)
    achieved_fpr = float(np.mean(preds[real_mask] == 1)) if n_real else 0.0
    achieved_tpr = float(np.mean(preds[fake_mask] == 1)) if int(fake_mask.sum()) else 0.0
    log.info(
        "threshold_working_point",
        extra={
            "fields": {
                "threshold": round(best, 4),
                "target_fpr": target_fpr,
                "achieved_fpr": round(achieved_fpr, 4),
                "achieved_tpr": round(achieved_tpr, 4),
                "n_real": int(n_real),
                "n_fake": int(fake_mask.sum()),
            }
        },
    )
    return best


def measure_latency(predict_fn, X, repeats=3):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict_fn(X)
        times.append((time.perf_counter() - start) * 1000.0 / max(1, len(X)))
    arr = np.asarray(times)
    return {
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "p99": round(float(np.percentile(arr, 99)), 3),
    }


def current_peak_rss_mb():
    """Peak resident set size (MB), cross-platform.

    Windows: PROCESS_MEMORY_COUNTERS.PeakWorkingSetSize via psapi/kernel32
    (correct ctypes prototypes on 64-bit). Unix: ru_maxrss via resource
    (KB on Linux, bytes on macOS).
    """
    # --- Windows -------------------------------------------------------
    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)

            kernel32 = ctypes.windll.kernel32
            get_current_process = kernel32.GetCurrentProcess
            get_current_process.argtypes = []
            get_current_process.restype = wintypes.HANDLE
            handle = get_current_process()

            # Prefer psapi for the info call, fall back to kernel32.
            try:
                get_mem = ctypes.windll.psapi.GetProcessMemoryInfo
            except (OSError, AttributeError):
                get_mem = kernel32.K32GetProcessMemoryInfo
            get_mem.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                wintypes.DWORD,
            ]
            get_mem.restype = wintypes.BOOL

            ok = get_mem(handle, ctypes.byref(counters), counters.cb)
            if ok:
                return round(float(counters.PeakWorkingSetSize) / (1024.0 * 1024.0), 2)
            return 0.0
        except Exception:
            return 0.0

    # --- Unix -----------------------------------------------------------
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB; macOS reports bytes.
        if sys.platform == "darwin":
            return round(float(ru) / (1024.0 * 1024.0), 2)
        return round(float(ru) / 1024.0, 2)
    except Exception:
        return 0.0


def evaluate_model(
    y_true,
    raw_fake_prob,
    calibrated_fake_prob,
    settings,
    predict_fn=None,
    generators=None,
    ece_fn=None,
    latency_probe=None,
    image_sizes=None,
):
    from ..models.calibration import expected_calibration_error as _default_ece

    y_true = np.asarray(y_true, dtype=np.int64)
    calibrated = np.asarray(calibrated_fake_prob, dtype=np.float64)
    threshold = find_fpr_threshold(y_true, calibrated, settings.threshold_fpr_target)
    preds = (calibrated >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, preds, labels=[0, 1]).tolist()
    fp, tn = cm[0][1], cm[0][0]
    fpr = float(fp) / max(1, int(fp + tn))
    fpr_arr, tpr_arr, _ = roc_curve(y_true, calibrated)
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, calibrated)
    per_generator = {}
    if generators is not None:
        generators = np.asarray(generators)
        for gen in np.unique(generators):
            mask = generators == gen
            if mask.sum() == 0:
                continue
            gen_preds = preds[mask]
            gen_y = y_true[mask]
            per_generator[str(gen)] = {
                "n": int(mask.sum()),
                "accuracy": round(float(accuracy_score(gen_y, gen_preds)), 4),
                "f1": round(float(f1_score(gen_y, gen_preds, zero_division=0)), 4),
            }
    ece_value = float(ece_fn(y_true, calibrated)) if ece_fn is not None else _default_ece(y_true, calibrated)

    per_resolution: dict[str, dict[str, float]] = {}
    if image_sizes is not None and len(image_sizes) == len(y_true):
        widths = [int(s[1]) for s in image_sizes]
        heights = [int(s[0]) for s in image_sizes]
        per_resolution = _slice_metrics(y_true, preds, np.asarray(resolution_buckets(widths, heights)))

    report = EvaluationReport(
        accuracy=round(float(accuracy_score(y_true, preds)), 4),
        precision=round(float(precision_score(y_true, preds, zero_division=0)), 4),
        recall=round(float(recall_score(y_true, preds, zero_division=0)), 4),
        f1=round(float(f1_score(y_true, preds, zero_division=0)), 4),
        roc_auc=round(float(roc_auc_score(y_true, calibrated)) if len(set(y_true.tolist())) > 1 else 0.5, 4),
        ece=round(ece_value, 4),
        threshold=round(threshold, 4),
        fpr_at_threshold=round(fpr, 4),
        confusion_matrix=cm,
        roc_points={
            "fpr": [round(float(v), 4) for v in fpr_arr],
            "tpr": [round(float(v), 4) for v in tpr_arr],
        },
        pr_points={
            "recall": [round(float(v), 4) for v in rec_arr],
            "precision": [round(float(v), 4) for v in prec_arr],
        },
        latency_ms=(
            measure_latency(predict_fn, latency_probe)
            if predict_fn is not None and latency_probe is not None
            else {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        ),
        peak_rss_mb=current_peak_rss_mb(),
        n_samples=int(len(y_true)),
        per_generator=per_generator,
        per_resolution=per_resolution,
    )
    return report


def persist_report(report, settings):
    reports_dir = settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"benchmark_{stamp}.json"
    payload = report.to_dict()
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    history_path = reports_dir / "metrics_history.json"
    history = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            history = []
    slim = {k: v for k, v in payload.items() if k not in ("roc_points", "pr_points")}
    history.append(slim)
    history_path.write_text(json.dumps(history[-100:], indent=2), encoding="utf-8")
    log.info("evaluation persisted", extra={"fields": {"report": str(report_path)}})
    return report_path
