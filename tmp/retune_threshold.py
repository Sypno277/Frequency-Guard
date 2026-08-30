"""One-off: re-tune the decision threshold against the already-trained model.

The train_real.py run already saved ensemble.joblib / calibrator.joblib /
meta_learner.joblib. Only the persisted threshold in performance.json was
wrong (1.0) because the first threshold sweep picked the most-conservative
operating point. This reloads the trained ensemble and picks the LOWEST
threshold whose FPR is still <= fpr_target (maximizes recall while keeping
the false-positive budget).
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from python_services.frequency_guard.config import Settings, ensure_dirs, load_settings
from python_services.frequency_guard.features.extractor import extract_features
from python_services.frequency_guard.io.loader import load_image_file
from python_services.frequency_guard.preprocess import preprocess
from python_services.frequency_guard.models.classifier import EnsembleClassifier

settings = load_settings()
ensure_dirs(settings)

ensemble = EnsembleClassifier.load(settings.model_dir / "ensemble.joblib")

manifest = settings.data_dir / "benchmark" / "train_manifest.csv"
X_list: list[np.ndarray] = []
y_list: list[int] = []
with manifest.open(newline="", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        loaded = load_image_file(row["image_path"], settings)
        prep = preprocess(loaded.bgr, settings)
        bundle = extract_features(prep, settings)
        X_list.append(bundle.vector)
        y_list.append(int(row["label"]))

X = np.stack(X_list)
y = np.asarray(y_list, dtype=np.int64)

proba = ensemble.predict_proba(X)[:, 1]
thresholds = np.unique(proba)[::-1]
fpr_target = 0.05
best_thr = float(thresholds[0])
for thr in thresholds:
    preds = (proba >= thr).astype(int)
    real_mask = y == 0
    if real_mask.sum() == 0:
        continue
    fpr = float(np.mean(preds[real_mask] == 1))
    if fpr <= fpr_target:
        best_thr = float(thr)
    else:
        break

print(f"re-tuned threshold: {best_thr:.6f} (target FPR {fpr_target})")

perf_path = settings.model_dir / "performance.json"
perf = json.loads(perf_path.read_text(encoding="utf-8"))
perf["threshold"] = best_thr
perf["fpr_at_threshold"] = fpr_target
perf["generated_at"] = datetime.now(timezone.utc).isoformat()
perf_path.write_text(json.dumps(perf, indent=2), encoding="utf-8")
print("performance.json updated")
