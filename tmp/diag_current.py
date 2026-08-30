"""Diagnostic: run the CURRENT loaded model over the labeled benchmark set.

This reproduces the detector's real-world behavior (the competitive-benchmark
failure) against the model that is actually served today, and prints a
confusion summary split by: genuine-AI vs procedural-proxy vs real photos.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

# Running from tmp/ means Python puts tmp/ on sys.path, not the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from python_services.frequency_guard.api.inference import InferenceService
from python_services.frequency_guard.config import Settings, ensure_dirs, load_settings

settings: Settings = load_settings()
ensure_dirs(settings)

svc = InferenceService(settings)
svc.ensure_model()

manifest = Path("data/benchmark/manifest.csv")

rows = []
with manifest.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        rows.append(row)

print(f"Loaded model. threshold={svc._threshold:.4f}  mode={svc._training_mode}")
print(f"n_features={len(svc._ensemble.feature_names) if svc._ensemble and svc._ensemble.feature_names else '?'}")
print("=" * 100)

confusion = Counter()  # (truth, pred_is_ai) -> n
per_image = []
for row in rows:
    path = row["path"]
    truth_label = row["label"].strip().lower()  # 'real' or 'ai'
    truth = 1 if truth_label in ("ai", "fake", "generated-ai") else 0
    image_id = row["image_id"]
    state = row["state"]

    try:
        outcome = svc.analyze_bytes(Path(path).read_bytes(), source=path)
        resp = outcome.response
        pred = 1 if resp.is_ai else 0
        fake_p = resp.fake_probability
        conf_pct = resp.confidence
        agreement = resp.evidence.agreement if resp.evidence else None
        verdict_state = resp.verdict_state
    except Exception as exc:  # noqa: BLE001
        pred = -1
        fake_p = float("nan")
        conf_pct = float("nan")
        agreement = None
        verdict_state = f"ERROR:{exc}"

    per_image.append(
        (image_id, state, truth, pred, fake_p, conf_pct, agreement, verdict_state)
    )
    confusion[(truth, pred)] += 1

    print(
        f"{image_id:28s} {state:11s} truth={truth}  pred={pred}  "
        f"fake_p={fake_p:.4f}  conf={conf_pct:6.2f}  agree={agreement if agreement is None else round(agreement,3)}  "
        f"vstate={verdict_state}"
    )

print("=" * 100)

tp = confusion[(1, 1)]
fn = confusion[(1, 0)]
fp = confusion[(0, 1)]
tn = confusion[(0, 0)]
total = tp + fn + fp + tn
acc = (tp + tn) / total if total else 0.0
fpr = fp / (fp + tn) if (fp + tn) else 0.0
recall_ai = tp / (tp + fn) if (tp + fn) else 0.0

print(f"OVERALL       : n={total}  accuracy={acc:.3f}")
print(f"REAL (label 0): n={fp+tn}  FPR={fpr:.3f}  (fp={fp} tn={tn})")
print(f"AI   (label 1): n={tp+fn}  recall={recall_ai:.3f}  (tp={tp} fn={fn})")

# Split the AI class: genuine portrait vs procedural proxies
genuine = [r for r in per_image if r[0] == "ai_0_portrait"]
proxy = [r for r in per_image if r[0].startswith("ai_") and r[0] != "ai_0_portrait"]
print("\n--- Genuine AI image (ai_0_portrait) ---")
for r in genuine:
    print(f"  state={r[1]:11s} truth={r[2]} pred={r[3]} fake_p={r[4]:.4f} conf={r[5]:.2f} vstate={r[7]}")
genuine_hits = sum(1 for r in genuine if r[3] == 1)
print(f"  genuine-AI detection: {genuine_hits}/{len(genuine)}")

print("\n--- Procedural proxies (ai_*_proxy) ---")
for r in proxy:
    print(f"  state={r[1]:11s} truth={r[2]} pred={r[3]} fake_p={r[4]:.4f} conf={r[5]:.2f} vstate={r[7]}")
proxy_hits = sum(1 for r in proxy if r[3] == 1)
print(f"  proxy-AI detection: {proxy_hits}/{len(proxy)}")
