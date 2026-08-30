"""Run Frequency Guard's own detector over the benchmark set to get honest
real-vs-AI + degradation-sensitivity numbers, and to surface which images it
gets wrong. Writes results to data/benchmark/our_results.csv and a JSON log."""
import csv, json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(".").resolve()))
from python_services.frequency_guard.config import load_settings  # noqa: E402
from python_services.frequency_guard.api.inference import InferenceService  # noqa: E402

settings = load_settings()
svc = InferenceService(settings)

manifest = Path("data/benchmark/manifest.csv")
rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
print(f"Loaded {len(rows)} manifest rows")

results = []
for r in rows:
    path = Path(r["path"])
    if not path.exists():
        print(f"MISSING {path}")
        continue
    try:
        data = path.read_bytes()
        t0 = time.perf_counter()
        outcome = svc.analyze_bytes(data, source=str(path))
        ms = (time.perf_counter() - t0) * 1000.0
        resp = outcome.response
        results.append({
            "file": r["file"],
            "label": r["label"],
            "state": r["state"],
            "image_id": r["image_id"],
            "provenance": r["provenance"],
            "is_ai_pred": int(resp.is_ai),
            "verdict_state": resp.verdict_state,
            "confidence": resp.confidence,
            "fake_prob": resp.fake_probability,
            "threshold": resp.threshold,
            "latency_ms": round(ms, 2),
            "agreement": resp.evidence.agreement if resp.evidence else None,
            "family": resp.families[0].family if resp.families else None,
            "family_prob": resp.families[0].probability if resp.families else None,
        })
        print(f"{r['file']:45s} label={r['label']:5s} state={r['state']:10s} "
              f"pred_ai={resp.is_ai} conf={resp.confidence:.2f} thr={resp.threshold:.2f} "
              f"fake_p={resp.fake_probability:.3f} agree={resp.evidence.agreement if resp.evidence else 0:.2f}")
    except Exception as e:  # noqa: BLE001
        print(f"ERROR {path}: {e}")
        results.append({
            "file": r["file"], "label": r["label"], "state": r["state"],
            "image_id": r["image_id"], "provenance": r["provenance"],
            "error": str(e),
        })

out_csv = Path("data/benchmark/our_results.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else ["file"])
    w.writeheader()
    w.writerows(results)
print(f"\nWrote {len(results)} rows to {out_csv}")
