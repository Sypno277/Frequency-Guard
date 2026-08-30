# Frequency Guard v3 — E-Masterplan (Bias & Calibration Integrity Fix)

> Status: **ACTIVE** — supersedes the previous "Advanced Accuracy Enhancement" roadmap.
> The prior plan aimed to push accuracy higher by adding features and a stacked ensemble. It is **superseded** because the reported failure — *an AI-generated image shown as "Uncertain" and then predicted "Authentic Image" at high confidence* — is not a missing-feature problem. It is a **threshold-calibration integrity** problem compounded by **training on a synthetic-only demo set** whose "fake" class does not resemble real diffusion/GAN output.
>
> **Order of operations is fixed.** We fix the bug first, then build the real-data foundation, then harden confidence semantics, and only then touch models/features — and every change is gated on *real-benchmark* metrics, never synthetic demo numbers.

---

## 0. Why the current system mislabels AI images as "Authentic"

The reported sequence — **"Uncertain"** → **"Authentic Image" at high confidence** — has four compounding causes, all traced in code:

| # | Cause | Evidence |
|---|---|---|
| 1 | **Synthetic-only training.** `scripts/train_demo.py` builds the "fake" class from `_synthetic_fake()`: a hand-tuned flat-spectrum image with a checkerboard term. Real SDXL/FLUX/Midjourney images share **none** of these exact statistics. The ensemble learned "my synthetic fake vs my synthetic real," not "real AI image vs real photo." | `train_demo.py`, `data/demo_dataset/manifest.csv` |
| 2 | **Stale / near-unity threshold.** `inference._resolve_threshold()` reads `performance.json`/`report.json` without sanity checks. `reports/metrics_history.json` shows persisted thresholds of **1.0**, **0.001**, then **0.0791**. With a stored `threshold = 1.0`, `is_ai = calibrated_fake >= 1.0` is essentially never true — so even a 0.95 confidence renders "Authentic." | `inference.py::_resolve_threshold`, `reports/metrics_history.json` |
| 3 | **Isotonic calibrator overfit to bimodal synthetic OOF.** Fit on ~480 near-0/near-1 scores, isotonic regression maps an unseen raw score (e.g. 0.3–0.4, what a real AI image elicits) down to ~0.0–0.05. Below a low threshold like 0.0791, that flips `is_ai = false`. | `models/calibration.py`, `metrics_history.json` (ECE 0.0 on synthetic, 0.0445 later) |
| 4 | **UI "Uncertain" gate is on a different axis than the verdict.** `AnalysisResults.tsx` returns "Uncertain" whenever `confidence < 50`, but the verdict is `calibrated_fake >= threshold`. With `threshold ≈ 0.08`, a score of 0.3 yields `is_ai = true` yet `confidence = 30` → "Uncertain." When the calibrator collapses the score below the threshold, it becomes `is_ai = false` at a high confidence → the reported flip. | `AnalysisResults.tsx::getResultText` |

No amount of new features fixes this. More features only change *which* wrong decision gets made more precisely. The fix is data + calibration/threshold integrity.

---

## D0 — Bug-fix lock & honest baseline (do this first)

**D0.1 Threshold integrity guard.**
- Replace `InferenceService._resolve_threshold()` with a `VerdictConfig` that:
  - Rejects any persisted threshold `>= 0.5` for the AI-detection gate (a detector whose "fake" gate is at 1.0 is not a detector).
  - Clamps the effective threshold to `[0.05, 0.5]`.
  - Reads threshold only from a **single** persisted source written at train time, eliminating the old `performance.json` vs `report.json` ambiguity.

**D0.2 Deprecate near-unity thresholds at training time.**
- In `evaluation/evaluate.py::find_fpr_threshold()`: if the FPR-target search returns `>= 0.5`, log `threshold_suspicious` and fall back to the `0.5` decision boundary — **never persist 1.0**.
- Store the *calibrated* threshold (not raw) in both `report.json` and `performance.json`.

**D0.3 Regression lock test.**
- New `tests/python/regression/test_threshold_integrity.py`:
  - Trains a tiny ensemble on the demo manifest.
  - Asserts the persisted threshold is in `(0.05, 0.5)`, never `>= 0.5`.
  - Asserts `is_ai` is consistent with `confidence >= 50` for the same input (kills the "Uncertain ≠ verdict" contradiction).

**Gate (D0):** full pytest + mypy + ruff/black clean; the "AI → high-confidence Authentic" reproduction test passes; no persisted threshold `>= 0.5`.

---

## D1 — Real-data foundation (the only durable accuracy fix)

### D1.1 Dataset acquisition
- New `scripts/fetch_datasets.py` with checksummed, licensed downloaders for:
  - **GenImage** (NeurIPS'23): SD1.5/SDXL/VQGAN + real photography — the primary training corpus.
  - **CIFAKE**: 120k images — **cross-dataset evaluation only, never trained on**.
  - **DiffusionForensics**-style FLUX/Midjourney-class sets where available.

### D1.2 Manifest + de-duplication
- New `scripts/build_manifest.py` normalizes any source into the existing CSV schema: `image_path,label,generator,dataset,split`.
- pHash near-duplicate removal **across** train/test boundaries — silent leakage is the #1 fake-accuracy killer.

### D1.3 Train on real data
- Wire `scripts/train_benchmark.py` to the full protocol on a real manifest:
  - Stratified CV → calibrator fit on **meta-OOF** scores → threshold from **calibrated** OOF → LOMO → cross-dataset.
  - Persist `reports/benchmark_<ts>.json` + extend `metrics_history.json` with `cross_dataset_auc` and `lomo_worst_auc`.

**Gate (D1):** real manifest trains end-to-end; LOMO + cross-dataset reports generated; thresholds persisted correctly; `data/README.md` documents provenance + licenses.

---

## D2 — Calibration & confidence integrity

### D2.1 Calibrator sanity guard
- In `models/calibration.py`:
  - Detect isotonic fits whose output histogram is a near-step (≥95% of mapped values in `{0,1}`) — a sign of overfit to bimodal synthetic data. Fall back to Platt/sigmoid in that case.
  - Always fit the calibrator on **meta-OOF** scores used at inference (already done in `train_pipeline.py`), and assert monotonicity.

### D2.2 Confidence → verdict consistency (`VerdictResolver`)
- New `python_services/frequency_guard/models/verdict.py`, single source of truth used by both `inference.py` and any UI consumer:

```
if calibrated_fake >= threshold:
    if calibrated_fake < 0.5:      -> "ambiguous-ai"   (UI: Uncertain)
    else:                          -> "ai"             (UI: AI Generated)
else:
    if calibrated_fake > 1 - threshold: -> "ambiguous-real" (UI: Uncertain)
    else:                                  -> "authentic"     (UI: Authentic)
```

This guarantees a **confident** label is never shown below the 0.5 confidence axis — killing the "Uncertain → Authentic@95%" contradiction.

**Gate (D2):** no confidence `< 50` ever renders "Authentic"/"AI" text; unit tests cover the ambiguity band; ECE `< 0.05` on the real benchmark.

---

## D3 — Confidence-aware abstention & trustworthy UI

### D3.1 Backend abstention field
- Add `verdict_state: "ai" | "authentic" | "uncertain"` to `AnalyzeResponse`. Keep `is_ai`/`confidence` for compatibility, but render off `verdict_state`.

### D3.2 Rebuild `AnalysisResults.tsx`
- Replace `getResultText()` / `getResultColor()` with a `VerdictState`-driven switch:
  - `uncertain` → always the warning triangle + "Uncertain", regardless of `is_ai`.
  - `ai` / `authentic` → only ever shown when `confidence >= 50`.

### D3.3 Ensemble-agreement gate
- Use the already-computed `evidence.agreement` (from `models/contributions.py`): `agreement < 0.5` forces `verdict_state = "uncertain"` even when confidence is high.

**Gate (D3):** vitest coverage for the verdict-state mapper; manual browser check confirms no AI image renders "Authentic" at `confidence < 50`; no text truncation/overflow.

---

## D4 — Model verification on real data (not synthetic)

### D4.1 Feature-family honest gate
- Each E1 feature family (chroma, HOS, JPEG, PRNU, freq-texture) is tested on the real GenImage/CIFAKE benchmark.
- Any family contributing `< 0.002 AUROC` is **dropped and documented** — no "we added features" claim without measured evidence.

### D4.2 Model upgrade gate
- The stacked meta-learner is **only kept** if it beats probability-averaging by `>= 0.005 AUROC` on the real benchmark. Otherwise the ensemble reverts to averaging, eliminating the meta-learner + step-calibrator mismatch that produced the flip.

**Gate (D4):** the real-benchmark report is the *only* place accuracy is claimed; synthetic demo numbers are explicitly labeled "unrepresentative."

---

## D5 — Success gates (measurable, honest)

| Metric | v2 "demo" claim | D-series target (real data) |
|---|---|---|
| Cross-dataset AUROC (GenImage→CIFAKE) | 1.0 (fake) | **≥ 0.85** |
| LOMO worst-family AUROC | dormant | **≥ 0.90** |
| FPR on real authentic photos | unmeasured | **≤ 1%** |
| ECE (real benchmark) | 0.0445 (synthetic) | **≤ 0.03** |
| Confident-wrong rate on real AI images | N/A (bug) | **≤ 5%** |
| "Uncertain" rate | inconsistent | **reported, not hidden** |
| Latency (CPU, default) | <150ms | **maintained <150ms** |
| CI drift | 0 regressions | **0 regressions incl. new `verdict_state`** |

---

## Implementation order & effort

| Phase | Depends on | Effort | Why |
|---|---|---|---|
| **D0** | none | S | fixes the exact reported bug immediately |
| **D1** | dataset downloads | M-L | the only durable fix |
| **D2** | D0, D1 | M | calibrator + verdict consistency |
| **D3** | D2 | S | UI truthfulness |
| **D4** | D1 | M | honest feature/model validation |
| **D5** | all | S | gates |

**D0 → D1 → D2 is the critical path.** D3 and D4 proceed in parallel once D1 and D2 land. Each phase ends with: full pytest green, ruff/black/mypy clean, `tsc --noEmit` + vitest + build green, benchmark written to `reports/`, drift check passing.

---

## New Files Summary

```
scripts/fetch_datasets.py                            # dataset downloaders + checksums
scripts/build_manifest.py                            # normalize sources → manifest, pHash dedup
python_services/frequency_guard/models/verdict.py    # VerdictResolver (D2.2)
tests/python/regression/test_threshold_integrity.py  # D0 regression lock
tests/python/unit/test_verdict.py                    # D2 verdict-state coverage
src/lib/services/verdictState.ts                     # D3 UI mapper
```

---

## Progress Checklist

- [x] **D0** — threshold-integrity guard (`VerdictConfig`), `find_fpr_threshold` "never ≥ 0.5" fallback, `test_threshold_integrity.py` regression lock
- [x] **D1** — `fetch_datasets.py` + `build_manifest.py` created; `data/README.md` documents provenance + licenses. *(Benchmark run on a real manifest still requires the actual GenImage/CIFAKE download — see Status note.)*
- [x] **D2** — calibrator sanity guard (near-step isotonic → Platt fallback); `models/verdict.py::VerdictResolver`; `test_verdict.py`
- [x] **D3** — `verdict_state` API field + `AnalysisResults.tsx` rebuild + `src/lib/services/verdictState.ts` + vitest coverage
- [x] **D4** — `scripts/feature_ablation.py` implements the E1 feature-family honest gate and the stacked meta-learner keep/drop gate
- [ ] **D5** — measure all §D5 targets on real data; record in `reports/metrics_history.json`; verify CI drift gates *(requires the real datasets downloaded + benchmark run)*

### Status note

D1/D4 are **code-complete** but the honest-benchmark measurement (the D5 gate) is
**gated on downloading the real corpora** (GenImage/CIFAKE) and running:

```bash
python -m scripts.fetch_datasets --cifake      # + import GenImage zip
python -m scripts.build_manifest --source data/datasets/genimage \
    --dataset genimage --out data/datasets/genimage_manifest.csv
python -m scripts.train_benchmark --manifest data/datasets/genimage_manifest.csv \
    --eval-manifest data/datasets/cifake_manifest.csv
python -m scripts.feature_ablation --manifest data/datasets/genimage_manifest.csv
```

Until then the demo-set numbers in `reports/metrics_history.json` remain
**explicitly "unrepresentative"** and must not be reported as real accuracy.
