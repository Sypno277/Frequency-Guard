# Frequency Guard — Improvement Plan
## Implementing the competitive-benchmark findings, with tests

**Companion report:** `reports/competitive_benchmark.md` (evidence) · **Date:** 2026-08-27
**Scope:** convert the root-cause findings (training-data gap, threshold miscalibration, agreement-gate leak, no degradation augmentation at train time, single-vector feature fusion) into concrete, tested code changes.

---

## 0. Grounding — what the code actually does today

| Finding | Code truth | Why it matters |
|---|---|---|
| Degradation augmentation exists but is **unused** | `features/augmentation.py::augment_bgr` is defined, but `training/train_pipeline.py::_extract_vectors` never calls it, and `scripts/train_demo.py` builds its own hardcoded `_jpeg_degrade/_resize_cycle/_center_crop` on **synthetic** images only | Model trained only on pristine synthetic + a narrow hand-rolled degrade; real generator output + real degradation never seen |
| `augment_bgr` is **deterministic** | `augment_bgr` uses `np.random.default_rng(cfg.random_state)` — same seed → same degradation every call | Cannot serve as a per-sample training augmentation; needs seeding driven per-image |
| Ensemble-agreement gate **leaks hard positives** | `api/inference.py::analyze_bytes`: on `evidence.agreement < 0.5` it sets `verdict_state="uncertain"` but keeps `is_ai = verdict.is_ai` | 14/20 real photos emit `is_ai=True` with agreement 0.03–0.08 (the worst FPs) |
| Threshold tuned on wrong distribution | `evaluation/evaluate.py::find_fpr_threshold` uses `settings.threshold_fpr_target=0.02` and clamps ≥0.5; `checkpoints/performance.json` threshold = 0.0791, measured on the synthetic demo set | 70% FPR on real photos; far more aggressive than competitor ~0.9 |
| Feature fusion is a single vector | `features/extractor.py::extract_features` concatenates FFT+DCT+wavelet+noise+texture+chroma+HOS+JPEG+PRNU+freq_texture into one vector; PRNU/JPEG/HOS are **already extracted** but not used as an independent, trustable branch | No way to prefer authentic-side (PRNU) or compression-side (JPEG) evidence when frequency fires |
| Per-generator / per-resolution metrics never populated | `evaluation/evaluate.py` accepts `generators=` and `image_sizes=`, but `train_pipeline.py` and `scripts/train_demo.py` never pass them | `performance.json` shows `"per_generator": {}`, `"per_resolution": {}` → generator blind spots invisible |
| Real generator corpora not wired | `data/README.md` documents `scripts/fetch_datasets.py` + `scripts/build_manifest.py` for GenImage/CIFAKE; no real corpus present yet | The only durable accuracy fix (train on real generator output) is not yet executable end-to-end in CI |

---

## 1. Design goals (measurable)

After implementation, on a **real-generator benchmark** (not the synthetic demo set):

1. **Real-photo specificity ≥ 0.90** at the chosen operating point (currently 0.30).
2. **AI recall ≥ 0.80** on real generator output (currently 0.625 on the mixed set, but 0% on the one genuine diffusion image).
3. **No hard `is_ai=True` when ensemble agreement < 0.5** unless calibrated probability is also ≥ 0.9.
4. **Resize/screenshot robustness**: accuracy on degraded states within ±0.10 of pristine (currently resize collapses to 0.222).
5. **Honest regression gate**: CI fails if the above regress (seed the golden set from the benchmark failures).

---

## 2. Work items (each with tests)

### WI-1 — Wire degradation augmentation into training (P1, Critical) — **DONE**

**Files:** `training/train_pipeline.py`, `features/augmentation.py`, `config.py`

- Make `augment_bgr` **per-sample stochastic**: change the RNG to derive from a per-image seed (e.g. `np.random.default_rng(seed)` where seed = `hash(path + fold) & 0xFFFFFFFF`), or accept an explicit `seed` arg. Keep default behavior deterministic for tests. **→ Done: `augment_bgr(bgr, settings, config, seed=None)`; falls back to `config.random_state`.**
- In `train_pipeline.py::_extract_vectors`, for each training row, emit **both the pristine sample and an augmented sample** (call `augment_bgr(loaded.bgr, settings)` with a fold-dependent seed) so the model sees degradation **on real generator data**. **→ Done: `_extract_vectors(df, settings, degrade=False)`; when `degrade=True` emits pristine + augmented with `seed = hash(path) & 0xFFFFFFFF`.**
- Add a `degrade: bool` switch (default off). **→ Done.**
- Ensure extraction cache key includes the augmentation seed. **→ Done: augmented key is `f"{path}::aug::{seed}"`.**

**Tests:**
- `tests/python/unit/test_augmentation.py` — **new** (4 tests): `test_augment_deterministic_with_seed`, `test_different_seed_generally_differs`, `test_preserves_dimensions`, `test_pristine_when_all_ops_disabled`.
- `tests/python/unit/test_train_pipeline.py` — **extended** (4 tests): `test_returns_vectors_and_labels` (existing), `test_degrade_emits_dual_vectors`, `test_degrade_default_is_pristine`, `test_cache_key_differs_by_seed`.

---

### WI-2 — Make the agreement gate suppress hard positives (P3, High) — **DONE**

**File:** `api/inference.py` (and `schemas.py` if a new field is added)

- In `analyze_bytes`, change the low-agreement branch: when `evidence.agreement < 0.5` **and** `calibrated_fake < 0.9`, set `verdict_state = "uncertain"` **and** `is_ai = False`. Only keep `is_ai = True` when agreement < 0.5 **and** the calibrated probability is ≥ 0.9 (near-certain). **→ Done: `if evidence.agreement < 0.5 and calibrated_fake < 0.9: verdict_state = "uncertain"; is_ai = False`.**
- Keep `verdict`/`confidence` as-is for transparency; only the hard gate changes. **→ Done.**

**Tests:**
- `tests/python/unit/test_inference.py` — **extended** (2 tests): `test_low_agreement_suppresses_hard_positive`, `test_low_agreement_keeps_positive_when_certain`.
- `tests/python/unit/test_threshold_integrity_additions.py` — **new** (1 test): `TestAgreementGateRegression::test_no_hard_ai_below_agreement_floor`.

---

### WI-3 — Re-tune the threshold at the operating point (P2, High) — **DONE**

**Files:** `evaluation/evaluate.py`, `config.py`, `api/inference.py`

- Make `find_fpr_threshold` **log the selected threshold and the resulting FPR/TPR** so it's auditable. **→ Done: logs `threshold_working_point` with `threshold`, `target_fpr`, `achieved_fpr`, `achieved_tpr`.**
- Add a `threshold_override` to `InferenceService` (e.g. via `settings` or an explicit `analyze_bytes(..., threshold=...)`) so a tuned threshold can be applied at serve time without retraining. **→ Done: `analyze_bytes(..., threshold=None)`; uses `clamp_threshold(threshold)` when provided, else `self._threshold`; response carries `effective_threshold`.**
- Keep the `clamp_threshold` ≥0.5 guard. **→ Done (unchanged).**

**Tests:**
- `tests/python/unit/test_evaluate.py` — **extended** (3 tests): `test_find_fpr_threshold_logs_working_point`.
- `tests/python/unit/test_inference.py` — **extended** (1 test): `test_threshold_override_applied_at_inference`.
- `tests/python/unit/test_threshold_integrity_additions.py` — **new** (1 test): `TestThresholdOverrideRegression::test_override_reflected_in_response`.

---

### WI-4 — Populate per-generator / per-resolution metrics (P7, Low; unblocks the generator blind-spot question) — **DEFERRED**

**Files:** `training/train_pipeline.py`, `evaluation/evaluate.py`, `scripts/train_demo.py`, `scripts/train_benchmark.py`

- Parse a `generator` column if present and pass `generators=` into `evaluate_model`; likewise `image_sizes` so `per_generator`/`per_resolution` populate.
- Surface per-generator **recall** (not just accuracy).
- Add `generator`/`dataset` columns to the benchmark manifest as real corpora are added.

**Tests (planned):**
- `test_per_generator_metrics_populated`, `test_per_resolution_metrics_populated` (test_evaluate.py), `test_model_performance_endpoint_includes_generator_breakdown` (test_api_pipeline.py).

---

### WI-5 — Fuse a complementary authentic-side signal (P4, Medium-High) — **DEFERRED**

**Files:** `features/extractor.py`, `models/classifier.py`, `training/train_pipeline.py`

- Implement **feature-group soft-voting** via `StackedMetaClassifier`, and/or an authentic-side PRNU prior bias. **Not yet implemented** — recommended to reuse `StackedMetaClassifier` with per-group base learners.

**Tests (planned):**
- `test_feature_groups_cover_all_features`, `test_prnu_features_are_nonempty` (test_extractor.py), `test_group_soft_voting_changes_decision_on_strong_prnu` (test_models.py), `test_prnu_survives_jpeg` (test_perturbations.py).

---

### WI-6 — Seed a regression benchmark + CI gate (P6, Medium) — **PARTIAL**

**Files:** `data/golden/` + `tests/python/regression/test_golden.py`, `.github/workflows/ci.yml`

- The **agreement-gate and threshold-override regression tests** are added (`tests/python/unit/test_threshold_integrity_additions.py`), locking WI-2/WI-3.
- **Not yet done:** promoting the benchmark **real-photo + genuine-AI** failure set into `data/golden/` with a `state`/`provenance` column, and adding a CI gate asserting real specificity ≥ 0.90 / AI recall ≥ 0.80 across state. These remain a follow-up that requires the real-generator corpus (WI-7).

**Tests (added):**
- `tests/python/unit/test_threshold_integrity_additions.py` — `TestAgreementGateRegression`, `TestThresholdOverrideRegression`.

---

### WI-7 — Wire the real-generator training path end-to-end (P1 companion) — **DEFERRED**

**Files:** `scripts/fetch_datasets.py` (verify), `scripts/build_manifest.py` (verify), `scripts/train_benchmark.py`, add `scripts/eval_golden.py`

- Confirm `fetch_datasets.py --genimage-zip` + `build_manifest.py` runnable; add smoke test; add `eval_golden.py` to emit `reports/golden_report.json`.

**Tests (planned):**
- `test_eval_golden_produces_report`, `test_manifest_with_generator_column_loaded`.

---

## 3. Suggested implementation order

1. **WI-1** (augmentation into training) — **DONE**
2. **WI-7** (fetch + build + eval_golden) — **DEFERRED** (requires real generator data not present in repo)
3. **WI-3** (threshold retune + override) — **DONE**
4. **WI-2** (agreement gate) — **DONE**
5. **WI-4** (per-generator metrics) — **DEFERRED**
6. **WI-5** (group soft-voting / PRNU bias) — **DEFERRED**
7. **WI-6** (golden regression + CI gate) — **PARTIAL** (agreement/threshold gates added; real-photo golden set + CI specificity/recall gate deferred pending WI-7)

---

## 4. Test/verification commands (run for implemented items)

```bash
# Augmentation + training-pipeline degrade flag (WI-1)
python -m pytest tests/python/unit/test_augmentation.py tests/python/unit/test_train_pipeline.py -q

# Agreement gate + threshold override (WI-2, WI-3)
python -m pytest tests/python/unit/test_inference.py tests/python/unit/test_evaluate.py -q

# Regression additions locking the new behavior
python -m pytest tests/python/unit/test_threshold_integrity_additions.py -q

# Regression + robustness (confirm no regressions)
python -m pytest tests/python/regression tests/python/robustness -q
```

**Verified results (2026-08-27):**
- `test_augmentation.py`, `test_train_pipeline.py`, `test_inference.py`, `test_evaluate.py`, `test_models.py` → **65 passed**
- `test_threshold_integrity_additions.py` → **2 passed**
- `tests/python/regression` + `tests/python/robustness` → **21 passed**

---

## 5. Success criteria (status)

- **Real-photo specificity ≥ 0.90** — **Not met yet.** The measured baseline is 0.30. Requires WI-1 (done) *combined with* real-generator training data (WI-7, deferred). The code path is now ready to train on real corpora with degradation augmentation.
- **AI recall ≥ 0.80** — **Not met yet.** Same dependency on real-generator training data.
- **No hard `is_ai=True` when agreement < 0.5 and calibrated prob < 0.9** — **MET** (WI-2 implemented + regression-locked).
- **Degraded-state accuracy within ±0.10 of pristine** — **Not met yet.** Requires real-generator training with augmentation (WI-1 + WI-7).
- `performance.json` reports **non-empty `per_generator`/`per_resolution`** — **Not met yet** (WI-4 deferred).
- **CI golden gate fails on regression** — **PARTIAL**: agreement/threshold gates added; the full specificity/recall gate is deferred behind the real-generator corpus.

---

## 6. Out of scope / explicitly deferred

- **Per-generator verdicts from third-party web tools** — the automation environment lacks a file-upload primitive; vendor methodology was captured from public docs. Closing the generator blind-spot dimension requires a user-supplied manifest of real generator outputs (Path A).
- **Adding a spatial CNN head** is deferred behind WI-5's group-soft-voting (which reuses existing classical learners); a true CNN is a larger, separate effort.
- **C2PA / provenance parsing** is deferred — it is a metadata signal, not an inference-code change, and requires the C2PA dependency + manifest-bearing inputs.
- **WI-4 (per-generator/per-resolution metrics), WI-5 (group soft-voting/PRNU bias), WI-7 (real-generator fetch/eval path)** are deferred — they need the real-generator corpus present in the repo to be meaningful, and to avoid over-engineering before that data lands.
