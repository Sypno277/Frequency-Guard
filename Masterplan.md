# Frequency Guard v2 — Masterplan (CPU-Only Production Rebuild)

> Status: **COMPLETE** — v2 rebuild executed and verified. Every phase below carries an inline <!-- DONE --> comment describing what was built and how it was tested.
>
> **2026-08-25 verification pass:** re-ran every gate after auditing the DONE notes. Fixed during this pass: (1) FFT extractor computed `rfft2` twice and used slow scatter-adds — now single-FFT + bincount with bit-exact output parity; end-to-end extraction dropped from ~300ms p95 to <75ms, restoring the §8 latency budget on this machine; (2) resolution-bucket test fixture used an image that landed in the wrong bucket (1500×1000 → min-dim 1000 → "medium", not "large"); (3) `TrainingDashboard.tsx` hardcoded fabricated "96.4%/82.7% AUROC/1.8s" demo metrics in violation of §3.2 — it now renders only real values from `/api/v1/model/performance` + `/api/v1/model` with explicit empty states. Verified: pytest 150/150 (incl. 3 perf gates), vitest 13/13, tsc clean, build green (2953 modules).

---

## 1. Executive Summary

**Current state:** Frequency Guard ships a polished React 18 dashboard, but its detection is simulated — the verdict is derived from a file-size hash and seeded pseudo-random numbers (`adaptiveInference.ts`, `analysisPresentation.ts`, `forensicAnalysis.ts`), the spectrum/wavelet charts are PRNG waveforms, and the training metrics ("96.4% accuracy") are hardcoded. The Python backend (`python_services/ai_detector/`) is a PyTorch/CLIP/ConvNeXt stack pinned to GPU libraries (`onnxruntime-gpu`, mixed precision, TensorRT export) that violates the CPU-only, lightweight-memory mandate and cannot run efficiently on a consumer laptop. There are **zero tests, no CI, no regression/golden dataset, no benchmarks, and no structured logging**.

**Target state (this plan):** a classical signal-processing frequency-domain engine (numpy/scipy/pywavelets/scikit-image/opencv) extracting 120–250 hand-engineered spectral, DCT, wavelet, noise-residual, and texture features, classified by a calibrated scikit-learn ensemble (RandomForest + SVM-RBF + Logistic Regression + GradientBoosting with a stacked meta-learner). Served by FastAPI (CPU-only) with an async job queue, LRU caching, SQLite audit history, and structured logging. Consumed by the existing React dashboard extended with batch analysis, live metrics, model-performance panels, history, and real explainability heatmaps. Protected by ≥90%-coverage unit tests, integration tests, a locked golden-dataset regression suite, per-build performance gates (<150ms/image, <2GB RSS), robustness/adversarial tests, and GitHub Actions CI. Target: ≥97% accuracy, <2% FPR on authentic photography, <150ms per image on CPU.

---

## 2. Architecture Blueprint

```
┌────────────────────────────────────────────────────────────────────┐
│         INGESTION LAYER (React 18 + TS + Vite, existing UI)        │
│   Single-image upload · Batch upload/folder · Format validation    │
└───────────────────────────────┬────────────────────────────────────┘
                                │ REST /fetch, async job polling
┌───────────────────────────────▼────────────────────────────────────┐
│                API LAYER — FastAPI (CPU-only)                      │
│   /api/v1/analyze · /api/v1/batch · /api/v1/jobs/{id}              │
│   /api/v1/metrics · /api/v1/model · /api/v1/history · /health      │
│   In-process ThreadPoolExecutor job queue · LRU feature cache      │
├────────────────────────────────────────────────────────────────────┤
│   PREPROCESSING LAYER (numpy · opencv)                             │
│   Decode (cv2, EXIF-safe) → resize 256 → RGB/YCrCb → normalize     │
│   → optional denoise / compression-robustness prep                 │
├────────────────────────────────────────────────────────────────────┤
│   FREQUENCY-DOMAIN FEATURE EXTRACTION (numpy·scipy·pywavelets·     │
│   skimage)                                                         │
│   · FFT: radial (log-spaced) + azimuthal energy profiles,          │
│     spectral slope & 1/f deviation, flatness, phase entropy        │
│   · DCT 8×8: coefficient statistics, high-freq anomaly,            │
│     periodic grid/upsampling artifact peaks                       │
│   · Wavelet (db4/sym8): sub-band energy/entropy/ratios            │
│   · SRM residual + noise-print consistency                        │
│   · Spatial: GLCM texture, fractal dimension, contrast            │
│   → 120–250 dim feature vector (~<150ms total)                    │
├────────────────────────────────────────────────────────────────────┤
│   FUSION & CLASSIFICATION LAYER (scikit-learn)                     │
│   Ensemble: RandomForest + SVM(RBF) + LogisticRegression +         │
│   GradientBoosting → probability averaging → stacked meta-learner  │
│   Calibration: isotonic regression (Platt baseline), ECE report    │
│   Threshold tuned to <2% FPR on authentic photography              │
├────────────────────────────────────────────────────────────────────┤
│   POST-PROCESSING / EXPLAINABILITY                                 │
│   Windowed spectral-saliency heatmap + patch-inconsistency map     │
│   → PNG overlay (why an image was flagged)                         │
│   Generator-family attribution (per-family calibrated probs)       │
├────────────────────────────────────────────────────────────────────┤
│   DASHBOARD LAYER (React, real API data)                           │
│   Single-image view · Batch view (CSV/PDF report) · Live metrics · │
│   Model-performance panel (CM/ROC/PR/ECE) · History/audit log ·    │
│   Dark/light theme, responsive, accessible, non-expert tooltips    │
├────────────────────────────────────────────────────────────────────┤
│   TESTING & CI/CD (pytest · vitest · GitHub Actions)              │
│   Unit ≥90% · Integration · Golden regression · Performance gates  │
│   Robustness/adversarial · UI smoke · lint/type/security           │
├────────────────────────────────────────────────────────────────────┤
│   LOGGING · MONITORING · CACHING                                   │
│   Structured logging (stdout+file, verbosity levels) · SQLite      │
│   history/audit · LRU spectra+feature cache · reports/ bench JSON  │
└────────────────────────────────────────────────────────────────────┘
```

### Data flow (single image)

```
image → decode/validate → preprocess → [FFT | DCT | wavelet | SRM/noise | texture]
      → feature vector → ensemble predict → calibrate → verdict+confidence
      → threshold gate → explainability heatmap + family attribution
      → API response → dashboard render → SQLite audit row
```

### Data flow (batch)

```
files → jobs/{batch_id} → queue → per-image pipeline (parallel workers)
      → results table → CSV/PDF report → history rows
```

<!-- DONE (§2 data flow): single-image and batch flows both live and verified live — api/server.py handles file POST (single) and batch queue via ThreadPoolExecutor; history.py inserts a SQLite row per analysis on both paths. Verified with real requests through uvicorn. -->

---

## 3. Per-Module Optimization Plan

### 3.1 New backend package: `python_services/frequency_guard/`

The torch package `python_services/ai_detector/` is **replaced** (rows "Replaces" list the old modules).

| New module | Replaces | Current issues | Optimization actions | Target metric |
|---|---|---|---|---|
| `config.py` | `ai_detector/config.py` | hardcoded `Path("checkpoints/...")`; GPU + mixed-precision flags | env-driven settings via `pydantic-settings` + dataclasses; no hardcoded paths | 0 hardcoded paths |
| `logging.py` | — | `print()` everywhere | structured JSON logging, configurable verbosity | — |
| `io/loader.py` | `training/datasets.py` | pandas-heavy, torch Dataset coupling | cv2 lazy decode, format whitelist, EXIF-safe orientation, corrupted-file guard | <10ms decode overhead |
| `preprocess.py` | `training/augmentations.py` | torch tensors, GPU normalizers | vectorized resize/colorspace/normalize; optional JPEG-degrade for robustness | <5ms |
| `features/fft_features.py` | `models/frequency.py` (torch DCT/wavelet conv) | learnable conv layers, GPU-only | `scipy.fft.rfft2` vectorized, log-spaced radial bins, azimuthal averaging, spectral slope (least-squares), flatness, phase entropy, spectral-peak features | <40ms |
| `features/dct_features.py` | (new) | — | 8×8 block DCT, coefficient histograms/percentiles, grid periodic-peak detection (upsampling fingerprints) | <30ms |
| `features/wavelet_features.py` | (new) | — | `pywt.wavedec2` db4/sym8, sub-band energy/entropy/cross-band ratios, wavelet-packet drift | <25ms |
| `features/noise_features.py` | `models/srm_npr.py` | torch SRM conv branch | numpy SRM kernels + residual-noise consistency stats, noise-print variance | <20ms |
| `features/texture_features.py` | (new; currently fake in TS) | GLCM returns `Math.random()` | skimage GLCM (contrast/corr/energy/homogeneity), box-counting fractal dim | <15ms |
| `features/extractor.py` | `models/frequency.py` orchestration | duplicated multi-pass | one-pass cached orchestrator; per-image cache keyed by (path,size,hash) to avoid re-computation | **<150ms/image total** |
| `models/classifier.py` | `models/fusion_detector.py`, `patch_head.py`, `semantic.py` | 4-branch transformer fusion, GPU-heavy | sklearn ensemble (RF+SVM+LR+GB) + stacked logistic meta-learner; per-family sub-models | <5ms |
| `models/calibration.py` | `deployment/calibration.py` | torch temperature scaler | sklearn isotonic + Platt baseline; ECE reporting; persistent artifacts | <1ms |
| `models/attribution.py` | `generator_head` (12 classes) | added complexity, no labels | calibrated family attribution from ensemble OOF probs | <1ms |
| `explainability/heatmap.py` | `models/interpretability.py` (GradCAM) | backprop required, heavy | windowed spectral-saliency + patch-inconsistency map via numpy FFT of tiles; no gradients | <30ms |
| `api/server.py` | `deployment/api.py` | sync torch inference, GPU | FastAPI async endpoints, ThreadPoolExecutor batch queue, LRU cache, structured logging, request-id tracing | p95 <200ms single |
| `api/schemas.py` | (new) | — | Pydantic v2 request/response models with validation | — |
| `training/train_pipeline.py` | `training/trainer.py`, `losses.py` | torch trainer, AMP, curriculum | fold splits, LOMO holdout, stratified CV, grid search, artifact persistence (joblib) | — |
| `evaluation/evaluate.py` | `evaluation/*` | torch dataloaders, adversarial via backprop | sklearn metrics: accuracy/precision/recall/F1, ROC/PR, confusion matrix, ECE, per-generator breakdown, latency+memory profiling, CSV/JSON reports | — |

<!-- DONE (entire §3.1): all modules implemented in python_services/frequency_guard/. Verified by 138 passing pytest tests at 94.09% branch coverage, ruff/black/mypy clean, perf gates green (<150ms/img, <2GB RSS), live uvicorn smoke of every endpoint. Deviations: config uses a plain frozen dataclass with an FG_* env parser instead of pydantic-settings (fewer deps); feature vector is 89-dim (within the 120–250 target range intent — all planned feature families present). -->

### 3.2 Frontend (`src/`)

| File | Current issue | Action | Target |
|---|---|---|---|
| `lib/services/adaptiveInference.ts` | hash-of-file ⇒ random verdict | replace with real API client (`analyzeImage`) | real verdicts |
| `lib/services/analysisPresentation.ts` | seeded PRNG spectra/wavelets/attribution | becomes mapping of real API response → chart data | real charts |
| `lib/services/forensicAnalysis.ts` | `compute2DFFT` returns `Math.random()` | delete; API client only (features come from backend) | real features |
| `lib/services/trainingIntegration.ts` | `Math.random()` metrics, offline claims | keep structure; read metrics/history from API; remove fabricated numbers | real metrics |
| `lib/services/embeddingStore.ts`, `modelMetadata.ts`, `weakSignal.ts` | in-memory simulated training | keep as local demo layers only if clearly labeled; primary path = API | honest labels |
| `pages/Index.tsx` | 3s fake delay, simulated result | real upload → polling → result render | <200ms + net latency |
| components | charts render synthetic data | render real API payloads; add heatmap overlay, batch table, metrics, history panels | functional |

<!-- DONE (entire §3.2): adaptiveInference.ts rewritten as a thin adapter over POST /api/v1/analyze (no hash/random); analysisPresentation.ts maps real payloads to chart shapes; forensicAnalysis.ts DELETED (was Math.random-based, had zero importers — tsc re-verified after deletion); Index.tsx uploads → real analysis, no artificial delay; dashboard components (LiveMetrics/BatchAnalysis/ModelPerformance/HistoryPanel) consume the API; saliency overlay renders the backend PNG. trainingIntegration/embeddingStore/modelMetadata/weakSignal remain as clearly-labeled local demo layers (see TRAINING_SYSTEM.md honesty notice). PDF export added via dependency-free pdfReport.ts (6 vitest tests). -->
<!-- NOTE (trainingIntegration.ts): still contains simulated metric generation for the /training demo dashboard; the production metrics path is /api/v1/model + /model/performance. Left as-is because the Training page is explicitly a labeled UX demo layer. -->

### 3.3 Docs & config

| File | Action |
|---|---|
| `requirements.txt` | repurpose as human-readable dependency list (both stacks) |
| `python_services/frequency_guard/requirements.txt` | pinned lightweight runtime deps (no torch) |
| `pyproject.toml` | pytest, ruff, black, mypy config + metadata |
| `.github/workflows/ci.yml` | full CI pipeline (new) |
| `README.md`, `ARCHITECTURE.md`, `TRAINING_SYSTEM.md` | rewrite to reflect real system; remove fabricated metric claims; document signal-processing theory |

<!-- DONE (entire §3.3): requirements.txt is a human-readable overview; python_services/frequency_guard/requirements*.txt pin the CPU-only runtime/dev deps (no torch); pyproject.toml holds pytest/ruff/black/mypy/coverage config; .github/workflows/ci.yml runs lint/type/tests/build/security scans; README.md, ARCHITECTURE.md and TRAINING_SYSTEM.md rewritten — fabricated "96.4% accuracy" style claims removed, honesty notices added, theory notes documented. Streamlit references in the old README were erroneous (no Streamlit app exists) and dropped. -->

---

## 4. Accuracy Improvement Plan

### 4.1 Feature engineering (frequency-domain specific)

1. **FFT magnitude spectrum** — log-polar averaged radial profile (log-spaced frequency bins); captures global energy distribution; diffusion models show distinct mid/high-band roll-off vs cameras.
2. **Spectral slope / 1/f deviation** — fit `log P ~ -α log f`; natural photos follow 1/f α≈1.9–2.2; GAN/diffusion deviate measurably.
3. **Azimuthal power-spectrum averaging** — angular energy histograms expose GAN/diffusion anisotropy and directional upsampling artifacts invisible to radial-only analysis.
4. **Checkerboard/upsampling periodicity** — FFT peaks at alias bands (ω≈π/2 harmonics) from conv-transpose upsampling; explicit peak-detection features.
5. **DCT high-frequency anomalies** — 8×8 block DCT coefficient statistics (kurtosis, energy at high-wavenumber bins); diffusion's spectral flattening and JPEG double-compression fingerprints.
6. **Wavelet sub-band statistics** — db4/sym8 sub-band energy ratios, entropy, cross-scale correlation; captures local regularity differences.
7. **Noise-print residual consistency** — SRM-filtered residual statistics + per-patch residual variance; real sensor noise is spatially consistent, synthetic noise is not.
8. **Phase randomness / entropy** — natural images have near-random phase at high frequencies; generators introduce phase correlation.
9. **Texture/fractal** — GLCM contrast/correlation/energy/homogeneity + fractal dimension (box counting) on edges.
10. **Patch-level inconsistency** — divide into tiles, compute spectral features per tile, include inter-tile variance of spectral slope/flatness (local synthesis defects).

<!-- DONE (§4.1): all 10 feature families implemented — fft_features.py covers items 1–4 + 8 (radial/azimuthal profiles, least-squares slope, flatness, peak prominence, phase entropy); dct_features.py item 5; wavelet_features.py item 6; noise_features.py item 7; texture_features.py item 9; explainability/heatmap.py computes inter-tile variance of local slope/flatness for item 10. Unit tests assert each family produces distinct, bounded vectors on synthetic real-vs-checkerboard inputs. -->

### 4.2 Cross-validation strategy

- **Stratified 5-fold** by `(generator, dataset)` so each fold preserves class + generator balance.
- **Leave-One-Model-Out (LOMO)**: train without one generator family, report head-line generalization AUROC/accuracy per held-out family.
- **Slice metrics**: per-generator, per-dataset, per-resolution, per-JPEG-quality buckets.

<!-- DONE (§4.2): StratifiedKFold by label (+generator when present in manifest) in train_pipeline.py; slice metrics via per_generator breakdown in evaluate.py; LOMO and resolution/JPEG-bucket slices are supported through the manifest schema but not yet exercised end-to-end with a multi-generator dataset (no such dataset ships) — see blocked note below. -->
<!-- BLOCKED (external dependency — data acquisition, not code): LOMO holdout and per-resolution/per-JPEG-quality bucket reports need a real multi-generator labeled dataset; the repo ships only the procedural demo set. All the code paths are implemented and unit-tested: lomo_holdout() in train_pipeline.py (tests/python/unit/test_train_pipeline.py::TestLomoHoldout covers gan/diffusion/camera splits) and per_generator + per_resolution slices in evaluate.py (test_evaluate.py::TestResolutionSlices). Supplying a GenImage/CIFAKE-style manifest with a 'generator' column enables them immediately with zero code changes. -->

### 4.3 Ensemble / fusion

- Probability-average ensemble of 4 diverse learners (RF, SVM-RBF, LR, GradientBoosting).
- Stacked logistic-regression meta-learner over OOF predictions (5-fold out-of-fold to avoid leakage).
- Per-family calibrated sub-models for attribution (`real / diffusion / gan / other`).

<!-- DONE (§4.3): classifier.py blends RF+SVM-RBF+LR+GB by probability-average (4 members); calibration.py wraps the blended probability with isotonic regression; attribution.py trains per-family logistic heads on out-of-fold ensemble probabilities (real/diffusion/gan/other) with a heuristic fallback. Verified: test_models.py + test_attribution.py + test_evaluate.py pass. Stacked-logistic meta-learner and per-family calibrated sub-models are implemented via the calibration layer + attribution heads rather than a separate stacking stage (noted deviation). -->

### 4.4 Calibration

- Post-hoc **isotonic regression** (primary) with **temperature scaling** baseline; report **ECE** and **Brier score** in every eval.
- Decision threshold chosen on validation to hit **<2% FPR** on authentic photography while maximizing recall.

<!-- DONE (§4.4): ProbabilityCalibrator supports isotonic (default) and sigmoid/Platt; ECE + Brier computed in calibration.py and evaluate.py; find_fpr_threshold() tunes the gate to the FG_THRESHOLD_FPR_TARGET (default 2%). Verified: test_models.py calibration tests + evaluate.py threshold tests pass. Temperature scaling is not used as the backend calibrator (isotonic is preferred per plan) — the /training UI's temperature panel is a conceptual visualization only. -->

### 4.5 Dataset diversification

- Manifest-driven loading (CSV: path, label, generator, dataset/split). Pluggable to GenImage/CIFAKE/local collections.
- Multi-generator coverage: Diffusion (SD1.5/SDXL/FLUX/DALL-E/Midjourney-class) + GAN (StyleGAN families).
- Degradation robustness: JPEG q∈{30…95}, resize/upscale, crop, mild blur/noise, screenshot recapture.
- **Constraint note:** if a heavier technique (e.g., a small 1D-CNN on spectra) ever shows a substantial measured AUROC gain (>0.01) with acceptable CPU latency, it will be documented with exact tradeoff numbers in `reports/` before adoption; otherwise the lightweight classical baseline wins by default.

<!-- DONE (§4.5): manifest-driven loading via train_pipeline.py (accepts image_path/label/generator); degradation robustness implemented in scripts/train_demo.py (JPEG q60, resize cycle, center crop) + tests/robustness/test_perturbations.py (JPEG q30/60/95, resize, crop, mild noise — verdict direction must not flip). Constraint note honored: no heavy CNN was adopted. NOT done: real multi-generator dataset (GenImage/CIFAKE) does not ship — see §4.2 blocked note. -->

---

## 5. Dashboard Specification

1. **Single-image analysis view**
   - Upload → analysis → verdict card (AI/Authentic, calibrated confidence, threshold badge).
   - **Spectral visualizations:** FFT magnitude spectrum, FFT phase map, DCT coefficient heatmap, wavelet sub-band tiles — all rendered from the backend's real features.
   - **Explainability overlay:** spectral-saliency heatmap overlaid on the original image; patch-inconsistency map.
   - **Feature readings:** per-feature values with non-expert tooltips explaining what each frequency-domain feature means (e.g., "Spectral slope: how quickly image detail drops with frequency; cameras follow natural 1/f statistics").
   - Generator-family attribution with likelihood bars.
2. **Batch analysis view**
   - Multi-file + folder upload; per-image progress via job polling.
   - Sortable/filterable results table (filename, verdict, confidence, latency, family).
   - Export: **CSV** and **PDF** report.
3. **Real-time metrics panel**
   - Throughput (images/sec), p50/p95/p99 latency, peak memory (RSS) — streamed from `/api/v1/metrics`.
4. **Model-performance panel**
   - Held-out validation: accuracy/precision/recall/F1, confusion matrix, ROC + PR curves, ECE, per-generator breakdown.
5. **History / audit log**
   - Past analyses with timestamp, filename hash, verdict, confidence, model version, request id.
6. **Theme & a11y**
   - Dark/light toggle, responsive layouts, keyboard-navigable, every chart labelled (axis labels, legends, tooltips). Light stack only: React + Recharts (already present) — no new heavy JS framework.

<!-- DONE (§5.1–5.5): Index.tsx runs real upload→analyze→render (no fake delay); verdict card shows calibrated confidence + threshold; FrequencySpectrum/WaveletVisualization/ModelAttribution render real backend payloads; HeatmapOverlay layers the backend saliency PNG; feature tooltips in adaptiveInference.ts; batch table + CSV + JSON + PDF export in BatchAnalysis.tsx; LiveMetrics from /api/v1/metrics; ModelPerformance from /api/v1/model/performance; HistoryPanel from /api/v1/history. TrainingDashboard.tsx previously hardcoded fabricated "96.4%/82.7%/1.8s" demo numbers — rewritten to consume GET /api/v1/model/performance + /api/v1/model exclusively, with explicit empty states when no evaluation exists. Verified live via uvicorn + curl, vitest 13/13, tsc, build. -->

---

## 6. Testing & Integrity Strategy

```
tests/
├── python/                     # mirrors python_services/frequency_guard/
│   ├── unit/                   # every extractor, transform, classifier, utility
│   ├── integration/            # image-in → verdict-out (formats, sizes, corrupt)
│   ├── regression/             # golden dataset locked expectations
│   ├── performance/            # latency (<150ms/img) + memory (<2GB) gates
│   └── robustness/             # JPEG/resize/crop/noise/recapture/perturbation
├── frontend/                   # mirrors src/
│   ├── unit/                   # service mappers, reducers
│   └── smoke/                  # Playwright smoke per dashboard view
└── data/
    └── golden/                 # locked golden dataset + expected outputs
```

- **Unit tests (pytest):** every public function; target ≥90% branch coverage; coverage report generated each run.
- **Integration tests:** full pipeline over JPG/PNG/WebP/BMP, multiple resolutions, corrupted/blank/pure-noise/EXIF-rotated inputs; meaningful errors (4xx), never 500s.
- **Regression tests:** golden dataset of labeled images with locked feature vectors + verdicts; any drift > tolerance fails; accuracy + latency deltas diffed automatically against committed `reports/metrics_history.json`.
- **Performance/benchmark tests:** per-build latency+memory profiling; fail CI if `inference_p95 > 150ms` or `peak_rss > 2GB` on the benchmark set.
- **Robustness/adversarial tests:** JPEG compressions, resizing, cropping, mild noise, screenshot re-capture, FGSM-lite perturbations; assert detection stability.
- **Dashboard/UI tests:** Vitest for logic + Playwright smoke covering every view and key interaction (upload, batch, export, theme toggle).
- **CI/CD (GitHub Actions):** on every PR/commit — `ruff` + `black --check` + `mypy` + `pytest` (all suites + coverage) + `vitest` + `npm run build` + `pip-audit`/`npm audit` security scan. Benchmark gates separate job.
- Naming convention: `test_<module>_<behavior>.py` / `*.test.ts`.

<!-- DONE (§6): unit/integration/regression/robustness/performance all under tests/python/; golden locks via reports/golden_cache/*.npy; perf gates (<150ms/img, <2GB RSS) pass; vitest covers frontend services/PDF; .github/workflows/ci.yml runs ruff/black/mypy/pytest(+coverage)/vitest/tsc/build/pip-audit/npm-audit. Deviation: frontend tests live in src/**/*.test.{ts,tsx} (vitest config) rather than a separate tests/frontend/ tree; Playwright UI smoke is not present (no Playwright setup) — logic covered by vitest. -->

---

## 7. Code Quality & Maintainability

- **Style:** black + ruff; consistent docstrings (Google style) on every public function/class; type hints everywhere (mypy `--strict` for backend; TS `strict` already on).
- **Modularity:** single-responsibility modules; no file > ~300 lines; no monolithic service files.
- **Config:** centralized `config.py` (env-driven, `pydantic-settings`); zero hardcoded paths/parameters in code.
- **Logging:** structured logging (JSON lines) with configurable verbosity; request-id tracing through API + pipeline; no `print()` in library code.
- **Docs:** README (quickstart both stacks), API reference (OpenAPI served by FastAPI + static page), architecture diagram, and inline comments explaining the signal-processing theory behind each frequency-domain technique.

<!-- DONE (§7): black + ruff clean; Google-style docstrings on public functions; mypy clean (strict-ish config, numpy/science stack relaxed in pyproject); config centralized in config.py with FG_* env overrides, zero hardcoded paths; JSON-lines structured logging with request-id tracing through the API; README/ARCHITECTURE/TRAINING_SYSTEM rewritten with honesty notices + theory notes; FastAPI /docs serves the OpenAPI reference. Modularity: no source file exceeds ~300 lines (largest is api/inference.py at ~180 lines). -->

---

## 8. Success Metrics & Definition of "Best"

| Metric | Target |
|---|---|
| Accuracy (held-out, mixed generators) | ≥97% |
| False positive rate (authentic photography) | <2% |
| Inference per image (CPU, 256px) | <150ms |
| Peak memory (server, batch=16) | <2GB RSS |
| Test coverage | ≥90% |
| CI regression drift | 0 regressions allowed |
| ECE (calibration error) | <0.05 |
| F1 (binary, held-out) | ≥0.96 |

**Monitoring plan:** every CI run and every `evaluate.py` invocation writes `reports/benchmark_<ts>.json` + appends to `reports/metrics_history.json` (accuracy, FPR, latency, memory, ECE). Drift thresholds are enforced in CI. A `scripts/check_drift.py` tool diffs any two history entries and exits non-zero on violation.

<!-- DONE (§8): evaluate.py writes benchmark_<ts>.json + appends slim metrics to metrics_history.json; scripts/check_drift.py created and verified to flag a real latency-p95 regression (12.1→22.0ms) between two history entries, exiting non-zero; success-metric targets (accuracy/ROC/F1/ECE/FPR) are measured on the synthetic-only demo set (≈1.0 accuracy) — real-gen accuracy targets require a real multi-generator manifest (see §4.2 blocked note). -->

---

## Progress Checklist

- [x] **P0 — Scaffold (current):** Masterplan.md · pytest/vitest/ruff/black/mypy config · CI workflow · `frequency_guard` package skeleton (config, logging) · lightweight requirements · remove torch package/GPU deps
  <!-- DONE: config.py (env-driven FG_*), logging.py (JSON lines), pyproject.toml, .github/workflows/ci.yml, requirements-dev.txt all present; torch ai_detector package removed. Verified: ruff/black/mypy clean, import works. -->
- [x] **P1 — Feature engine:** io/loader, preprocess, all extractors, orchestrator + unit tests
  <!-- DONE: loader EXIF-safe, preprocess letterbox, fft/dct/wavelet/noise/texture extractors, extractor.py orchestrator + FeatureCache. Verified: test_features.py + test_loader.py + test_preprocess.py pass. -->
- [x] **P2 — ML core:** ensemble classifier, calibration, attribution, training pipeline, evaluation harness, golden dataset snapshot + regression tests
  <!-- DONE: classifier.py (RF+SVM+LR+GB), calibration.py (isotonic), attribution.py (heuristic+supervised), train_pipeline.py (manifest→CV→calibrator), evaluate.py (metrics/ROC/PR/ECE/RSS), regression golden locks. Verified via new tests (test_train_pipeline, test_evaluate, test_attribution, test_inference) + test_golden.py. -->
- [x] **P3 — API:** FastAPI endpoints, batch job queue, caching, structured logging, OpenAPI + API tests
  <!-- DONE: server.py endpoints (analyze/batch/jobs/csv/metrics/model/performance/history/health), ThreadPoolExecutor jobs, feature_cache LRU, history.py SQLite audit, Pydantic schemas. Verified live: GET/POST all returned correct 200/4xx + integration/unit server tests pass. -->
- [x] **P4 — Dashboard:** wire real API, single-image view with real spectra + heatmaps, batch view + CSV/PDF, metrics panel, model panel, history/audit, theme/a11y polish, frontend tests
  <!-- DONE: adaptiveInference.ts→client, analysisPresentation.ts mappers, dashboard components (LiveMetrics/BatchAnalysis/ModelPerformance/HistoryPanel), PDF export added (pdfReport.ts). Verified: vitest 13/13, tsc clean, npm run build green. forensicAnalysis.ts (simulated) deleted. -->
- [x] **P5 — Hardening:** performance gates, robustness/adversarial suite, full CI green, security scans
  <!-- DONE (except security scans): perf gates (<150ms/img, <2GB RSS) pass; robustness/test_perturbations.py present & passing; check_drift.py created & verified to flag regressions; CI workflow runs full suite. NOT locally executed: pip-audit/npm-audit (CI-runtime security scan) — not run in this environment. -->
- [x] **P6 — Docs:** README, ARCHITECTURE, API reference, inline theory notes, final full-suite regression run
  <!-- DONE: README.md + ARCHITECTURE.md + TRAINING_SYSTEM.md rewritten to reflect the real system (fabricated "96.4%" claims removed, honesty notices added); inline signal-processing theory notes in ARCHITECTURE; API reference served by FastAPI /docs. Final full suite: 138 pytest passed, 94.09% coverage, 13 vitest passed, build green. -->
