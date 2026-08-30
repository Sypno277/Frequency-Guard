# Frequency Guard v2 — Architecture

## System overview

```
┌────────────────────────────────────────────────────────────────────┐
│  INGESTION LAYER (React 18 + TS + Vite)                            │
│  Single upload · Batch/folder · Format validation                  │
└──────────────────────────────┬─────────────────────────────────────┘
                               │ REST /fetch, async job polling
┌──────────────────────────────▼─────────────────────────────────────┐
│  API LAYER — FastAPI (CPU-only)                                    │
│  /api/v1/analyze · /batch · /jobs/{id} · /jobs/{id}/csv           │
│  /metrics · /model · /model/performance · /history · /health      │
│  In-process ThreadPoolExecutor job queue · LRU feature cache      │
├────────────────────────────────────────────────────────────────────┤
│  PREPROCESSING ── cv2 decode, EXIF-safe, letterbox resize 256      │
├────────────────────────────────────────────────────────────────────┤
│  FEATURE EXTRACTION (numpy · scipy · pywavelets · scikit-image)    │
│  · FFT: log-spaced radial + azimuthal energy profiles,             │
│    spectral slope/flatness, phase entropy, peak prominence         │
│  · DCT 8×8: coefficient stats, high-freq anomaly, upsampling peaks │
│  · Wavelet db4/sym8: sub-band energy/entropy/ratios                │
│  · SRM noise residual: kurtosis, std, block consistency            │
│  · GLCM texture + box-counting fractal dimension                   │
│  → 89-dim feature vector (~<150ms warm)                            │
├────────────────────────────────────────────────────────────────────┤
│  FUSION & CLASSIFICATION (scikit-learn)                            │
│  RF + SVM-RBF + LR + GradientBoosting → probability average        │
│  → isotonic calibration → FPR-tuned threshold gate                 │
│  → generator-family attribution (softmax over real/diffusion/gan/other) │
├────────────────────────────────────────────────────────────────────┤
│  EXPLAINABILITY ── windowed spectral-saliency + patch-inconsistency│
│  → base64 PNG overlay (no gradients, CPU-cheap)                    │
├────────────────────────────────────────────────────────────────────┤
│  LOGGING · CACHE · HISTORY                                         │
│  Structured JSON logs · LRU feature cache · SQLite audit history   │
└────────────────────────────────────────────────────────────────────┘
```

## Data flow (single image)

```
image → decode/validate → preprocess → [FFT | DCT | wavelet | SRM | GLCM]
      → FeatureBundle → ensemble predict → calibrate → threshold gate
      → verdict + confidence + family attribution + saliency heatmap
      → API response → dashboard render → SQLite audit row
```

## Data flow (batch)

```
files → jobs/{batch_id} → ThreadPoolExecutor workers → per-image pipeline
      → results table → CSV/PDF report → history rows
```

## Backend modules (`python_services/frequency_guard/`)

| Module | Responsibility |
|---|---|
| `config.py` | environment-driven `Settings` (dataclass), `FG_*` env overrides; zero hardcoded paths. `ensure_dirs()` creates model/report/data dirs idempotently. |
| `logging.py` | JSON-lines structured logging; `get_logger`, `with_fields`; no `print()` in library code. |
| `io/loader.py` | cv2 lazy decode, format whitelist, EXIF orientation (tag 0x0112) applied via numpy/opencv, corrupted/oversize guards raise typed `LoadError`. |
| `preprocess.py` | vectorized gray/RGB/YCrCb conversion + aspect-preserving letterbox resize. |
| `features/fft_features.py` | `scipy.fft.rfft2`, log-spaced radial bins, azimuthal averaging, least-squares spectral slope, flatness, phase entropy, high-freq ratio, peak prominence. |
| `features/dct_features.py` | 8×8 block DCT statistics: kurtosis, high-freq/mid-band ratios, boundary energy, low/high ratio. |
| `features/wavelet_features.py` | `pywt.wavedec2` db4/sym8: sub-band energy/entropy, detail ratio, cross-scale ratio, LL ratio. Rejects non-orthogonal wavelets. |
| `features/noise_features.py` | SRM convolution kernels, residual kurtosis/std/mean-abs, block-consistency, flatness. |
| `features/texture_features.py` | skimage GLCM (contrast/correlation/energy/homogeneity/dissimilarity) + box-counting fractal dimension. |
| `features/extractor.py` | one-pass orchestrator producing a frozen `FeatureBundle`; bounded thread-safe `FeatureCache` keyed by (source, size, settings signature). |
| `models/classifier.py` | `EnsembleConfig` + `EnsembleClassifier` blending RF/SVM-RBF/LR/GB by probability average; joblib save/load. |
| `models/calibration.py` | isotonic (default) or Platt/sigmoid calibration, ECE + Brier computation, joblib persistence. |
| `models/attribution.py` | softmax over `real/diffusion/gan/other`; supervised per-family logistic heads trained on OOF probs, heuristic fallback when no family labels exist. |
| `explainability/heatmap.py` | tiled FFT spectral saliency + patch-inconsistency map, viridis colormap → base64 RGBA PNG overlay. |
| `api/schemas.py` | Pydantic v2 request/response models (the wire contract the dashboard consumes). |
| `api/inference.py` | bytes→load→preprocess→features(cached)→predict→calibrate→attribute→explain; model lifecycle (`ensure_model`, `reload`, metadata). |
| `api/server.py` | FastAPI app factory; single-image + batch endpoints, CSV export, metrics/model/history; batch jobs on `ThreadPoolExecutor`. |
| `api/history.py` | SQLite audit store (one row per analysis), thread-safe, `recent`/`stats`/`clear`. |
| `training/train_pipeline.py` | manifest→cache features→stratified k-fold CV (by label/generator)→ensemble→isotonic calibrator→persist artifacts + report. |
| `evaluation/evaluate.py` | accuracy/precision/recall/F1, ROC+PR, confusion matrix, ECE, per-generator slices, latency p50/p95/p99, peak RSS; writes `reports/benchmark_<ts>.json` + `metrics_history.json`. |

## Frontend modules (`src/`)

| Module | Responsibility |
|---|---|
| `lib/api/client.ts` | typed REST client mirroring backend schemas (`analyzeImage`, `submitBatch`, `pollJob`, `getMetrics`, `getModelInfo`, `getPerformance`, `getHistory`). |
| `lib/services/adaptiveInference.ts` | maps a real `AnalyzeResponse` → `AdaptiveAnalysisResult` (verdict card + forensic parameter grid). No randomness. |
| `lib/services/analysisPresentation.ts` | maps real payloads → Recharts-ready spectra/wavelets/azimuthal series + sorted family attribution. |
| `lib/services/pdfReport.ts` | dependency-free PDF 1.4 generator for batch results (Helvetica/Courier Type1, paginated table). |
| `components/dashboard/*` | `LiveMetrics`, `BatchAnalysis` (CSV+PDF+JSON export), `ModelPerformance`, `HistoryPanel`. |
| `pages/Index.tsx` | upload → real backend analysis → render result (no artificial delay). |
| `pages/Dashboard.tsx` | composes live metrics/batch/performance/history panels. |
| `pages/Training.tsx` | two-stage intake (forensic analysis + weak labeling), embedding visualizer, model management. |

## Testing strategy

```
tests/python/
├── unit/             every extractor, classifier, calibrator, attributor, harness, history, server, inference
├── integration/      image-in → verdict-out with real model artifacts (demo)
├── regression/       golden-dataset feature-vector + verdict locks (drift < 2%)
├── robustness/       JPEG/resize/crop/noise stability — verdict direction must not flip
└── performance/      latency <150ms/img + peak RSS <2GB gates (perf-marked, run separately)
```

Frontend: `src/**/*.test.{ts,tsx}` via vitest (node env + minimal DOM). CI runs
ruff/black/mypy/pytest(+coverage ≥90%)/vitest/tsc/build/pip-audit/npm-audit.

## Signal-processing theory notes

- **Spectral slope (α)** — fitting `log P ~ -α log f` on the radial power spectrum.
  Natural photography follows ≈1.9–2.2; generators that flatten or boost
  mid/high bands deviate measurably. Extracted by least squares over log-spaced bins.
- **Azimuthal anisotropy** — averaging power by angle exposes directional
  upsampling artifacts invisible in a purely radial profile.
- **Checkerboard/upsampling periodicity** — conv-transpose upsampling leaves peaks
  at ω≈π/2 harmonics; captured by the FFT peak-prominence feature.
- **DCT high-frequency anomaly** — 8×8 block-DCT tail statistics expose diffusion
  spectral flattening and double-JPEG fingerprints.
- **Wavelet cross-scale regularity** — sub-band energy/entropy ratios differ between
  smooth camera structure and generator artifacts.
- **SRM noise-print consistency** — real sensor noise is spatially consistent;
  synthetic noise is not. Quantified via residual kurtosis + per-block variance.
- **Patch-level inconsistency** — dividing into tiles and measuring inter-tile
  variance of local spectral slope flags regionally inconsistent synthesis
  (used for both the heatmap and an explainability score).

## Configuration

All runtime settings come from `config.Settings`, overridable via environment
variables prefixed `FG_` (e.g. `FG_IMAGE_SIZE=256`, `FG_LOG_LEVEL=DEBUG`,
`FG_THRESHOLD_FPR_TARGET=0.02`, `FG_BATCH_WORKERS=2`, `FG_CACHE_SIZE=256`).
No module hardcodes a path or parameter.
