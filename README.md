# Frequency Guard v2

**CPU-only, frequency-domain AI-image detection.**

Frequency Guard classifies images as **AI-generated or authentic camera captures**
using classical signal processing — no GPU, no deep-learning runtime. Every verdict
is backed by measured spectral evidence: FFT radial/azimuthal profiles, 8×8 block-DCT
statistics, wavelet sub-band energies, SRM noise residuals, and GLCM texture features,
classified by a calibrated scikit-learn ensemble.

> **Honesty note:** the headline metrics below are produced by `scripts/train_demo.py`
> on a *procedurally generated* demo dataset (synthetic 1/f "camera" textures vs.
> checkerboard-artifacted "generator" images). They demonstrate that the full pipeline
> works end-to-end; they are **not** claims about real-world generator detection.
> Train on a real labeled manifest (GenImage / CIFAKE / your own) via
> `train_from_manifest` to get production numbers.

## How it works

```
image → decode (cv2, EXIF-safe) → letterbox resize 256 → RGB/YCrCb normalize
      → FFT radial + azimuthal profiles · spectral slope · flatness · phase entropy
      → DCT 8×8 coefficient stats · high-freq anomaly · grid-periodicity peaks
      → wavelet (db4) sub-band energy/entropy/ratios
      → SRM residual noise stats · noise-print consistency
      → GLCM texture · fractal dimension
      → 89-dim feature vector → RF + SVM-RBF + LR + GradientBoosting ensemble
      → isotonic calibration → FPR-tuned threshold gate
      → verdict + confidence + family attribution + spectral-saliency heatmap
```

Key signal-processing ideas (see `ARCHITECTURE.md` for details):

- **Spectral slope / 1/f law** — natural photos follow power-law decay α≈1.9–2.2;
  generators measurably deviate.
- **Checkerboard periodicity** — conv-transpose upsampling leaves peaks at ω≈π/2 harmonics.
- **Noise-print consistency** — real sensor noise is spatially consistent; synthetic noise is not.
- **Patch inconsistency** — inter-tile variance of local spectra exposes local synthesis defects.

## Quickstart

### Backend (FastAPI, CPU-only)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate on Unix)
pip install -r python_services/frequency_guard/requirements-dev.txt

# One-time: build demo dataset + train + calibrate + evaluate (~4 min)
python -m scripts.train_demo

# Serve
python -m uvicorn python_services.frequency_guard.api.server:app --port 8000
```

Endpoints (OpenAPI docs at `http://localhost:8000/docs`):

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/analyze` | single image → verdict + features + heatmap |
| `POST /api/v1/batch` | multi-file upload → async job id |
| `GET /api/v1/jobs/{id}` | poll batch progress/results |
| `GET /api/v1/jobs/{id}/csv` | batch results as CSV |
| `GET /api/v1/metrics` | live throughput/latency/RSS snapshot |
| `GET /api/v1/model` | model metadata |
| `GET /api/v1/model/performance` | held-out evaluation report |
| `GET /api/v1/history` | SQLite audit log |
| `GET /health` | liveness |

### Frontend (React 18 + Vite)

```bash
npm install
npm run dev          # http://localhost:5173
```

Set `VITE_API_BASE=http://localhost:8000` in `.env.local` if the API runs elsewhere.

Views:

- **Home** — upload → real verdict card, spectrum/wavelet/DCT charts rendered from
  backend payloads, saliency heatmap overlay, family attribution bars.
- **Dashboard** (`/dashboard`) — live metrics, batch analysis with CSV/PDF export,
  held-out performance panel (confusion matrix, ROC/PR), audit history.
- **Training** (`/training`) — two-stage intake (real forensic analysis + weak user
  labeling), embedding visualizer, model management.

## Training on your own data

Create a manifest CSV with at least `image_path,label` columns (label: 0=real, 1=fake;
optional `generator` column enables per-family attribution):

```csv
image_path,label,generator
photos/img_001.jpg,0,camera
gen/sd_001.png,1,diffusion
```

Then:

```python
from python_services.frequency_guard.training.train_pipeline import train_from_manifest

result = train_from_manifest("manifest.csv", output_dir="checkpoints")
print(result.metrics)
```

The pipeline runs stratified k-fold CV by label/generator, fits an isotonic calibrator
on out-of-fold predictions, tunes the decision threshold to your FPR target
(`FG_THRESHOLD_FPR_TARGET`, default 2%), and persists everything to `checkpoints/`.

## Testing & quality gates

```bash
pytest -m "not perf" --cov=python_services/frequency_guard   # unit+integration+regression+robustness, ≥90% coverage gate
pytest -m perf                                               # latency <150ms/img, RSS <2GB gates
ruff check python_services/frequency_guard tests/python
black --check python_services/frequency_guard tests/python
mypy python_services/frequency_guard
npx tsc --noEmit && npx vitest run && npm run build
python -m scripts.check_drift                                # diff benchmark history for regressions
```

CI (`.github/workflows/ci.yml`) runs all of the above plus pip-audit/npm-audit on every PR.

## Project structure

```
├── src/                          # React dashboard (real API client, no simulated data)
│   ├── components/dashboard/     # LiveMetrics · BatchAnalysis · ModelPerformance · HistoryPanel
│   ├── lib/api/client.ts         # typed REST client mirroring backend schemas
│   └── lib/services/             # presentation mappers + PDF report generator
├── python_services/frequency_guard/
│   ├── io/loader.py              # cv2 decode, EXIF orientation, format whitelist
│   ├── preprocess.py             # letterbox resize, color spaces
│   ├── features/                 # fft · dct · wavelet · noise(SRM) · texture(GLCM) extractors
│   ├── models/                   # sklearn ensemble · isotonic calibration · family attribution
│   ├── explainability/heatmap.py # gradient-free spectral-saliency maps
│   ├── api/                      # FastAPI server · inference service · SQLite history
│   ├── training/train_pipeline.py# manifest → CV → ensemble → calibration artifacts
│   └── evaluation/evaluate.py    # metrics, ROC/PR, ECE, latency/RSS profiling
├── tests/python/                 # unit · integration · regression(golden) · robustness · perf
├── scripts/
│   ├── train_demo.py             # end-to-end bootstrap on synthetic data
│   └── check_drift.py            # benchmark regression gate
└── reports/                      # benchmark_<ts>.json + metrics_history.json
```

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — system architecture + signal-processing theory
- **[TRAINING_SYSTEM.md](TRAINING_SYSTEM.md)** — training pipeline: CV, calibration, LOMO, slices
- **[QUICK_START.md](QUICK_START.md)** — 5-minute quick start (both stacks)
- **[VERCEL_DEPLOY.md](VERCEL_DEPLOY.md)** — Vercel frontend deployment

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/improvement`
3. Commit: `git commit -m 'Add improvement'`
4. Push: `git push origin feature/improvement`
5. Open a Pull Request

All contributions must keep the CPU-only, test-covered bar: `ruff`/`black`/`mypy` clean,
`pytest --cov` ≥ 90%, vitest green, `npm run build` green.
