# Training Section — UX Layer Documentation

> **⚠️ Honesty notice:** this documents the **frontend training UI at `/training`**.
> It is a **local, in-browser demonstration layer** for the weak-labeling,
> embedding-visualization, and model-management workflows. **It does not train
> the real backend classifier.**
>
> The **production training path** is the Python pipeline:
> `python_services/frequency_guard/training/train_pipeline.py` — manifest →
> stratified CV → sklearn ensemble → isotonic calibration → persisted artifacts in
> `checkpoints/`. The `/training` page's Stage-1 forensic analysis **does** call the
> real `POST /api/v1/analyze` (see `components/training/ForensicAnalysisStage.tsx`),
> but the embedding store, cluster assignment, and "model addition" are client-side
> UI simulations.
>
> Any numeric claim in the original version of this doc (e.g. "96.4% accuracy",
> "82.7% AUROC") was **fabricated** and has been removed. Real metrics come from
> `scripts/train_demo.py` / `evaluate.py` and are written to `reports/`.

## Overview

The /training page provides a research-style interface for the **two-stage intake**
workflow (automated forensic analysis + optional weak user labeling), plus
embedding visualization and a model-management surface. It demonstrates how a
forensic-first training loop *could* be wired; the actual detector model is trained
server-side by `train_pipeline.py`.

## Architecture

### Core Components

1. **Two-Stage Intake Pipeline** (`TrainingIntake.tsx`)
   - Stage 1: Automated forensic analysis with CDCS computation
   - Stage 2: Optional weak user labeling with confidence weighting
   - Seamless flow from upload to labeled training signal

2. **Forensic Analysis Service** (`forensicAnalysis.ts`)
   - Multi-resolution frequency features (Contourlet Transform, Multi-Resolution FFT)
   - Statistical fingerprints (Fractal dimension, GLCM features)
   - Model-specific artifact detection (Checkerboard, attention maps, light transport)
   - CDCS (Cross-Domain Consistency Score) calculation

3. **Embedding Store** (`embeddingStore.ts`)
   - FAISS-like functionality for 512/1024-dim vectors
   - Dynamic clustering with DBSCAN
   - Unknown AI detection via open-set recognition
   - Cluster health metrics (cohesion, separation, silhouette score)

4. **Weak Signal Processing** (`weakSignal.ts`)
   - Weighted fusion of detector predictions and user labels
   - Calibration accuracy tracking
   - Label noise detection
   - User accuracy profiling

5. **Model Metadata Store** (`modelMetadata.ts`)
   - Statistical fingerprint distributions
   - Model-specific artifact patterns
   - External provenance signals (EXIF, prompts, reverse image search)
   - Model similarity comparison

6. **Training Integration** (`trainingIntegration.ts`)
   - REST API layer for main detector integration
   - Cryptographic proof generation for reproducibility
   - Training data export/import with verification

## Key Features

### Forensic-First Design
All training leverages comprehensive forensic features:
- **Frequency Domain**: Phase randomness, cross-channel coherence, spectral entropy gradients
- **Spatial Domain**: Fractal dimension, GLCM texture features
- **Cross-Domain**: CDCS score from spatial, frequency, wavelet, and statistical branches
- **Artifacts**: Checkerboard patterns, attention map consistency, light transport analysis

### Weak Signal Learning
Treats user labels as probabilistic signals rather than ground truth:
- Weighted fusion based on detector calibration accuracy and user historical accuracy
- Confidence sliders allow users to express uncertainty
- Agreement bonuses when detector and user align
- Label noise detection for quality control

### Embedding-Based Clustering
Replaces hard-coded model names with dynamic embedding space:
- 512-dimensional feature vectors per image
- Known model clusters with centroids and variance tracking
- Unknown AI detection when new clusters form
- Few-shot learning for custom model addition (5-10 examples)

### Open-World Recognition
Supports unlimited model addition without retraining:
- One-Class SVM on real image embeddings
- DBSCAN clustering for unknown AI families
- Notification system for new cluster review
- Seamless integration of labeled unknown clusters

## Usage

### Accessing the Training Section

Navigate to `/training` or click the "Training" link in the header (with graduation cap icon).

### Training Intake Workflow

1. **Upload Images**: Drag & drop or select training images
2. **Stage 1**: Automatic forensic analysis runs
   - Frequency features extracted
   - CDCS score computed
   - Detector prediction generated
3. **Stage 2**: Optional user labeling
   - Pre-populated with detector's top prediction
   - Select model family and specific model
   - Add confidence slider (0-100%)
   - Submit or skip labeling
4. **Complete**: Training signals generated and stored

### Adding Custom Models

1. Navigate to "Model Management" tab
2. Click "Add Custom Model"
3. **Step 1: Details**
   - Enter model name (e.g., "My Custom SDXL Fine-tune")
   - Select model family (Diffusion, GAN, Transformer, Unknown)
   - Provide description
4. **Step 2: Examples**
   - Upload 5-10 example images (few-shot learning)
   - System auto-generates cluster centroid
5. **Step 3: Review**
   - Confirm details
   - Model is added to embedding store and detector

### Monitoring Training Progress

The **Training Dashboard** provides:
- Technical metrics (accuracy, AUROC, inference time, FAR)
- Per-model performance (precision, recall, F1 score)
- Calibration metrics (ECE, temperature scaling, Brier score)
- LOMO cross-validation results

### Visualizing Embeddings

The **Embedding Visualizer** shows:
- t-SNE/UMAP projection of 512-dim embedding space
- Known model clusters with cohesion and separation metrics
- Unknown AI clusters pending review
- Cluster health indicators

## Technical Specifications

### Performance Targets

Real, measured metrics are produced by `scripts/train_demo.py` on a **synthetic
demo dataset** and reported to `reports/benchmark_<ts>.json`. They are **not**
claims about real generator detection (see README honesty note). Typical demo-run
output on this repo's procedural dataset:

| Metric | Demo value (synthetic-only) |
|--------|------------------------------|
| Held-out accuracy | ≈1.0 (demo set is trivially separable) |
| ROC-AUC | ≈1.0 |
| ECE | <0.01 |
| FPR @ tuned threshold | ≈0.0 |
| Peak RSS | ≈150 MB |

Train on a real labeled manifest to get production-grade numbers.

### Calibration (real backend)

- **Expected Calibration Error (ECE)**: computed by `evaluate.py` on every run
  and written to `reports/`; the demo model measures ECE < 0.01.
- **Isotonic regression** is the production calibrator (`models/calibration.py`);
  Platt/sigmoid is available as a baseline. The UI's "temperature scaling" panel
  is a visualization of the concept, not the backend mechanism.

### Clustering Quality

- **Cohesion**: Average within-cluster distance (higher = tighter)
- **Separation**: Distance to nearest other cluster (higher = more distinct)
- **Silhouette Score**: (separation - cohesion) / max(separation, cohesion)
- **Davies-Bouldin Index**: scatter / separation (lower = better)

## Integration with Main Detector

The training section integrates with the main detector via:

1. **Shared Embedding Vectors**: Training generates embeddings used by detector
2. **Cluster Synchronization**: New models automatically update detector's classifier
3. **Feedback Loop**: Detector predictions become training data
4. **Cryptographic Proofs**: Signed hashes ensure data integrity

### API Endpoints (Placeholder)

```typescript
// Submit training signal
POST /api/training/signal
Body: { imageId, embeddingVector, detectorPrediction, userLabel }

// Sync model clusters
GET /api/training/clusters

// Detect unknown clusters
POST /api/training/detect-unknown

// Get training metrics
GET /api/training/metrics
```

## Advanced Features

### Cryptographic Proofs

Training data includes lightweight cryptographic proofs for reproducibility:
- SHA-256 hash of training data (detector predictions, user labels, embeddings)
- Timestamped signatures
- Version tracking
- Verifiable imports/exports

### LOMO Cross-Validation

Leave-One-Model-Out evaluation tests generalization:
- Train on all models except one
- Evaluate on held-out model
- Measures adaptability to unseen models

### Temperature Scaling

Calibrates detector confidence scores:
- Computed from validation set
- Applied to soften/harden confidences
- Aligns predicted confidence with actual accuracy

## Development Roadmap

### Week 1: Two-Stage Pipeline ✅
- [x] Forensic analysis stage
- [x] Weak user labeling stage
- [x] Signal fusion mechanism

### Week 2: Embedding Infrastructure ✅
- [x] Embedding store with FAISS-like functionality
- [x] Clustering algorithms (DBSCAN, k-means)
- [x] Unknown AI detection

### Week 3: Model Management ✅
- [x] Custom model addition UI
- [x] Few-shot learning integration
- [x] Model metadata storage

### Week 4: Forensic Integration ✅
- [x] Full forensic feature suite
- [x] CDCS-weighted training
- [x] Artifact pattern analysis

### Week 5: Optimization & Validation ✅
- [x] Calibration metrics
- [x] LOMO evaluation
- [x] Training dashboard

### Future Enhancements
- [ ] Production API integration
- [ ] Real-time clustering updates
- [ ] Advanced few-shot learning (MAML, Prototypical Networks)
- [ ] Adversarial training pipeline
- [ ] Multi-user collaboration features

## Research Contributions

This training system demonstrates:

1. **Dynamic Scalability**: Unlimited model addition without full retraining
2. **Weak Signal Learning**: Principled fusion of uncertain labels
3. **Forensic-First Design**: Deep integration of forensic features
4. **Open-World Recognition**: Detection and labeling of unknown AI families
5. **Continuous Improvement**: Feedback loop between detector and training

These features are rare in internship projects and position this as a **research-grade system** suitable for publication.

## References

### Forensic Analysis
- Contourlet Transform: Do & Vetterli (2005)
- GLCM Features: Haralick et al. (1973)
- Fractal Dimension: Mandelbrot (1982)

### Machine Learning
- Few-Shot Learning: Snell et al. (2017) - Prototypical Networks
- MAML: Finn et al. (2017)
- Temperature Scaling: Guo et al. (2017)

### Clustering
- DBSCAN: Ester et al. (1996)
- Silhouette Score: Rousseeuw (1987)

## License

MIT License - See LICENSE file for details

## Contributors

Built for AI Image Forensics Internship Project
Expandable Training System v1.0.0
