# Architecture Diagram - Expandable Training Section

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE (React/TypeScript)                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐│
│  │  Training    │  │   Training   │  │    Model     │  │ Embeddings  ││
│  │   Intake     │  │  Dashboard   │  │ Management   │  │ Visualizer  ││
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘│
│         │                 │                 │                 │         │
└─────────┼─────────────────┼─────────────────┼─────────────────┼─────────┘
          │                 │                 │                 │
          ▼                 ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         SERVICE LAYER (TypeScript)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐  │
│  │  Forensic        │    │  Weak Signal     │    │  Embedding      │  │
│  │  Analysis        │───▶│  Processing      │───▶│  Store          │  │
│  │  Service         │    │  Service         │    │  Service        │  │
│  └──────────────────┘    └──────────────────┘    └─────────┬───────┘  │
│          │                       │                          │           │
│          │                       │                          │           │
│          ▼                       ▼                          ▼           │
│  ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐  │
│  │  Model Metadata  │    │  Training        │    │  Clustering     │  │
│  │  Store           │    │  Integration     │    │  Algorithms     │  │
│  └──────────────────┘    └──────────────────┘    └─────────────────┘  │
│                                                                           │
└───────────────────────────────────────────────────────────────────────┘
```

## Two-Stage Intake Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     IMAGE UPLOAD (User Action)                       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: FORENSIC ANALYSIS                        │
├─────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ Multi-Resolution Frequency Features                           │ │
│  │  • Contourlet Transform (128 coefficients)                    │ │
│  │  • Multi-Resolution FFT (5 scales × 64 features)              │ │
│  │  • Phase Randomness (entropy metric)                          │ │
│  │  • Cross-Channel Coherence (RGB correlation)                  │ │
│  │  • Spectral Entropy Gradients (32-dim vector)                 │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                             │                                         │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ Statistical Fingerprints                                      │ │
│  │  • Fractal Dimension (box-counting method)                    │ │
│  │  • GLCM Features (contrast, correlation, energy, homogeneity) │ │
│  │  • Texture Analysis                                           │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                             │                                         │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ Model-Specific Artifacts                                      │ │
│  │  • Checkerboard Pattern Detection (GAN upsampling)            │ │
│  │  • Attention Map Consistency (diffusion models)               │ │
│  │  • Light Transport Analysis (physical plausibility)           │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                             │                                         │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ CDCS Computation                                              │ │
│  │  Spatial Branch      ──┐                                      │ │
│  │  Frequency Branch    ──┼──▶ Consistency Score (0-1)          │ │
│  │  Wavelet Branch      ──┤                                      │ │
│  │  Statistical Branch  ──┘                                      │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                             │                                         │
│                             ▼                                         │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ DETECTOR PREDICTION                                           │ │
│  │  • Family: Diffusion/GAN/Transformer/Real                     │ │
│  │  • Model: Stable Diffusion XL / DALL-E 3 / etc.              │ │
│  │  • Confidence: 0.87 (calibrated)                              │ │
│  │  • Forensic Features: [all extracted features]                │ │
│  └───────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: WEAK USER LABELING                      │
├─────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ Display Detector Prediction                                   │ │
│  │  Pre-populate: "Likely Stable Diffusion XL (87%)"            │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                             │                                         │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ User Input (Optional)                                         │ │
│  │  • Select Model Family: [Diffusion ▼]                        │ │
│  │  • Select Specific Model: [Stable Diffusion XL v1.0 ▼]      │ │
│  │  • Confidence Slider: [████████░░] 85%                       │ │
│  │  • Historical Accuracy: 82%                                   │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                             │                                         │
│                    ┌────────┴────────┐                               │
│                    │                 │                               │
│                    ▼                 ▼                               │
│            ┌──────────────┐  ┌──────────────┐                       │
│            │   Submit     │  │     Skip     │                       │
│            │   Label      │  │   Labeling   │                       │
│            └──────┬───────┘  └──────┬───────┘                       │
│                   │                 │                               │
└───────────────────┼─────────────────┼───────────────────────────────┘
                    │                 │
                    └────────┬────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    WEAK SIGNAL FUSION                                │
├─────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ Weight Calculation                                            │ │
│  │  Detector Weight = Calibration Accuracy × 0.7                │ │
│  │                  = 0.87 × 0.7 = 0.609                        │ │
│  │                                                               │ │
│  │  User Weight = (Confidence / 100) × User Accuracy × 0.3      │ │
│  │              = (85 / 100) × 0.82 × 0.3 = 0.209              │ │
│  │                                                               │ │
│  │  Normalized:                                                  │ │
│  │    Detector: 0.609 / 0.818 = 74.5%                          │ │
│  │    User:     0.209 / 0.818 = 25.5%                          │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                             │                                         │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ Label Selection                                               │ │
│  │  Detector Score = 0.87 × 0.745 = 0.648                      │ │
│  │  User Score     = 0.85 × 0.255 = 0.217                      │ │
│  │  Total          = 0.648 + 0.217 = 0.865                     │ │
│  │                                                               │ │
│  │  Agreement Bonus (if models match): +10%                      │ │
│  │  Final Confidence = min(0.865 × 1.1, 1.0) = 0.95            │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                             │                                         │
│                             ▼                                         │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ TRAINING SIGNAL                                               │ │
│  │  • Image ID: img_1234567890_0                                │ │
│  │  • Fused Label: Stable Diffusion XL (95%)                    │ │
│  │  • Detector Weight: 74.5%                                     │ │
│  │  • User Weight: 25.5%                                         │ │
│  │  • Embedding: [512-dim vector]                                │ │
│  │  • Status: labeled                                            │ │
│  └───────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
                    [Embedding Store]
```

## Embedding-Based Clustering

```
┌─────────────────────────────────────────────────────────────────────┐
│                     512-DIMENSIONAL EMBEDDING SPACE                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│    Stable Diffusion XL        DALL-E 3           Midjourney v6      │
│    ╔════════════════╗      ╔═══════════════╗   ╔══════════════╗    │
│    ║   ●  ● ● ●     ║      ║   ●  ●  ●     ║   ║  ●  ●  ●    ║    │
│    ║  ●  ⊕   ●  ●   ║      ║  ●  ⊕  ●  ●   ║   ║ ●  ⊕   ●    ║    │
│    ║   ● ●  ●       ║      ║    ●  ●       ║   ║  ●   ●  ●   ║    │
│    ║  ●   ●  ●  ●   ║      ║  ●   ●  ●     ║   ║ ●  ●    ●   ║    │
│    ╚════════════════╝      ╚═══════════════╝   ╚══════════════╝    │
│    342 samples              218 samples          187 samples        │
│    Cohesion: 0.87           Cohesion: 0.82       Cohesion: 0.84    │
│    Separation: 2.34         Separation: 2.89     Separation: 2.56  │
│                                                                       │
│    StyleGAN3                Unknown Cluster #1   Unknown Cluster #2 │
│    ╔═══════════════╗        ┌─────────────┐     ┌──────────────┐   │
│    ║  ●  ●  ●      ║        │  ●  ●  ●    │     │  ●  ●       │   │
│    ║ ●  ⊕   ●  ●   ║        │ ●  ?   ●    │     │ ●  ?   ●    │   │
│    ║  ●   ●   ●    ║        │   ●  ●      │     │  ●  ●  ●    │   │
│    ╚═══════════════╝        └─────────────┘     └──────────────┘   │
│    156 samples              23 samples           17 samples         │
│    Cohesion: 0.91           CDCS: 0.76           CDCS: 0.81         │
│                             Status: Pending       Status: Pending   │
│                                                                       │
│    Legend:                                                            │
│    ● = Training sample      ⊕ = Cluster centroid   ? = Unknown      │
│    ╔═══╗ = Known model      ┌───┐ = Unknown cluster                 │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘

Clustering Algorithm: DBSCAN
Distance Metric: Cosine Similarity
Minimum Samples: 5
Epsilon (ε): 0.3
```

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRAINING DATA FLOW                           │
└─────────────────────────────────────────────────────────────────────┘

User Upload ──┐
              │
              ├──▶ Forensic Analysis ──┐
              │                        │
              │                        ├──▶ Training Signal ──┐
              │                        │                       │
              └──▶ User Labeling  ─────┘                       │
                                                               │
                                                               ▼
                                            ┌─────────────────────────┐
                                            │   Embedding Store       │
                                            ├─────────────────────────┤
                                            │ • Vector: [512-dim]     │
                                            │ • Metadata: forensics   │
                                            │ • Cluster ID            │
                                            └──────────┬──────────────┘
                                                       │
                         ┌─────────────────────────────┼─────────────────┐
                         │                             │                 │
                         ▼                             ▼                 ▼
              ┌──────────────────┐        ┌──────────────────┐  ┌─────────────┐
              │ Cluster Update   │        │ Metadata Update  │  │ Unknown     │
              ├──────────────────┤        ├──────────────────┤  │ Detection   │
              │ • Centroid       │        │ • Fingerprints   │  │ DBSCAN      │
              │ • Variance       │        │ • Artifacts      │  └─────────────┘
              │ • Cohesion       │        │ • Provenance     │
              └──────────────────┘        └──────────────────┘
                         │                             │
                         └─────────────┬───────────────┘
                                       │
                                       ▼
                         ┌─────────────────────────┐
                         │  Training Metrics       │
                         ├─────────────────────────┤
                         │ • Accuracy              │
                         │ • AUROC                 │
                         │ • Calibration Error     │
                         │ • Per-Model Stats       │
                         └─────────────────────────┘
                                       │
                                       ▼
                         ┌─────────────────────────┐
                         │  Dashboard Display      │
                         └─────────────────────────┘
```

## Service Integration

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SERVICE CALL HIERARCHY                          │
└─────────────────────────────────────────────────────────────────────┘

TrainingIntake Component
    │
    ├─▶ forensicAnalysisService.analyzeImage()
    │       │
    │       ├─▶ computeContourletTransform()
    │       ├─▶ computeMultiResolutionFFT()
    │       ├─▶ computePhaseRandomness()
    │       ├─▶ computeFractalDimension()
    │       ├─▶ computeGLCMFeatures()
    │       ├─▶ detectCheckerboardPattern()
    │       └─▶ computeCDCS()
    │
    ├─▶ weakSignalService.fuseSignals()
    │       │
    │       ├─▶ calculateDetectorWeight()
    │       ├─▶ calculateUserWeight()
    │       ├─▶ getUserAccuracy()
    │       └─▶ calculateLabelQuality()
    │
    └─▶ embeddingStoreService.addEmbedding()
            │
            ├─▶ assignToCluster()
            │       │
            │       ├─▶ findSimilar() [k-NN search]
            │       └─▶ updateCluster()
            │               │
            │               ├─▶ computeClusterVariance()
            │               ├─▶ updateForensicFingerprint()
            │               ├─▶ computeCohesion()
            │               └─▶ computeSeparation()
            │
            └─▶ detectUnknownClusters() [DBSCAN]
                    │
                    └─▶ modelMetadataStore.updateMetadata()
                            │
                            ├─▶ computeStatisticalFingerprint()
                            ├─▶ computeArtifactSignature()
                            └─▶ computeProvenanceProfile()
```

## Key Algorithms

### CDCS Calculation
```
CDCS = 1 - σ / μ

Where:
σ = std([spatialConf, frequencyConf, waveletConf, statisticalConf])
μ = mean([spatialConf, frequencyConf, waveletConf, statisticalConf])

High CDCS (>0.7) → All branches agree → Likely AI
Low CDCS (<0.3) → Branches disagree → Likely Real
```

### Weak Signal Fusion
```
W_detector = calibrationAccuracy × 0.7
W_user = (confidence / 100) × userAccuracy × 0.3

Score_detector = detectorConfidence × W_detector
Score_user = userConfidence × W_user

If Score_detector > Score_user:
    FinalLabel = DetectorLabel
    FinalConfidence = Score_detector + Score_user
Else:
    FinalLabel = UserLabel
    FinalConfidence = Score_detector + Score_user

If DetectorLabel == UserLabel:
    FinalConfidence *= 1.1  // Agreement bonus
```

### Cluster Assignment
```
For new embedding E:
1. Find k=5 nearest neighbors in embedding space
2. If nearest_distance < threshold:
       Assign to existing cluster
       Update centroid: C_new = (C_old × n + E) / (n + 1)
   Else:
       Create new cluster with C = E
3. Compute cohesion = 1 / (1 + avg_distance_to_centroid)
4. Compute separation = min(distance_to_other_centroids)
```

## Technology Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TECHNOLOGY STACK                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Frontend Framework:     React 18 + TypeScript                       │
│  UI Components:          shadcn/ui + Radix UI                        │
│  Styling:                Tailwind CSS                                │
│  Routing:                React Router v6                             │
│  File Upload:            react-dropzone                              │
│  Charts:                 Recharts                                    │
│  Icons:                  lucide-react                                │
│                                                                       │
│  Build Tool:             Vite                                        │
│  Package Manager:        npm/bun                                     │
│  Type Checking:          TypeScript 5.8                              │
│  Linting:                ESLint                                      │
│                                                                       │
│  Services (Client-Side):                                             │
│    • Forensic Analysis Service                                       │
│    • Embedding Store Service                                         │
│    • Weak Signal Service                                             │
│    • Model Metadata Store                                            │
│    • Training Integration Service                                    │
│                                                                       │
│  Future Backend (Production):                                        │
│    • API: FastAPI/Express                                            │
│    • Vector DB: FAISS/Pinecone/Weaviate                             │
│    • Database: PostgreSQL + S3                                       │
│    • ML Framework: PyTorch/TensorFlow                                │
│    • Deployment: Docker + Kubernetes                                 │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

**This architecture enables:**
- ✅ Dynamic model addition without retraining
- ✅ Weak signal learning with calibration
- ✅ Open-world AI detection
- ✅ Scalable embedding-based clustering
- ✅ Research-grade forensic analysis
