/**
 * Type definitions for the expandable training system
 * Supports weak signal learning, embedding-based clustering, and forensic-first design
 */

// Model families and lineages
export type ModelFamily = 'Diffusion' | 'GAN' | 'Transformer' | 'Real' | 'Unknown';

export type KnownModel = 
  | 'Stable Diffusion XL'
  | 'Stable Diffusion v2.1'
  | 'Stable Diffusion v1.5'
  | 'DALL-E 3'
  | 'DALL-E 2'
  | 'Midjourney v6'
  | 'Midjourney v5'
  | 'StyleGAN3'
  | 'StyleGAN2'
  | 'Custom';

// Forensic features extracted during Stage 1
export interface ForensicFeatures {
  // Multi-resolution frequency features
  contourletCoefficients: number[];
  multiResFFT: number[][];
  phaseRandomness: number;
  crossChannelCoherence: number;
  spectralEntropyGradients: number[];
  
  // Statistical fingerprints
  fractalDimension: number;
  glcmFeatures: {
    contrast: number;
    correlation: number;
    energy: number;
    homogeneity: number;
  };
  
  // Model-specific artifacts
  attentionMapConsistency: number;
  checkerboardScore: number;
  lightTransportConsistency: number;
  
  // CDCS (Cross-Domain Consistency Score)
  cdcsScore: number;
  spatialBranchConfidence: number;
  frequencyBranchConfidence: number;
  waveletBranchConfidence: number;
  statisticalBranchConfidence: number;
}

// Detector prediction from Stage 1
export interface DetectorPrediction {
  family: ModelFamily;
  model: KnownModel | string;
  confidence: number;
  probabilities: {
    real: number;
    ganLike: number;
    diffusionLike: number;
    [modelName: string]: number;
  };
  forensicFeatures: ForensicFeatures;
  calibrationAccuracy: number; // Historical accuracy for weighting
  timestamp: number;
}

// User label from Stage 2 (weak signal)
export interface UserLabel {
  family: ModelFamily;
  model: KnownModel | string;
  confidence: number; // 0-100 from slider
  userAccuracy?: number; // Optional: user's past labeling accuracy
  timestamp: number;
  userId?: string;
}

// Combined training signal
export interface TrainingSignal {
  imageId: string;
  detectorPrediction: DetectorPrediction;
  userLabel: UserLabel | null;
  fusedLabel: {
    family: ModelFamily;
    model: string;
    confidence: number;
    detectorWeight: number;
    userWeight: number;
  };
  embeddingVector: number[];
  status: 'pending_label' | 'labeled' | 'trained' | 'validated';
}

// Embedding store entry
export interface EmbeddingEntry {
  id: string;
  vector: number[]; // 512 or 1024-dim
  metadata: {
    modelFamily: ModelFamily;
    modelName: string;
    forensicFeatures: ForensicFeatures;
    detectorConfidence: number;
    userConfidence?: number;
    uploadTimestamp: number;
    clusterId?: string;
  };
}

// Cluster metadata
export interface ModelCluster {
  id: string;
  modelName: string;
  modelFamily: ModelFamily;
  centroid: number[];
  variance: number;
  sampleCount: number;
  
  // Statistical fingerprint distributions
  forensicFingerprint: {
    meanFeatures: Partial<ForensicFeatures>;
    stdFeatures: Partial<ForensicFeatures>;
  };
  
  // Model-specific artifact patterns
  artifactPatterns: {
    commonArtifacts: string[];
    artifactFrequency: Record<string, number>;
  };
  
  // External provenance signals
  provenanceSignals: {
    commonEXIFPatterns: string[];
    commonPromptKeywords: string[];
    reverseImageSearchSources: string[];
  };
  
  // Cluster health metrics
  cohesion: number; // How tight the cluster is
  separation: number; // Distance to nearest other cluster
  lastUpdated: number;
}

// Custom model addition
export interface CustomModel {
  id: string;
  name: string;
  family: ModelFamily;
  description: string;
  exampleImages: string[]; // 5-10 examples for few-shot learning
  clusterId: string;
  createdBy: string;
  createdAt: number;
  sampleCount: number;
  averageConfidence: number;
}

// Training metrics
export interface TrainingMetrics {
  // Technical metrics
  overallAccuracy: number;
  unknownAIDetectionAUROC: number;
  inferenceTimeMs: number;
  falseAttributionRate: number;
  
  // Per-model metrics
  modelMetrics: Record<string, {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
    sampleCount: number;
  }>;
  
  // LOMO cross-validation results
  lomoResults: {
    modelName: string;
    accuracy: number;
    avgConfidence: number;
  }[];
  
  // Calibration metrics
  calibrationError: number;
  confidenceIntervals: Record<string, [number, number]>;
  
  lastUpdated: number;
}

// Unknown AI cluster for review
export interface UnknownCluster {
  id: string;
  centroid: number[];
  sampleCount: number;
  averageCDCS: number;
  representativeImages: string[];
  forensicSignature: Partial<ForensicFeatures>;
  discoveredAt: number;
  status: 'pending_review' | 'under_review' | 'labeled' | 'rejected';
  reviewedBy?: string;
  assignedModelName?: string;
}

// Training session
export interface TrainingSession {
  id: string;
  startTime: number;
  endTime?: number;
  samplesProcessed: number;
  newModelsAdded: number;
  clustersUpdated: string[];
  metrics: TrainingMetrics;
  status: 'active' | 'completed' | 'failed';
}

// Cryptographic proof for reproducibility
export interface CryptographicProof {
  dataHash: string; // SHA-256 of training data
  timestamp: number;
  signature: string; // Lightweight crypto signature
  version: string;
  metadata: {
    detectorVersion: string;
    embeddingDimension: number;
    sampleCount: number;
  };
}
