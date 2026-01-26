/**
 * Model Metadata Store
 * Manages statistical fingerprints, model-specific artifacts, and external provenance signals
 */

import type { ModelCluster, ForensicFeatures } from '@/lib/types/training';

export interface ModelMetadata {
  modelId: string;
  modelName: string;
  modelFamily: string;
  
  // Statistical fingerprints
  statisticalFingerprint: {
    // Frequency domain
    avgPhaseRandomness: number;
    stdPhaseRandomness: number;
    avgCrossChannelCoherence: number;
    avgSpectralEntropy: number;
    
    // Spatial domain
    avgFractalDimension: number;
    stdFractalDimension: number;
    glcmProfile: {
      contrast: { mean: number; std: number };
      correlation: { mean: number; std: number };
      energy: { mean: number; std: number };
      homogeneity: { mean: number; std: number };
    };
    
    // Cross-domain
    avgCDCS: number;
    stdCDCS: number;
  };
  
  // Model-specific artifacts
  artifactSignature: {
    checkerboardPrevalence: number; // % of images showing checkerboard
    attentionArtifactScore: number;
    lightTransportInconsistency: number;
    commonArtifactTypes: string[];
    artifactLocations: string[]; // e.g., "edges", "high-freq", "smooth-regions"
  };
  
  // External provenance signals
  provenanceProfile: {
    exifPatterns: {
      software: string[];
      cameraModel: string[];
      processingDetails: string[];
    };
    commonPromptPatterns: string[];
    styleKeywords: string[];
    reverseImageSources: string[];
    temporalSignature: {
      averageCreationDate?: Date;
      dateRange?: [Date, Date];
    };
  };
  
  // Cluster health
  clusterQuality: {
    cohesion: number;
    separation: number;
    silhouetteScore: number;
    dbIndex: number; // Davies-Bouldin Index
  };
  
  // Metadata
  sampleCount: number;
  lastUpdated: number;
  version: number;
}

export class ModelMetadataStore {
  private metadata: Map<string, ModelMetadata> = new Map();

  /**
   * Create or update metadata for a model
   */
  async updateMetadata(
    cluster: ModelCluster,
    forensicFeatures: ForensicFeatures[]
  ): Promise<ModelMetadata> {
    const existing = this.metadata.get(cluster.id);
    
    const newMetadata: ModelMetadata = {
      modelId: cluster.id,
      modelName: cluster.modelName,
      modelFamily: cluster.modelFamily,
      
      statisticalFingerprint: this.computeStatisticalFingerprint(forensicFeatures),
      artifactSignature: this.computeArtifactSignature(forensicFeatures),
      provenanceProfile: this.computeProvenanceProfile(cluster),
      clusterQuality: {
        cohesion: cluster.cohesion,
        separation: cluster.separation,
        silhouetteScore: this.calculateSilhouetteScore(cluster),
        dbIndex: this.calculateDaviesBouldinIndex(cluster)
      },
      
      sampleCount: cluster.sampleCount,
      lastUpdated: Date.now(),
      version: existing ? existing.version + 1 : 1
    };

    this.metadata.set(cluster.id, newMetadata);
    return newMetadata;
  }

  /**
   * Compute statistical fingerprint from forensic features
   */
  private computeStatisticalFingerprint(features: ForensicFeatures[]): ModelMetadata['statisticalFingerprint'] {
    const n = features.length;
    
    // Phase randomness stats
    const phaseRandomness = features.map(f => f.phaseRandomness);
    const avgPhaseRandomness = this.mean(phaseRandomness);
    const stdPhaseRandomness = this.std(phaseRandomness);

    // Cross-channel coherence
    const coherence = features.map(f => f.crossChannelCoherence);
    const avgCrossChannelCoherence = this.mean(coherence);

    // Spectral entropy
    const entropy = features.flatMap(f => f.spectralEntropyGradients);
    const avgSpectralEntropy = this.mean(entropy);

    // Fractal dimension
    const fractalDim = features.map(f => f.fractalDimension);
    const avgFractalDimension = this.mean(fractalDim);
    const stdFractalDimension = this.std(fractalDim);

    // GLCM features
    const glcmContrast = features.map(f => f.glcmFeatures.contrast);
    const glcmCorrelation = features.map(f => f.glcmFeatures.correlation);
    const glcmEnergy = features.map(f => f.glcmFeatures.energy);
    const glcmHomogeneity = features.map(f => f.glcmFeatures.homogeneity);

    // CDCS
    const cdcs = features.map(f => f.cdcsScore);
    const avgCDCS = this.mean(cdcs);
    const stdCDCS = this.std(cdcs);

    return {
      avgPhaseRandomness,
      stdPhaseRandomness,
      avgCrossChannelCoherence,
      avgSpectralEntropy,
      avgFractalDimension,
      stdFractalDimension,
      glcmProfile: {
        contrast: { mean: this.mean(glcmContrast), std: this.std(glcmContrast) },
        correlation: { mean: this.mean(glcmCorrelation), std: this.std(glcmCorrelation) },
        energy: { mean: this.mean(glcmEnergy), std: this.std(glcmEnergy) },
        homogeneity: { mean: this.mean(glcmHomogeneity), std: this.std(glcmHomogeneity) }
      },
      avgCDCS,
      stdCDCS
    };
  }

  /**
   * Compute artifact signature
   */
  private computeArtifactSignature(features: ForensicFeatures[]): ModelMetadata['artifactSignature'] {
    const n = features.length;

    // Checkerboard prevalence
    const checkerboardScores = features.map(f => f.checkerboardScore);
    const checkerboardPrevalence = checkerboardScores.filter(s => s > 0.3).length / n;

    // Attention artifacts
    const attentionScores = features.map(f => f.attentionMapConsistency);
    const attentionArtifactScore = this.mean(attentionScores);

    // Light transport
    const lightScores = features.map(f => f.lightTransportConsistency);
    const lightTransportInconsistency = 1 - this.mean(lightScores);

    // Classify common artifact types
    const commonArtifactTypes: string[] = [];
    if (checkerboardPrevalence > 0.3) commonArtifactTypes.push('checkerboard');
    if (attentionArtifactScore > 0.7) commonArtifactTypes.push('attention-maps');
    if (lightTransportInconsistency > 0.4) commonArtifactTypes.push('lighting-inconsistency');

    // Artifact locations (simplified)
    const artifactLocations = this.identifyArtifactLocations(features);

    return {
      checkerboardPrevalence,
      attentionArtifactScore,
      lightTransportInconsistency,
      commonArtifactTypes,
      artifactLocations
    };
  }

  /**
   * Compute provenance profile
   */
  private computeProvenanceProfile(cluster: ModelCluster): ModelMetadata['provenanceProfile'] {
    // Extract from cluster's provenance signals
    return {
      exifPatterns: {
        software: cluster.provenanceSignals.commonEXIFPatterns.filter(p => p.includes('software')),
        cameraModel: cluster.provenanceSignals.commonEXIFPatterns.filter(p => p.includes('camera')),
        processingDetails: cluster.provenanceSignals.commonEXIFPatterns.filter(p => p.includes('processing'))
      },
      commonPromptPatterns: cluster.provenanceSignals.commonPromptKeywords,
      styleKeywords: this.extractStyleKeywords(cluster),
      reverseImageSources: cluster.provenanceSignals.reverseImageSearchSources,
      temporalSignature: {}
    };
  }

  /**
   * Calculate silhouette score for cluster quality
   */
  private calculateSilhouetteScore(cluster: ModelCluster): number {
    // Simplified: use cohesion and separation
    const a = 1 - cluster.cohesion; // intra-cluster distance
    const b = cluster.separation; // inter-cluster distance
    
    if (Math.max(a, b) === 0) return 0;
    return (b - a) / Math.max(a, b);
  }

  /**
   * Calculate Davies-Bouldin Index
   */
  private calculateDaviesBouldinIndex(cluster: ModelCluster): number {
    // Simplified: lower is better
    // Ideal value < 1
    const scatter = cluster.variance;
    const separation = cluster.separation;
    
    if (separation === 0) return Infinity;
    return scatter / separation;
  }

  /**
   * Identify common artifact locations
   */
  private identifyArtifactLocations(features: ForensicFeatures[]): string[] {
    const locations: string[] = [];
    
    // Analyze where artifacts appear most frequently
    const highFreqArtifacts = features.filter(f => 
      f.spectralEntropyGradients.some(g => g > 0.8)
    ).length / features.length;

    if (highFreqArtifacts > 0.5) locations.push('high-frequency');

    // Check for edge artifacts
    const edgeArtifacts = features.filter(f => f.checkerboardScore > 0.3).length / features.length;
    if (edgeArtifacts > 0.4) locations.push('edges');

    // Check for smooth region artifacts
    const smoothArtifacts = features.filter(f => f.attentionMapConsistency > 0.7).length / features.length;
    if (smoothArtifacts > 0.5) locations.push('smooth-regions');

    return locations;
  }

  /**
   * Extract style keywords from cluster
   */
  private extractStyleKeywords(cluster: ModelCluster): string[] {
    // Placeholder: would analyze prompt keywords
    const keywords: string[] = [];
    
    if (cluster.modelFamily === 'Diffusion') {
      keywords.push('diffusion', 'stable', 'latent');
    } else if (cluster.modelFamily === 'GAN') {
      keywords.push('gan', 'stylegan', 'generator');
    } else if (cluster.modelFamily === 'Transformer') {
      keywords.push('transformer', 'dalle', 'attention');
    }

    return keywords;
  }

  /**
   * Compare two models' metadata
   */
  compareModels(modelId1: string, modelId2: string): {
    similarity: number;
    differences: string[];
  } {
    const meta1 = this.metadata.get(modelId1);
    const meta2 = this.metadata.get(modelId2);

    if (!meta1 || !meta2) {
      return { similarity: 0, differences: ['One or both models not found'] };
    }

    const differences: string[] = [];
    let similarityScore = 1.0;

    // Compare CDCS
    const cdcsDiff = Math.abs(meta1.statisticalFingerprint.avgCDCS - meta2.statisticalFingerprint.avgCDCS);
    if (cdcsDiff > 0.2) {
      differences.push(`CDCS differs by ${(cdcsDiff * 100).toFixed(1)}%`);
      similarityScore -= 0.15;
    }

    // Compare checkerboard prevalence
    const checkerboardDiff = Math.abs(
      meta1.artifactSignature.checkerboardPrevalence - 
      meta2.artifactSignature.checkerboardPrevalence
    );
    if (checkerboardDiff > 0.3) {
      differences.push(`Checkerboard artifact differs by ${(checkerboardDiff * 100).toFixed(1)}%`);
      similarityScore -= 0.1;
    }

    // Compare fractal dimension
    const fractalDiff = Math.abs(
      meta1.statisticalFingerprint.avgFractalDimension - 
      meta2.statisticalFingerprint.avgFractalDimension
    );
    if (fractalDiff > 0.2) {
      differences.push(`Fractal dimension differs by ${fractalDiff.toFixed(2)}`);
      similarityScore -= 0.1;
    }

    // Compare artifact types
    const artifacts1 = new Set(meta1.artifactSignature.commonArtifactTypes);
    const artifacts2 = new Set(meta2.artifactSignature.commonArtifactTypes);
    const commonArtifacts = [...artifacts1].filter(a => artifacts2.has(a));
    
    if (commonArtifacts.length === 0 && (artifacts1.size > 0 || artifacts2.size > 0)) {
      differences.push('No common artifact types');
      similarityScore -= 0.15;
    }

    return {
      similarity: Math.max(0, similarityScore),
      differences
    };
  }

  /**
   * Get metadata for a model
   */
  getMetadata(modelId: string): ModelMetadata | undefined {
    return this.metadata.get(modelId);
  }

  /**
   * Get all metadata
   */
  getAllMetadata(): ModelMetadata[] {
    return Array.from(this.metadata.values());
  }

  /**
   * Find similar models based on metadata
   */
  findSimilarModels(modelId: string, threshold: number = 0.7): {
    modelId: string;
    similarity: number;
  }[] {
    const results: { modelId: string; similarity: number }[] = [];

    for (const [otherId, _] of this.metadata) {
      if (otherId === modelId) continue;

      const comparison = this.compareModels(modelId, otherId);
      if (comparison.similarity >= threshold) {
        results.push({ modelId: otherId, similarity: comparison.similarity });
      }
    }

    return results.sort((a, b) => b.similarity - a.similarity);
  }

  /**
   * Export metadata for persistence
   */
  export(): ModelMetadata[] {
    return Array.from(this.metadata.values());
  }

  /**
   * Import metadata
   */
  import(metadataList: ModelMetadata[]): void {
    this.metadata.clear();
    for (const meta of metadataList) {
      this.metadata.set(meta.modelId, meta);
    }
  }

  // ===== Helper Methods =====

  private mean(arr: number[]): number {
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  private std(arr: number[]): number {
    const avg = this.mean(arr);
    const variance = arr.reduce((sum, val) => sum + Math.pow(val - avg, 2), 0) / arr.length;
    return Math.sqrt(variance);
  }
}

export const modelMetadataStore = new ModelMetadataStore();
