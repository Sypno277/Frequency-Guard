/**
 * Embedding Store Service
 * Manages embedding vectors with FAISS-like functionality for clustering and similarity search
 */

import type { EmbeddingEntry, ModelCluster, UnknownCluster } from '@/lib/types/training';

export class EmbeddingStoreService {
  private embeddings: Map<string, EmbeddingEntry> = new Map();
  private clusters: Map<string, ModelCluster> = new Map();
  private unknownClusters: UnknownCluster[] = [];
  private dimension: number = 512;

  /**
   * Add an embedding vector to the store
   */
  async addEmbedding(entry: EmbeddingEntry): Promise<void> {
    if (entry.vector.length !== this.dimension) {
      throw new Error(`Embedding dimension mismatch: expected ${this.dimension}, got ${entry.vector.length}`);
    }

    this.embeddings.set(entry.id, entry);

    // Attempt to assign to existing cluster or create new
    await this.assignToCluster(entry);
  }

  /**
   * Batch add embeddings
   */
  async addEmbeddingsBatch(entries: EmbeddingEntry[]): Promise<void> {
    for (const entry of entries) {
      await this.addEmbedding(entry);
    }
  }

  /**
   * Find k-nearest neighbors for a query vector
   */
  async findSimilar(queryVector: number[], k: number = 5): Promise<{
    id: string;
    distance: number;
    entry: EmbeddingEntry;
  }[]> {
    const similarities: { id: string; distance: number; entry: EmbeddingEntry }[] = [];

    for (const [id, entry] of this.embeddings) {
      const distance = this.cosineSimilarity(queryVector, entry.vector);
      similarities.push({ id, distance, entry });
    }

    // Sort by distance (higher cosine similarity = lower distance)
    similarities.sort((a, b) => b.distance - a.distance);

    return similarities.slice(0, k);
  }

  /**
   * Assign embedding to a cluster or create new cluster
   */
  private async assignToCluster(entry: EmbeddingEntry): Promise<void> {
    const modelName = entry.metadata.modelName;
    const modelFamily = entry.metadata.modelFamily;

    // Check if cluster exists for this model
    let clusterId = `${modelFamily}_${modelName}`;
    let cluster = this.clusters.get(clusterId);

    if (!cluster) {
      // Create new cluster
      cluster = {
        id: clusterId,
        modelName,
        modelFamily,
        centroid: [...entry.vector],
        variance: 0,
        sampleCount: 1,
        forensicFingerprint: {
          meanFeatures: entry.metadata.forensicFeatures,
          stdFeatures: {}
        },
        artifactPatterns: {
          commonArtifacts: [],
          artifactFrequency: {}
        },
        provenanceSignals: {
          commonEXIFPatterns: [],
          commonPromptKeywords: [],
          reverseImageSearchSources: []
        },
        cohesion: 1.0,
        separation: 0,
        lastUpdated: Date.now()
      };
      this.clusters.set(clusterId, cluster);
    } else {
      // Update existing cluster
      await this.updateCluster(cluster, entry);
    }

    // Update embedding metadata with cluster ID
    entry.metadata.clusterId = clusterId;
  }

  /**
   * Update cluster centroid and statistics with new embedding
   */
  private async updateCluster(cluster: ModelCluster, newEntry: EmbeddingEntry): Promise<void> {
    const n = cluster.sampleCount;
    
    // Update centroid using incremental mean
    for (let i = 0; i < this.dimension; i++) {
      cluster.centroid[i] = (cluster.centroid[i] * n + newEntry.vector[i]) / (n + 1);
    }

    cluster.sampleCount++;
    
    // Update variance
    cluster.variance = this.computeClusterVariance(cluster);
    
    // Update forensic fingerprint (running statistics)
    this.updateForensicFingerprint(cluster, newEntry);
    
    // Update cluster health metrics
    cluster.cohesion = this.computeCohesion(cluster);
    cluster.separation = await this.computeSeparation(cluster);
    cluster.lastUpdated = Date.now();
  }

  /**
   * Compute cluster variance
   */
  private computeClusterVariance(cluster: ModelCluster): number {
    const embeddings = Array.from(this.embeddings.values())
      .filter(e => e.metadata.clusterId === cluster.id);

    if (embeddings.length === 0) return 0;

    let totalVariance = 0;
    for (const embedding of embeddings) {
      const distance = this.euclideanDistance(embedding.vector, cluster.centroid);
      totalVariance += distance * distance;
    }

    return totalVariance / embeddings.length;
  }

  /**
   * Update forensic fingerprint with running statistics
   */
  private updateForensicFingerprint(cluster: ModelCluster, newEntry: EmbeddingEntry): void {
    // Update mean features
    const features = newEntry.metadata.forensicFeatures;
    const n = cluster.sampleCount;
    const mean = cluster.forensicFingerprint.meanFeatures;

    // Update numeric features
    if (features.cdcsScore !== undefined) {
      mean.cdcsScore = ((mean.cdcsScore || 0) * (n - 1) + features.cdcsScore) / n;
    }
    if (features.phaseRandomness !== undefined) {
      mean.phaseRandomness = ((mean.phaseRandomness || 0) * (n - 1) + features.phaseRandomness) / n;
    }
    if (features.fractalDimension !== undefined) {
      mean.fractalDimension = ((mean.fractalDimension || 0) * (n - 1) + features.fractalDimension) / n;
    }
    if (features.checkerboardScore !== undefined) {
      mean.checkerboardScore = ((mean.checkerboardScore || 0) * (n - 1) + features.checkerboardScore) / n;
    }
  }

  /**
   * Compute cluster cohesion (tightness)
   */
  private computeCohesion(cluster: ModelCluster): number {
    const embeddings = Array.from(this.embeddings.values())
      .filter(e => e.metadata.clusterId === cluster.id);

    if (embeddings.length < 2) return 1.0;

    let avgDistance = 0;
    for (const embedding of embeddings) {
      avgDistance += this.euclideanDistance(embedding.vector, cluster.centroid);
    }
    avgDistance /= embeddings.length;

    // Normalize: lower distance = higher cohesion
    return 1 / (1 + avgDistance);
  }

  /**
   * Compute cluster separation from nearest other cluster
   */
  private async computeSeparation(cluster: ModelCluster): Promise<number> {
    let minDistance = Infinity;

    for (const [otherId, otherCluster] of this.clusters) {
      if (otherId !== cluster.id) {
        const distance = this.euclideanDistance(cluster.centroid, otherCluster.centroid);
        minDistance = Math.min(minDistance, distance);
      }
    }

    return minDistance === Infinity ? 0 : minDistance;
  }

  /**
   * Detect unknown AI clusters using DBSCAN-like approach
   */
  async detectUnknownClusters(minSamples: number = 5, eps: number = 0.3): Promise<UnknownCluster[]> {
    const unassigned = Array.from(this.embeddings.values())
      .filter(e => !e.metadata.clusterId || e.metadata.clusterId === 'unknown');

    if (unassigned.length < minSamples) return [];

    // Simple clustering: group embeddings by proximity
    const clusters: UnknownCluster[] = [];
    const visited = new Set<string>();

    for (const seed of unassigned) {
      if (visited.has(seed.id)) continue;

      // Find neighbors within eps radius
      const neighbors = unassigned.filter(e => 
        !visited.has(e.id) && 
        this.euclideanDistance(seed.vector, e.vector) < eps
      );

      if (neighbors.length >= minSamples) {
        visited.add(seed.id);
        neighbors.forEach(n => visited.add(n.id));

        // Compute cluster centroid
        const centroid = this.computeCentroid(neighbors.map(n => n.vector));

        // Compute average CDCS
        const avgCDCS = neighbors.reduce((sum, n) => 
          sum + (n.metadata.forensicFeatures.cdcsScore || 0), 0
        ) / neighbors.length;

        // Create unknown cluster
        const unknownCluster: UnknownCluster = {
          id: `unknown_${Date.now()}_${Math.random().toString(36).slice(2)}`,
          centroid,
          sampleCount: neighbors.length,
          averageCDCS: avgCDCS,
          representativeImages: neighbors.slice(0, 5).map(n => n.id),
          forensicSignature: this.extractForensicSignature(neighbors),
          discoveredAt: Date.now(),
          status: 'pending_review'
        };

        clusters.push(unknownCluster);
      }
    }

    this.unknownClusters = clusters;
    return clusters;
  }

  /**
   * Label an unknown cluster as a new model
   */
  async labelUnknownCluster(
    clusterId: string, 
    modelName: string, 
    modelFamily: string
  ): Promise<ModelCluster> {
    const unknownCluster = this.unknownClusters.find(c => c.id === clusterId);
    if (!unknownCluster) {
      throw new Error(`Unknown cluster ${clusterId} not found`);
    }

    // Create new model cluster
    const newClusterId = `${modelFamily}_${modelName}`;
    const modelCluster: ModelCluster = {
      id: newClusterId,
      modelName,
      modelFamily: modelFamily as any,
      centroid: unknownCluster.centroid,
      variance: 0,
      sampleCount: unknownCluster.sampleCount,
      forensicFingerprint: {
        meanFeatures: unknownCluster.forensicSignature,
        stdFeatures: {}
      },
      artifactPatterns: {
        commonArtifacts: [],
        artifactFrequency: {}
      },
      provenanceSignals: {
        commonEXIFPatterns: [],
        commonPromptKeywords: [],
        reverseImageSearchSources: []
      },
      cohesion: 1.0,
      separation: 0,
      lastUpdated: Date.now()
    };

    this.clusters.set(newClusterId, modelCluster);

    // Update embeddings with new cluster ID
    for (const imageId of unknownCluster.representativeImages) {
      const embedding = this.embeddings.get(imageId);
      if (embedding) {
        embedding.metadata.clusterId = newClusterId;
        embedding.metadata.modelName = modelName;
        embedding.metadata.modelFamily = modelFamily as any;
      }
    }

    // Mark unknown cluster as labeled
    unknownCluster.status = 'labeled';
    unknownCluster.assignedModelName = modelName;

    return modelCluster;
  }

  /**
   * Get all clusters
   */
  getClusters(): ModelCluster[] {
    return Array.from(this.clusters.values());
  }

  /**
   * Get unknown clusters
   */
  getUnknownClusters(): UnknownCluster[] {
    return this.unknownClusters;
  }

  /**
   * Get cluster by ID
   */
  getCluster(clusterId: string): ModelCluster | undefined {
    return this.clusters.get(clusterId);
  }

  /**
   * Get embeddings for a cluster
   */
  getClusterEmbeddings(clusterId: string): EmbeddingEntry[] {
    return Array.from(this.embeddings.values())
      .filter(e => e.metadata.clusterId === clusterId);
  }

  /**
   * Export store for persistence
   */
  export(): {
    embeddings: EmbeddingEntry[];
    clusters: ModelCluster[];
    unknownClusters: UnknownCluster[];
  } {
    return {
      embeddings: Array.from(this.embeddings.values()),
      clusters: Array.from(this.clusters.values()),
      unknownClusters: this.unknownClusters
    };
  }

  /**
   * Import store from persistence
   */
  import(data: {
    embeddings: EmbeddingEntry[];
    clusters: ModelCluster[];
    unknownClusters: UnknownCluster[];
  }): void {
    this.embeddings.clear();
    this.clusters.clear();

    for (const embedding of data.embeddings) {
      this.embeddings.set(embedding.id, embedding);
    }

    for (const cluster of data.clusters) {
      this.clusters.set(cluster.id, cluster);
    }

    this.unknownClusters = data.unknownClusters;
  }

  // ===== Helper Methods =====

  private cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  private euclideanDistance(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += Math.pow(a[i] - b[i], 2);
    }
    return Math.sqrt(sum);
  }

  private computeCentroid(vectors: number[][]): number[] {
    if (vectors.length === 0) return [];

    const centroid = new Array(vectors[0].length).fill(0);

    for (const vector of vectors) {
      for (let i = 0; i < vector.length; i++) {
        centroid[i] += vector[i];
      }
    }

    for (let i = 0; i < centroid.length; i++) {
      centroid[i] /= vectors.length;
    }

    return centroid;
  }

  private extractForensicSignature(embeddings: EmbeddingEntry[]): any {
    const features = embeddings.map(e => e.metadata.forensicFeatures);
    
    return {
      cdcsScore: features.reduce((sum, f) => sum + (f.cdcsScore || 0), 0) / features.length,
      phaseRandomness: features.reduce((sum, f) => sum + (f.phaseRandomness || 0), 0) / features.length,
      fractalDimension: features.reduce((sum, f) => sum + (f.fractalDimension || 0), 0) / features.length,
      checkerboardScore: features.reduce((sum, f) => sum + (f.checkerboardScore || 0), 0) / features.length
    };
  }
}

export const embeddingStoreService = new EmbeddingStoreService();
