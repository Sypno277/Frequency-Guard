/**
 * Weak Signal Processing Service
 * Handles fusion of detector predictions and user labels with calibration tracking
 */

import type { DetectorPrediction, UserLabel, TrainingSignal } from '@/lib/types/training';

interface UserHistory {
  userId: string;
  totalLabels: number;
  correctLabels: number;
  accuracy: number;
  lastUpdated: number;
}

export class WeakSignalService {
  private userHistories: Map<string, UserHistory> = new Map();
  private detectorCalibrationHistory: { confidence: number; correct: boolean }[] = [];

  /**
   * Fuse detector prediction and user label into training signal
   */
  fuseSignals(
    detector: DetectorPrediction,
    user: UserLabel | null,
    imageId: string,
    embeddingVector: number[]
  ): TrainingSignal {
    if (!user) {
      // Detector-only signal
      return {
        imageId,
        detectorPrediction: detector,
        userLabel: null,
        fusedLabel: {
          family: detector.family,
          model: detector.model,
          confidence: detector.confidence,
          detectorWeight: 1.0,
          userWeight: 0.0
        },
        embeddingVector,
        status: 'pending_label'
      };
    }

    // Get user accuracy
    const userAccuracy = user.userAccuracy || this.getUserAccuracy(user.userId || 'anonymous');

    // Calculate weights
    const detectorWeight = this.calculateDetectorWeight(detector);
    const userWeight = this.calculateUserWeight(user, userAccuracy);

    // Normalize weights
    const totalWeight = detectorWeight + userWeight;
    const normDetectorWeight = detectorWeight / totalWeight;
    const normUserWeight = userWeight / totalWeight;

    // Calculate weighted confidence scores
    const detectorScore = detector.confidence * normDetectorWeight;
    const userScore = (user.confidence / 100) * normUserWeight;

    // Choose label based on higher weighted score
    let fusedFamily = detector.family;
    let fusedModel = detector.model;
    let fusedConfidence = detectorScore + userScore;

    if (userScore > detectorScore) {
      fusedFamily = user.family;
      fusedModel = user.model;
    }

    // Apply agreement bonus: if detector and user agree, boost confidence
    if (detector.model === user.model) {
      fusedConfidence = Math.min(fusedConfidence * 1.1, 1.0);
    }

    return {
      imageId,
      detectorPrediction: detector,
      userLabel: user,
      fusedLabel: {
        family: fusedFamily,
        model: fusedModel,
        confidence: fusedConfidence,
        detectorWeight: normDetectorWeight,
        userWeight: normUserWeight
      },
      embeddingVector,
      status: 'labeled'
    };
  }

  /**
   * Calculate detector weight based on historical calibration
   */
  private calculateDetectorWeight(detector: DetectorPrediction): number {
    // Base weight from calibration accuracy
    const calibrationWeight = detector.calibrationAccuracy;

    // Confidence-dependent weighting: high confidence should be trusted more
    const confidenceBoost = detector.confidence > 0.9 ? 1.2 : 1.0;

    // CDCS-dependent weighting: high CDCS indicates clear AI signal
    const cdcsBoost = detector.forensicFeatures.cdcsScore > 0.7 ? 1.15 : 1.0;

    return calibrationWeight * confidenceBoost * cdcsBoost * 0.7;
  }

  /**
   * Calculate user weight based on confidence slider and historical accuracy
   */
  private calculateUserWeight(user: UserLabel, userAccuracy: number): number {
    // Confidence slider value (0-100)
    const confidenceWeight = user.confidence / 100;

    // Historical accuracy weight
    const accuracyWeight = userAccuracy;

    // Combined weight with dampening factor
    return confidenceWeight * accuracyWeight * 0.3;
  }

  /**
   * Get user's historical labeling accuracy
   */
  getUserAccuracy(userId: string): number {
    const history = this.userHistories.get(userId);
    return history ? history.accuracy : 0.5; // Default to 50% for new users
  }

  /**
   * Update user accuracy after validation
   */
  updateUserAccuracy(userId: string, wasCorrect: boolean): void {
    let history = this.userHistories.get(userId);

    if (!history) {
      history = {
        userId,
        totalLabels: 0,
        correctLabels: 0,
        accuracy: 0.5,
        lastUpdated: Date.now()
      };
      this.userHistories.set(userId, history);
    }

    history.totalLabels++;
    if (wasCorrect) {
      history.correctLabels++;
    }

    // Update accuracy with exponential moving average for recent performance
    const alpha = 0.1; // Weight for new observation
    const newAccuracy = wasCorrect ? 1.0 : 0.0;
    history.accuracy = alpha * newAccuracy + (1 - alpha) * history.accuracy;
    history.lastUpdated = Date.now();
  }

  /**
   * Update detector calibration history
   */
  updateDetectorCalibration(confidence: number, wasCorrect: boolean): void {
    this.detectorCalibrationHistory.push({ confidence, correct: wasCorrect });

    // Keep only recent history (last 1000 predictions)
    if (this.detectorCalibrationHistory.length > 1000) {
      this.detectorCalibrationHistory.shift();
    }
  }

  /**
   * Calculate current detector calibration error (ECE)
   */
  calculateExpectedCalibrationError(numBins: number = 10): number {
    if (this.detectorCalibrationHistory.length === 0) return 0;

    const bins = Array(numBins).fill(0).map(() => ({ 
      totalConfidence: 0, 
      totalCorrect: 0, 
      count: 0 
    }));

    // Assign predictions to bins
    for (const pred of this.detectorCalibrationHistory) {
      const binIdx = Math.min(Math.floor(pred.confidence * numBins), numBins - 1);
      bins[binIdx].totalConfidence += pred.confidence;
      bins[binIdx].totalCorrect += pred.correct ? 1 : 0;
      bins[binIdx].count++;
    }

    // Calculate ECE
    let ece = 0;
    const totalPredictions = this.detectorCalibrationHistory.length;

    for (const bin of bins) {
      if (bin.count === 0) continue;

      const avgConfidence = bin.totalConfidence / bin.count;
      const avgAccuracy = bin.totalCorrect / bin.count;
      const weight = bin.count / totalPredictions;

      ece += weight * Math.abs(avgConfidence - avgAccuracy);
    }

    return ece;
  }

  /**
   * Get temperature scaling factor for calibration
   */
  calculateTemperatureScaling(): number {
    // Simplified temperature scaling
    // In production: use validation set to optimize temperature
    const ece = this.calculateExpectedCalibrationError();
    
    // If ECE is high, temperature > 1 (soften confidences)
    // If ECE is low, temperature ≈ 1
    return 1 + (ece * 2);
  }

  /**
   * Apply temperature scaling to detector confidence
   */
  applyTemperatureScaling(confidence: number, temperature: number): number {
    // Logit transformation
    const logit = Math.log(confidence / (1 - confidence));
    
    // Scale by temperature
    const scaledLogit = logit / temperature;
    
    // Inverse logit
    return 1 / (1 + Math.exp(-scaledLogit));
  }

  /**
   * Detect label noise (conflicting signals)
   */
  detectLabelNoise(trainingSignal: TrainingSignal): {
    isNoisy: boolean;
    reason: string;
    confidence: number;
  } {
    const { detectorPrediction, userLabel } = trainingSignal;

    if (!userLabel) {
      return { isNoisy: false, reason: 'No user label', confidence: 0 };
    }

    // Check for strong disagreement
    const modelMismatch = detectorPrediction.model !== userLabel.model;
    const highDetectorConfidence = detectorPrediction.confidence > 0.85;
    const highUserConfidence = userLabel.confidence > 85;

    if (modelMismatch && highDetectorConfidence && highUserConfidence) {
      return {
        isNoisy: true,
        reason: 'Strong disagreement between high-confidence detector and user',
        confidence: 0.8
      };
    }

    // Check for CDCS inconsistency
    const highCDCS = detectorPrediction.forensicFeatures.cdcsScore > 0.8;
    const userLabeledReal = userLabel.family === 'Real';

    if (highCDCS && userLabeledReal) {
      return {
        isNoisy: true,
        reason: 'High CDCS (AI-like) conflicts with Real label',
        confidence: 0.7
      };
    }

    // Check for low CDCS with AI label
    const lowCDCS = detectorPrediction.forensicFeatures.cdcsScore < 0.3;
    const userLabeledAI = ['Diffusion', 'GAN', 'Transformer'].includes(userLabel.family);

    if (lowCDCS && userLabeledAI) {
      return {
        isNoisy: true,
        reason: 'Low CDCS (Real-like) conflicts with AI label',
        confidence: 0.6
      };
    }

    return { isNoisy: false, reason: 'Signals are consistent', confidence: 0 };
  }

  /**
   * Calculate label quality score
   */
  calculateLabelQuality(trainingSignal: TrainingSignal): number {
    const { detectorPrediction, userLabel, fusedLabel } = trainingSignal;

    let quality = 0.5; // Base quality

    // Boost quality if detector has high confidence
    if (detectorPrediction.confidence > 0.9) {
      quality += 0.2;
    }

    // Boost quality if user provided label with high confidence
    if (userLabel && userLabel.confidence > 85) {
      quality += 0.15;
    }

    // Boost quality if detector and user agree
    if (userLabel && detectorPrediction.model === userLabel.model) {
      quality += 0.2;
    }

    // Boost quality if CDCS is clear (very high or very low)
    const cdcs = detectorPrediction.forensicFeatures.cdcsScore;
    if (cdcs > 0.8 || cdcs < 0.2) {
      quality += 0.15;
    }

    // Penalize quality if noise detected
    const noise = this.detectLabelNoise(trainingSignal);
    if (noise.isNoisy) {
      quality -= 0.3 * noise.confidence;
    }

    // Penalize quality if fused confidence is low
    if (fusedLabel.confidence < 0.6) {
      quality -= 0.2;
    }

    return Math.max(0, Math.min(1, quality));
  }

  /**
   * Export user histories for persistence
   */
  exportUserHistories(): UserHistory[] {
    return Array.from(this.userHistories.values());
  }

  /**
   * Import user histories
   */
  importUserHistories(histories: UserHistory[]): void {
    this.userHistories.clear();
    for (const history of histories) {
      this.userHistories.set(history.userId, history);
    }
  }

  /**
   * Get user statistics
   */
  getUserStatistics(userId: string): UserHistory | null {
    return this.userHistories.get(userId) || null;
  }

  /**
   * Get top contributors
   */
  getTopContributors(limit: number = 10): UserHistory[] {
    return Array.from(this.userHistories.values())
      .sort((a, b) => b.totalLabels - a.totalLabels)
      .slice(0, limit);
  }

  /**
   * Get most accurate users
   */
  getMostAccurateUsers(minLabels: number = 10, limit: number = 10): UserHistory[] {
    return Array.from(this.userHistories.values())
      .filter(h => h.totalLabels >= minLabels)
      .sort((a, b) => b.accuracy - a.accuracy)
      .slice(0, limit);
  }
}

export const weakSignalService = new WeakSignalService();
