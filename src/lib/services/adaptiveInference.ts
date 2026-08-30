/**
 * Adaptive inference service (Masterplan §3.2 — rewritten).
 *
 * Previously derived verdicts from a hash of the file bytes; now a thin
 * adapter over the real backend API. The forensic parameter grid is built
 * from the actual feature readings returned by the pipeline, so every
 * number shown in the dashboard traces to the image's measured spectrum.
 */

import { analyzeImage, type AnalyzeResponse } from "@/lib/api/client";
import {
  resolveVerdictPresentation,
  type VerdictPresentation,
} from "@/lib/services/verdictState";

export interface ForensicParameter {
  label: string;
  value: string;
  score: number;
  description: string;
}

export interface AdaptiveAnalysisResult {
  isAI: boolean;
  confidence: number;
  model: string;
  raw: AnalyzeResponse;
  forensicParameters: ForensicParameter[];
  verdict: VerdictPresentation;
}

/** Map real feature readings to dashboard-friendly parameter cards. */
function buildForensicParameterGrid(raw: AnalyzeResponse): ForensicParameter[] {
  const f = raw.features;
  return [
    {
      label: "Spectral Slope",
      value: f.spectral_slope.toFixed(3),
      // |slope| near 2.0 = natural 1/f law; deviation raises suspicion.
      score: Math.min(1, Math.abs(Math.abs(f.spectral_slope) - 2.0) / 2),
      description:
        "How fast image detail decays with frequency; cameras follow natural 1/f statistics (α ≈ 2).",
    },
    {
      label: "Spectral Flatness",
      value: f.spectral_flatness.toFixed(4),
      score: Math.min(1, f.spectral_flatness * 8),
      description:
        "Harmonic smoothness vs peaky structure; generator noise floors flatten the spectrum.",
    },
    {
      label: "Phase Entropy",
      value: f.phase_entropy.toFixed(3),
      score: Math.min(1, Math.max(0, 1 - f.phase_entropy)),
      description:
        "Randomness of high-frequency phase; natural photos are near-random, generators correlate it.",
    },
    {
      label: "High-Freq Energy",
      value: `${(f.high_freq_ratio * 100).toFixed(2)}%`,
      score: Math.min(1, f.high_freq_ratio * 20),
      description: "Share of energy in the highest frequency band; upsampling leaves spikes here.",
    },
    {
      label: "DCT High-Freq",
      value: `${(f.dct_high_freq_ratio * 100).toFixed(2)}%`,
      score: Math.min(1, f.dct_high_freq_ratio * 12),
      description: "8×8 block-DCT tail energy; diffusion flattening and JPEG fingerprints show up here.",
    },
    {
      label: "Checkerboard Peaks",
      value: f.peak_prominence.toFixed(2),
      score: Math.min(1, f.peak_prominence / 10),
      description: "Strongest spectral peak vs median — GAN conv-transpose upsampling echoes.",
    },
    {
      label: "Noise Kurtosis",
      value: f.noise_kurtosis.toFixed(2),
      score: Math.min(1, Math.abs(f.noise_kurtosis) / 6),
      description: "Shape of the SRM sensor-noise residual; synthetic noise deviates from camera noise.",
    },
    {
      label: "Fractal Dimension",
      value: f.texture_fractal_dim.toFixed(3),
      score: Math.min(1, Math.max(0, (f.texture_fractal_dim - 1.0) / 0.9)),
      description: "Edge-map complexity via box counting; over-smoothed textures drift low.",
    },
    {
      label: "Wavelet Detail Ratio",
      value: f.wavelet_detail_ratio.toFixed(3),
      score: Math.min(1, f.wavelet_detail_ratio * 2),
      description: "Detail vs total wavelet energy across scales; cross-scale regularity differs for AI.",
    },
    {
      label: "Patch Inconsistency",
      value: raw.explainability.patch_inconsistency_score.toFixed(3),
      score: raw.explainability.patch_inconsistency_score,
      description: "How much per-tile spectra disagree; local synthesis defects break spatial consistency.",
    },
  ];
}

/** Top family label used as the "model" chip in the verdict card. */
function topFamilyLabel(raw: AnalyzeResponse): string {
  if (!raw.is_ai) return "Real Camera";
  const top = [...raw.families].sort((a, b) => b.probability - a.probability)[0];
  if (!top || top.family === "real") return "Unknown Generator";
  return top.family === "diffusion"
    ? "Diffusion Model"
    : top.family === "gan"
      ? "GAN"
      : "Other Generator";
}

/** Run the real analysis pipeline on an uploaded image. */
export async function analyzeWithTrainingSignal(file: File): Promise<AdaptiveAnalysisResult> {
  const raw = await analyzeImage(file);
  return {
    isAI: raw.is_ai,
    confidence: raw.confidence,
    model: topFamilyLabel(raw),
    raw,
    forensicParameters: buildForensicParameterGrid(raw),
    verdict: resolveVerdictPresentation(
      raw.verdict_state,
      raw.confidence,
      raw.evidence?.agreement ?? undefined,
    ),
  };
}
