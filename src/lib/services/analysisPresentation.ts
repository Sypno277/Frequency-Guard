/**
 * Presentation mappers (Masterplan §3.2 — rewritten).
 *
 * Previously this module generated seeded-PRNG spectra, wavelet levels,
 * model attributions, and forensic parameter grids. Every function now
 * maps REAL backend payloads (src/lib/api/client.ts types) to chart-ready
 * shapes. No randomness: identical input always yields identical output.
 */

import type {
  AnalyzeResponse,
  FamilyProbability,
  SpectrumBin,
  AzimuthalPoint,
  WaveletBand,
} from "@/lib/api/client";

export interface IdentificationParameter {
  label: string;
  value: string;
  score: number;
  description: string;
}

/** Map measured feature readings + explainability to dashboard cards. */
export function buildForensicParameterGrid(raw: AnalyzeResponse): IdentificationParameter[] {
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
      description:
        "8×8 block-DCT tail energy; diffusion flattening and JPEG fingerprints show up here.",
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
      description:
        "Shape of the SRM sensor-noise residual; synthetic noise deviates from camera noise.",
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
      description:
        "Detail vs total wavelet energy across scales; cross-scale regularity differs for AI.",
    },
    {
      label: "Patch Inconsistency",
      value: raw.explainability.patch_inconsistency_score.toFixed(3),
      score: Math.min(1, Math.max(0, raw.explainability.patch_inconsistency_score)),
      description:
        "How much per-tile spectra disagree; local synthesis defects break spatial consistency.",
    },
  ];
}

// --- chart-ready shapes (kept for potential Recharts consumers) ----------

export interface SpectrumPoint {
  bin: number;
  hz: number;
  magnitude: number;
  energy: number;
}

export interface WaveletLevel {
  id: string;
  name: string;
  energy: number;
  entropy: number;
}

const MAX_DISPLAY_HZ = 22050;

/** Radial spectrum bins → chart points (real magnitudes). */
export function buildSpectrumData(spectrum: SpectrumBin[]): SpectrumPoint[] {
  return spectrum.map((p) => ({
    bin: p.bin,
    hz: Math.round(p.frequency * MAX_DISPLAY_HZ),
    magnitude: p.magnitude,
    energy: p.magnitude * p.magnitude,
  }));
}

/** Wavelet bands → per-level aggregates (real energies/entropies). */
export function buildWaveletLevels(bands: WaveletBand[]): WaveletLevel[] {
  const map = new Map<number, WaveletBand[]>();
  for (const band of bands) {
    const list = map.get(band.level) ?? [];
    list.push(band);
    map.set(band.level, list);
  }
  return [...map.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([level, items]) => ({
      id: `L${level}`,
      name: `Level ${level}`,
      energy: items.reduce((s, b) => s + b.energy, 0),
      entropy: items.reduce((s, b) => s + b.entropy, 0) / items.length,
    }));
}

/** Real azimuthal profile passthrough with normalized labels. */
export function buildAzimuthalSeries(azimuthal: AzimuthalPoint[]) {
  return azimuthal.map((p) => ({ angle: `${p.angle_deg}°`, magnitude: p.magnitude }));
}

/**
 * Family attribution → sorted display candidates.
 * Uses ONLY backend-calibrated probabilities; family names map to
 * human-readable labels.
 */
export function buildFamilyAttribution(
  families: FamilyProbability[]
): { name: string; probability: number }[] {
  const labels: Record<string, string> = {
    real: "Real Camera",
    diffusion: "Diffusion Model",
    gan: "GAN",
    other: "Other / Unknown",
  };
  return [...families]
    .sort((a, b) => b.probability - a.probability)
    .map((f) => ({ name: labels[f.family] ?? f.family, probability: f.probability }));
}
