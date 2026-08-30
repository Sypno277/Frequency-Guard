import { describe, it, expect } from "vitest";
import {
  buildForensicParameterGrid,
  buildSpectrumData,
  buildWaveletLevels,
  buildFamilyAttribution,
} from "./analysisPresentation";
import type { AnalyzeResponse, SpectrumBin, WaveletBand } from "@/lib/api/client";

function makeAnalyzeResponse(overrides: Partial<AnalyzeResponse> = {}): AnalyzeResponse {
  return {
    request_id: "test-123",
    is_ai: true,
    verdict_state: "ai" as const,
    confidence: 87.3,
    fake_probability: 0.873,
    threshold: 0.5,
    attribution_mode: "heuristic",
    families: [
      { family: "gan", probability: 0.55 },
      { family: "diffusion", probability: 0.3 },
      { family: "real", probability: 0.1 },
      { family: "other", probability: 0.05 },
    ],
    features: {
      spectral_slope: 1.4,
      spectral_flatness: 0.12,
      phase_entropy: 0.55,
      high_freq_ratio: 0.21,
      dct_high_freq_ratio: 0.18,
      wavelet_detail_ratio: 0.42,
      noise_kurtosis: 2.1,
      texture_fractal_dim: 1.72,
      peak_prominence: 3.4,
    },
    spectrum: [
      { bin: 0, frequency: 0.0, magnitude: 0.5 },
      { bin: 1, frequency: 0.1, magnitude: 0.31 },
      { bin: 2, frequency: 0.2, magnitude: 0.19 },
    ],
    azimuthal: [
      { angle_deg: 0, magnitude: 0.4 },
      { angle_deg: 180, magnitude: 0.25 },
    ],
    wavelet_bands: [
      { level: 1, band: "LH", energy: 0.5, entropy: 0.72 },
      { level: 1, band: "HL", energy: 0.35, entropy: 0.65 },
      { level: 1, band: "HH", energy: 0.15, entropy: 0.5 },
      { level: 2, band: "LH", energy: 0.42, entropy: 0.68 },
    ],
    explainability: {
      saliency_png_base64: "data:image/png;base64,AAAA",
      patch_inconsistency_score: 0.34,
      mean_spectral_deviation: 0.22,
    },
    latency_ms: 145.2,
    image_size: [256, 256],
    model_version: "2.0.0",
    sha256: "abc123",
    ...overrides,
  };
}

describe("buildForensicParameterGrid", () => {
  it("maps measured features to exactly 10 parameter cards", () => {
    const raw = makeAnalyzeResponse();
    const grid = buildForensicParameterGrid(raw);
    expect(grid).toHaveLength(10);
  });

  it("scores spectral slope by deviation from the natural 1/f value of 2.0", () => {
    const raw = makeAnalyzeResponse();
    const grid = buildForensicParameterGrid(raw);
    const slopeCard = grid.find((g) => g.label === "Spectral Slope");
    expect(slopeCard).toBeDefined();
    // slope=1.4 → |1.4 - 2.0|/2 = 0.3
    expect(slopeCard!.score).toBeCloseTo(0.3, 5);
    expect(slopeCard!.value).toBe("1.400");
  });

  it("includes patch inconsistency from explainability payload", () => {
    const base = makeAnalyzeResponse();
    const raw = makeAnalyzeResponse({
      explainability: { ...base.explainability, patch_inconsistency_score: 0.66 },
    });
    const grid = buildForensicParameterGrid(raw);
    const patch = grid.find((g) => g.label === "Patch Inconsistency");
    expect(patch).toBeDefined();
    expect(patch!.score).toBeCloseTo(0.66, 5);
  });
});

describe("buildSpectrumData", () => {
  it("maps each radial bin to a chart point with hz + energy derived from magnitude", () => {
    const spectrum: SpectrumBin[] = [
      { bin: 0, frequency: 0, magnitude: 0.5 },
      { bin: 1, frequency: 0.25, magnitude: 0.3 },
    ];
    const data = buildSpectrumData(spectrum);
    expect(data).toHaveLength(2);
    expect(data[0].bin).toBe(0);
    expect(data[0].hz).toBe(0);
    expect(data[0].energy).toBeCloseTo(0.25, 5);
    expect(data[1].hz).toBe(Math.round(0.25 * 22050));
    expect(data[1].energy).toBeCloseTo(0.09, 5);
  });

  it("returns an empty array for empty input", () => {
    expect(buildSpectrumData([])).toEqual([]);
  });
});

describe("buildWaveletLevels", () => {
  it("groups bands by level and averages entropy", () => {
    const bands: WaveletBand[] = [
      { level: 1, band: "LH", energy: 0.5, entropy: 0.7 },
      { level: 1, band: "HL", energy: 0.3, entropy: 0.6 },
      { level: 2, band: "LH", energy: 0.4, entropy: 0.8 },
    ];
    const levels = buildWaveletLevels(bands);
    expect(levels).toHaveLength(2);
    expect(levels[0].id).toBe("L1");
    expect(levels[0].energy).toBeCloseTo(0.8, 5);
    expect(levels[0].entropy).toBeCloseTo(0.65, 5);
    expect(levels[1].id).toBe("L2");
  });
});

describe("buildFamilyAttribution", () => {
  it("sorts by probability descending and maps family names to labels", () => {
    const attribution = buildFamilyAttribution([
      { family: "real", probability: 0.1 },
      { family: "gan", probability: 0.55 },
      { family: "diffusion", probability: 0.3 },
    ]);
    expect(attribution[0].name).toBe("GAN");
    expect(attribution[0].probability).toBeCloseTo(0.55, 5);
    expect(attribution[1].name).toBe("Diffusion Model");
    expect(attribution[2].name).toBe("Real Camera");
  });
});
