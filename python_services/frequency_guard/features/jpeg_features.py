"""JPEG quantization & double-compression fingerprints (E-Masterplan E1 #3).

A genuine camera JPEG has a specific quantization-table signature. A
generated image that is later saved as JPEG goes through a *second*
compression, leaving periodic DCT-histogram artifacts (double-compression
fingerprint). This module estimates the JPEG quality and recovers
double-compression signals from the DCT coefficient histograms without a
full JPEG parser, using only the luma plane and numpy/scipy.

The signal is strongest for authentic-side discrimination: real photos have
clean single-compression histograms, generated-then-saved images show
garbled quantization modes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dctn

from ..config import Settings

_EPS = 1e-12


@dataclass(frozen=True)
class JPEGFeatures:
    """DCT-histogram statistics that fingerprint JPEG processing history."""

    quant_estimate: float  # inferred JPEG quality 0..100
    dct_hist_kurtosis: float  # peakiness of DCT coefficient histogram
    dct_hist_periodicity: float  # double-compression grid periodicity strength
    dct_zero_ratio: float  # fraction of (near-)zero quantized coefficients
    dct_mode_spread: float  # spread of the dominant histogram mode (broad vs sharp)

    def as_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.quant_estimate,
                self.dct_hist_kurtosis,
                self.dct_hist_periodicity,
                self.dct_zero_ratio,
                self.dct_mode_spread,
            ],
            dtype=np.float64,
        )


def _quantize_luma(gray: np.ndarray, block: int) -> np.ndarray:
    """8x8 block DCT coefficients; returns (n_blocks, block, block)."""
    h, w = gray.shape
    th = (h // block) * block
    tw = (w // block) * block
    tiles = gray[:th, :tw].reshape(th // block, block, tw // block, block)
    tiles = tiles.transpose(0, 2, 1, 3).reshape(-1, block, block)
    # dctn over the last two axes; type 2 orthonormal (scipy default)
    coeffs = np.empty_like(tiles, dtype=np.float64)
    for i in range(tiles.shape[0]):
        coeffs[i] = dctn(tiles[i], type=2, norm="ortho")
    return coeffs


def _estimate_quality(coeffs: np.ndarray) -> float:
    """Rough quality estimate from the AC-energy fraction of the 8x8 grid.

    Heuristic: JPEG quality tracks the low-frequency concentration of the
    block-DCT coefficients (higher quality retains more mid/high AC energy).
    """
    ac = coeffs[:, 1:, :]
    ac2 = ac[:, :, 1:]
    total_energy = float(np.sum(coeffs**2)) + _EPS
    ac_energy = float(np.sum(ac2**2))
    ac_frac = ac_energy / total_energy
    # Map the AC fraction (typically 0.1..0.6) to a quality 40..100.
    qual = float(np.clip(40.0 + (ac_frac - 0.1) * (60.0 / 0.5), 0.0, 100.0))
    return round(qual, 2)


def _histogram_stats(coeffs: np.ndarray) -> tuple[float, float, float, float]:
    """Return (kurtosis, periodicity, zero_ratio, mode_spread) of DCT hist."""
    ac = coeffs[:, 1:, 1:].ravel()  # AC coefficients only
    hist, edges = np.histogram(ac, bins=64, range=(ac.min(), ac.max()))
    hist = hist.astype(np.float64) + _EPS
    p = hist / hist.sum()

    # Kurtosis of the histogram (peakiness).
    centers = (edges[:-1] + edges[1:]) / 2.0
    mean = float(np.sum(p * centers))
    var = float(np.sum(p * (centers - mean) ** 2))
    kurt = float(np.sum(p * (centers - mean) ** 4) / (var**2 + _EPS)) if var > _EPS else 0.0

    # Periodicity: autocorrelation peak in the histogram away from lag 0.
    ac_hist = hist - hist.mean()
    acorr = np.correlate(ac_hist, ac_hist, mode="full")[len(hist) - 1 :]
    acorr = acorr / (acorr[0] + _EPS)
    periodicity = float(np.max(acorr[2 : min(12, len(acorr))])) if len(acorr) > 4 else 0.0

    # Fraction of near-zero coefficients (strong quantization -> many zeros).
    zero_ratio = float(np.mean(np.abs(ac) < 0.5))

    # Spread of the dominant mode (broad camera vs sharp/patchy generated).
    peak_idx = int(np.argmax(hist))
    lo = max(0, peak_idx - 3)
    hi = min(len(hist), peak_idx + 4)
    mode_mass = float(hist[lo:hi].sum() / hist.sum())
    mode_spread = float(np.clip(mode_mass, 0.0, 1.0))

    return round(kurt, 4), round(periodicity, 6), round(zero_ratio, 6), round(mode_spread, 4)


def extract_jpeg_features(gray: np.ndarray, settings: Settings) -> JPEGFeatures:
    """Extract JPEG compression-history fingerprints.

    Args:
        gray: HxW float32 in [0, 1].
        settings: ``dct_block_size`` (default 8) controls the block grid.

    Returns:
        JPEGFeatures.
    """
    coef = _quantize_luma(np.asarray(gray, dtype=np.float32), settings.dct_block_size)
    qual = _estimate_quality(coef)
    kurt, periodicity, zero_ratio, mode_spread = _histogram_stats(coef)
    return JPEGFeatures(
        quant_estimate=qual,
        dct_hist_kurtosis=kurt,
        dct_hist_periodicity=periodicity,
        dct_zero_ratio=zero_ratio,
        dct_mode_spread=mode_spread,
    )
