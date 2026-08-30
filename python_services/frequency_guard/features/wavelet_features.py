"""Wavelet sub-band statistics (pywt.wavedec2).

Multi-resolution decomposition with a Daubechies/symlet wavelet separates
image structure across scales. AI generators leave cross-scale regularity
drifts: sub-band energy ratios, entropy, and energy-per-scale trajectories
differ measurably from natural images. Pure python/numpy on top of pywt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt

from ..config import Settings

_EPS = 1e-12


@dataclass(frozen=True)
class WaveletFeatures:
    """Wavelet sub-band statistics."""

    subband_energies: np.ndarray  # per-level [LH, HL, HH] energy ratios
    subband_entropies: np.ndarray  # per-level per-subband entropy
    detail_energy_ratio: float  # total detail / total energy
    cross_scale_ratio: float  # high-freq level 1 / low-freq coarsest level
    ll_energy_ratio: float  # approximation energy share

    def as_vector(self) -> np.ndarray:
        return np.concatenate(
            (
                self.subband_energies,
                self.subband_entropies,
                np.asarray(
                    [
                        self.detail_energy_ratio,
                        self.cross_scale_ratio,
                        self.ll_energy_ratio,
                    ],
                    dtype=np.float64,
                ),
            )
        )


def _level_entropy(coeffs: np.ndarray, bins: int = 32) -> float:
    """Shannon entropy (normalized to [0,1]) of a sub-band's coefficients."""
    flat = coeffs.ravel()
    if flat.size == 0:
        return 0.0
    hist, _ = np.histogram(flat, bins=bins)
    hist = hist.astype(np.float64)
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    p = hist / hist.sum()
    return float(-(p * np.log2(p)).sum() / np.log2(bins))


def extract_wavelet_features(gray: np.ndarray, settings: Settings) -> WaveletFeatures:
    """Extract wavelet features from a normalized grayscale image.

    Args:
        gray: HxW float32 in [0, 1].
        settings: configuration (wavelet_name, wavelet_levels).

    Returns:
        WaveletFeatures.
    """
    if not pywt.Wavelet(settings.wavelet_name).orthogonal:
        raise ValueError(f"Wavelet '{settings.wavelet_name}' must be orthogonal")

    coeffs = pywt.wavedec2(gray, settings.wavelet_name, level=settings.wavelet_levels)
    ll = coeffs[0]
    details = coeffs[1:]  # list of (cH, cV, cD) per level, level 1 = finest

    n_levels = len(details)
    energies = np.zeros((n_levels, 3), dtype=np.float64)
    entropies = np.zeros((n_levels, 3), dtype=np.float64)

    total_energy = float(np.sum(np.asarray(gray, dtype=np.float64) ** 2)) + _EPS

    for i, (c_h, c_v, c_d) in enumerate(details):
        for j, sub in enumerate((c_h, c_v, c_d)):
            sub = np.asarray(sub, dtype=np.float64)
            energies[i, j] = float(np.sum(sub**2))
            entropies[i, j] = _level_entropy(sub)

    detail_total = energies.sum()
    ll_energy = float(np.sum(np.asarray(ll, dtype=np.float64) ** 2))

    # Normalize subband energies to unit sum per level for scale-invariance
    level_sums = energies.sum(axis=1, keepdims=True)
    level_sums[level_sums < 1e-12] = 1.0
    normalized = energies / level_sums

    finest = energies[0, :].sum() if n_levels > 0 else 0.0
    coarsest = energies[-1, :].sum() if n_levels > 0 else 0.0

    return WaveletFeatures(
        subband_energies=normalized.ravel(),
        subband_entropies=entropies.ravel(),
        detail_energy_ratio=detail_total / total_energy,
        cross_scale_ratio=(finest + _EPS) / (coarsest + _EPS),
        ll_energy_ratio=ll_energy / total_energy,
    )
