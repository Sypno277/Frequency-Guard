"""Noise-residual (SRM) features.

AI generators do not produce sensor-like noise. Real camera images carry a
spatially consistent, per-pixel noise pattern shaped by the sensor; SGM
residuals (high-pass filters from steganalysis) expose this residual layer.
We compute statistics of the SRM residual (kurtosis, variance, block-level
consistency) which separate natural sensor noise from synthetic smoothness
or over-sharpened generator output. Pure numpy/OpenCV convolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import Settings

_EPS = 1e-12


# Classic SRM high-pass kernels (normalized). These emphasize subtle local
# discontinuities that survive compression and are weak in generator output.
_SRM_KERNELS: tuple[np.ndarray, ...] = (
    np.asarray(
        [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    ),
    np.asarray(
        [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ],
        dtype=np.float64,
    ),
    np.asarray(
        [
            [0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0],
            [0, -1, 4, -1, 0],
            [0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    ),
)


@dataclass(frozen=True)
class NoiseFeatures:
    """Statistics over the SRM residual layer."""

    residual_kurtosis: float
    residual_std: float
    residual_mean_abs: float
    block_consistency: float  # 1 - normalized std of block-level residual energy
    peak_signal_flatness: float  # spectral flatness of the residual layer

    def as_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.residual_kurtosis,
                self.residual_std,
                self.residual_mean_abs,
                self.block_consistency,
                self.peak_signal_flatness,
            ],
            dtype=np.float64,
        )


def _filter_residual(gray: np.ndarray, block: int = 3) -> np.ndarray:
    """Combine SRM kernel responses into one residual map (float64)."""
    residual = np.zeros_like(gray, dtype=np.float64)
    for kernel in _SRM_KERNELS:
        denom = np.abs(kernel).sum()
        if denom < 1e-9:
            continue
        norm = kernel / denom
        response = cv2.filter2D(gray, -1, norm, borderType=cv2.BORDER_REFLECT)
        residual += response
    return residual / max(1, len(_SRM_KERNELS))


def _spectral_flatness(signal: np.ndarray) -> float:
    """Geometric/arithmetic mean of power spectrum (in [0,1])."""
    power = np.abs(np.fft.rfft(signal - signal.mean())) ** 2
    power = power[power > _EPS]
    if power.size < 4:
        return 0.0
    geo = np.exp(np.mean(np.log(power)))
    arith = np.mean(power)
    return float(geo / (arith + _EPS))


def extract_noise_features(gray: np.ndarray, settings: Settings) -> NoiseFeatures:
    """Extract SRM-residual features from a normalized grayscale image.

    Args:
        gray: HxW float32 in [0, 1].
        settings: unused beyond API consistency (kept for symmetry).

    Returns:
        NoiseFeatures.
    """
    residual = _filter_residual(gray)

    flat = residual.ravel()
    n = flat.size
    mean = float(flat.mean()) if n else 0.0
    std = float(flat.std()) if n else 0.0
    mean_abs = float(np.mean(np.abs(flat))) if n else 0.0

    kurtosis = 0.0
    if n > 4 and std > _EPS:
        m4 = float(((flat - mean) ** 4).mean())
        m2 = float(((flat - mean) ** 2).mean())
        kurtosis = m4 / (m2**2 + _EPS) - 3.0

    # Block-level consistency: split residual into 32x32 tiles, measure
    # energy dispersion across tiles. Real sensor noise is uniform-ish;
    # generative artifacts concentrate energy in localized regions.
    h, w = residual.shape
    ts = 32
    if h >= ts and w >= ts:
        th, tw = (h // ts) * ts, (w // ts) * ts
        tiles = residual[:th, :tw].reshape(th // ts, ts, tw // ts, ts)
        tiles = tiles.transpose(0, 2, 1, 3).reshape(-1, ts, ts)
        # Vectorized per-tile energy: one einsum instead of a Python loop.
        tile_energy = np.einsum("tij,tij->t", tiles, tiles)
        te_std = float(tile_energy.std()) if tile_energy.size else 0.0
        te_mean = float(tile_energy.mean()) if tile_energy.size else 1.0
        block_consistency = 1.0 - te_std / (te_mean + _EPS)
        block_consistency = float(np.clip(block_consistency, 0.0, 1.0))
    else:
        block_consistency = 0.0

    return NoiseFeatures(
        residual_kurtosis=kurtosis,
        residual_std=std,
        residual_mean_abs=mean_abs,
        block_consistency=block_consistency,
        peak_signal_flatness=_spectral_flatness(flat),
    )
