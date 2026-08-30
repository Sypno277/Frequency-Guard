"""Chroma-channel spectral features (E-Masterplan E1 #1).

v2 pipeline is luma-dominant: every extractor consumes ``PreprocessedImage.y_channel``.
But generative upsamplers operate on all three channel planes, and the inter-channel
statistics of real photos are strongly constrained by Bayer/CFA demosaicing. This
module extracts FFT/DCT statistics from the Cb and Cr chroma planes plus inter-channel
correlation structure, which luma analysis alone cannot see.

Uses only numpy/opencv (the chroma planes are small; a single 256px image costs a few ms).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import Settings
from ..preprocess import PreprocessedImage

_EPS = 1e-12


@dataclass(frozen=True)
class ChromaFeatures:
    """Chroma-plane spectral statistics + inter-channel correlation structure."""

    chroma_flatness: float
    chroma_high_freq_ratio: float
    chroma_slope: float
    cb_dc_ratio: float
    cr_dc_ratio: float
    cb_cr_corr: float
    rg_corr: float
    gb_corr: float
    chroma_entropy: float
    chroma_radial_kurtosis: float

    def as_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.chroma_flatness,
                self.chroma_high_freq_ratio,
                self.chroma_slope,
                self.cb_dc_ratio,
                self.cr_dc_ratio,
                self.cb_cr_corr,
                self.rg_corr,
                self.gb_corr,
                self.chroma_entropy,
                self.chroma_radial_kurtosis,
            ],
            dtype=np.float64,
        )


def _flatness(power: np.ndarray) -> float:
    """Geometric/arithmetic mean ratio of a power spectrum (0=peaky, 1=flat)."""
    flat = power.ravel()
    flat = flat[flat > _EPS]
    if flat.size < 4:
        return 0.0
    return float(np.exp(np.mean(np.log(flat))) / (np.mean(flat) + _EPS))


def _radius_grid(shape: tuple[int, int]) -> np.ndarray:
    """Float64 radial-distance grid for a 2D spectrum shape."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    return np.sqrt(xx.astype(np.float64) ** 2 + yy.astype(np.float64) ** 2)


def _spectral_slope(magnitude: np.ndarray) -> float:
    """Least-squares log-log slope over radii 3..0.8*rmax."""
    radius = _radius_grid(magnitude.shape)
    mask = (radius > 2.0) & (radius < radius.max() * 0.8)
    if mask.sum() < 5:
        return 0.0
    x = np.log(radius[mask])
    y = np.log(magnitude[mask] + _EPS)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _channel_stats(channel: np.ndarray) -> tuple[float, float, float, float, float]:
    """Return (flatness, high_freq_ratio, slope, dc_ratio, entropy) for one plane."""
    centered = channel - channel.mean()
    spectrum = np.fft.rfft2(centered)
    magnitude = np.abs(spectrum)
    power = magnitude**2
    total = float(power.sum()) + _EPS

    nyquist = min(power.shape[0], power.shape[1]) // 2
    high = float(power[-nyquist:, -nyquist:].sum()) if nyquist > 2 else 0.0
    dc = float(power[0, 0])

    flatness = _flatness(power)
    slope = _spectral_slope(magnitude)
    high_ratio = high / total
    dc_ratio = dc / total

    # Phase entropy of the plane.
    phase = np.angle(spectrum)
    hist, _ = np.histogram(phase, bins=16, range=(-np.pi, np.pi))
    hist = hist.astype(np.float64)
    hist = hist[hist > 0]
    entropy = (
        float(-np.sum((hist / hist.sum()) * np.log2(hist / hist.sum() + _EPS)) / np.log2(16))
        if hist.size
        else 0.0
    )

    return flatness, high_ratio, slope, dc_ratio, entropy


def _radius_mask(shape: tuple[int, int]) -> np.ndarray:
    """Boolean mask of the outer half of a 2D shape (radius > 0.5*rmax)."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.sqrt(xx.astype(np.float64) ** 2 + yy.astype(np.float64) ** 2)
    return radius > radius.max() * 0.5


def _radial_kurtosis(channel: np.ndarray) -> float:
    """Kurtosis of the radial energy profile: peaky vs spread spectra."""
    centered = channel - channel.mean()
    magnitude = np.abs(np.fft.rfft2(centered))
    h, w = magnitude.shape
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.sqrt(xx.astype(np.float64) ** 2 + yy.astype(np.float64) ** 2)
    bins = np.linspace(0.0, radius.max(), 24)
    idx = np.digitize(radius.ravel(), bins) - 1
    idx = np.clip(idx, 0, len(bins) - 2)
    flat = magnitude.ravel()
    sums = np.bincount(idx, weights=flat, minlength=len(bins) - 1)
    counts = np.bincount(idx, minlength=len(bins) - 1)
    counts[counts == 0] = 1
    profile = sums / counts
    if profile.std() < _EPS:
        return 0.0
    return float(((profile - profile.mean()) ** 4).mean() / (profile.std() ** 4 + _EPS))


def extract_chroma_features(prep: PreprocessedImage, settings: Settings) -> ChromaFeatures:
    """Extract chroma-plane + inter-channel correlation features.

    Args:
        prep: PreprocessedImage (uses ``rgb`` and ``y_channel``).
        settings: unused beyond API consistency (kept for symmetry).

    Returns:
        ChromaFeatures.
    """
    rgb = np.asarray(prep.rgb, dtype=np.float32)
    # YCrCb -> chroma planes (normalized 0..1)
    ycrcb = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb) / 255.0
    cb = ycrcb[:, :, 1].astype(np.float32)
    cr = ycrcb[:, :, 2].astype(np.float32)

    cb_flat, cb_high, cb_slope, cb_dc, cb_ent = _channel_stats(cb)
    cr_flat, cr_high, cr_slope, cr_dc, cr_ent = _channel_stats(cr)

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    def corr(a: np.ndarray, b_: np.ndarray) -> float:
        if a.std() < _EPS or b_.std() < _EPS:
            return 0.0
        return float(np.corrcoef(a.ravel(), b_.ravel())[0, 1])

    return ChromaFeatures(
        chroma_flatness=round((cb_flat + cr_flat) / 2.0, 6),
        chroma_high_freq_ratio=round((cb_high + cr_high) / 2.0, 6),
        chroma_slope=round((cb_slope + cr_slope) / 2.0, 4),
        cb_dc_ratio=round(cb_dc, 6),
        cr_dc_ratio=round(cr_dc, 6),
        cb_cr_corr=round(corr(cb, cr), 4),
        rg_corr=round(corr(r, g), 4),
        gb_corr=round(corr(g, b), 4),
        chroma_entropy=round((cb_ent + cr_ent) / 2.0, 4),
        chroma_radial_kurtosis=round((_radial_kurtosis(cb) + _radial_kurtosis(cr)) / 2.0, 4),
    )
