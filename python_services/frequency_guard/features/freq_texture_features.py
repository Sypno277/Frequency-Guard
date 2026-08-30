"""Frequency-domain texture features (E-Masterplan E1 #5).

Second-order statistics of the *frequency domain*: local binary patterns
(LBP) computed on wavelet sub-band images, and co-occurrence statistics of
DCT coefficients across adjacent blocks. First-order statistics (means,
variances — v2's bread and butter) miss spatial organization of frequency
artifacts; generators impose block-regular structure that second-order
measures expose.

CPU-cheap: LBP on small sub-bands and a vectorized co-occurrence histogram.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pywt

from ..config import Settings

_EPS = 1e-12


@dataclass(frozen=True)
class FreqTextureFeatures:
    """LBP-on-subband + DCT co-occurrence second-order statistics."""

    lbp_uniformity: float  # fraction of uniform LBP patterns (smooth textures)
    lbp_entropy: float  # entropy of the LBP histogram
    subband_lbp_contrast: float  # mean LBP contrast measure across sub-bands
    dct_cooc_homogeneity: float  # GLCM-like homogeneity of DCT magnitudes
    dct_cooc_contrast: float
    subband_grad_consistency: float  # orientation consistency of sub-band gradients

    def as_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.lbp_uniformity,
                self.lbp_entropy,
                self.subband_lbp_contrast,
                self.dct_cooc_homogeneity,
                self.dct_cooc_contrast,
                self.subband_grad_consistency,
            ],
            dtype=np.float64,
        )


def _lbp_basic(img: np.ndarray) -> np.ndarray:
    """Basic 8-neighbor LBP map (uint8 codes) for a float32 image."""
    img_u8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    h, w = img_u8.shape
    if h < 3 or w < 3:
        return np.zeros((0, 0), dtype=np.uint8)
    center = img_u8[1:-1, 1:-1]
    codes = np.zeros((h - 2, w - 2), dtype=np.uint8)
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = img_u8[1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx]
        codes |= (neighbor >= center).astype(np.uint8) << bit
    return codes


def _lbp_stats(codes: np.ndarray) -> tuple[float, float, float]:
    """Return (uniformity, entropy, contrast-proxy) of an LBP code map."""
    if codes.size == 0:
        return 0.0, 0.0, 0.0
    hist = np.bincount(codes.ravel(), minlength=256).astype(np.float64)
    hist /= hist.sum() + _EPS

    # Uniform patterns have at most 2 bitwise transitions (0->1, 1->0).
    uniform_codes = [c for c in range(256) if bin(c).count("01") + bin(c).count("10") <= 2]
    uniformity = float(hist[uniform_codes].sum())

    nz = hist[hist > 0]
    entropy = float(-np.sum(nz * np.log2(nz)) / np.log2(256))

    # Contrast proxy: variance of code values.
    vals = np.arange(256, dtype=np.float64)
    mean = float(np.sum(hist * vals))
    contrast = float(np.sum(hist * (vals - mean) ** 2))

    return round(uniformity, 4), round(entropy, 4), round(contrast, 2)


def _dct_cooccurrence(gray: np.ndarray, block: int) -> tuple[float, float]:
    """Homogeneity + contrast of the DCT-magnitude co-occurrence across blocks.

    For each block we take the mean |DCT| in the low band and the high band;
    the joint histogram of (low, high) across blocks is scored GLCM-style.
    """
    h, w = gray.shape
    th = (h // block) * block
    tw = (w // block) * block
    if th < block or tw < block:
        return 0.0, 0.0
    tiles = gray[:th, :tw].reshape(th // block, block, tw // block, block)
    tiles = tiles.transpose(0, 2, 1, 3).reshape(-1, block, block)

    half = block // 2
    low_band = tiles[:, :half, :half]
    high_band = tiles[:, half:, half:]

    low_mag = np.mean(np.abs(low_band), axis=(1, 2))
    high_mag = np.mean(np.abs(high_band), axis=(1, 2))

    # 16x16 joint histogram over log-magnitude bins.
    def log_bins(arr: np.ndarray) -> np.ndarray:
        m = np.log1p(np.maximum(arr, 0.0))
        lo, hi = float(m.min()), float(m.max() + _EPS)
        if hi - lo < _EPS:
            return np.zeros_like(m, dtype=np.int64)
        return np.clip(((m - lo) / (hi - lo) * 15), 0, 15).astype(np.int64)

    lb = log_bins(low_mag)
    hb = log_bins(high_mag)
    joint = np.bincount(lb * 16 + hb, minlength=256).astype(np.float64).reshape(16, 16)
    joint /= joint.sum() + _EPS

    i, j = np.mgrid[0:16, 0:16]
    homogeneity = float(np.sum(joint / (1.0 + np.abs(i - j))))
    contrast = float(np.sum(joint * (i - j) ** 2))
    return round(homogeneity, 4), round(contrast, 4)


def _subband_gradient_consistency(gray: np.ndarray, wavelet: str, levels: int) -> float:
    """Mean gradient-orientation consistency of the finest HH sub-band.

    Natural noise has random gradient orientation; structured synthesis
    artifacts produce locally-aligned gradients.
    """
    coeffs = pywt.wavedec2(np.asarray(gray, dtype=np.float32), wavelet, level=min(levels, 3))
    hh = coeffs[-1][2]  # finest HH
    gx = cv2.Sobel(hh, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(hh, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    if mag.mean() < _EPS:
        return 0.0
    strong = mag > (mag.mean() + mag.std())
    if strong.sum() < 10:
        return 0.0
    angles = np.arctan2(gy[strong], gx[strong])
    # Resultant length of the angle distribution: 1 = perfectly aligned.
    c = np.mean(np.cos(2 * angles))
    s = np.mean(np.sin(2 * angles))
    return float(np.clip(np.sqrt(c**2 + s**2), 0.0, 1.0))


def extract_freq_texture_features(gray: np.ndarray, settings: Settings) -> FreqTextureFeatures:
    """Extract second-order frequency-domain texture features.

    Args:
        gray: HxW float32 in [0, 1].
        settings: wavelet_name / wavelet_levels / dct_block_size.

    Returns:
        FreqTextureFeatures.
    """
    g = np.asarray(gray, dtype=np.float32)

    # LBP on the two finest wavelet detail sub-bands (LH of level 1).
    try:
        coeffs = pywt.wavedec2(g, settings.wavelet_name, level=min(settings.wavelet_levels, 2))
        sub = coeffs[-1][0]
    except ValueError:
        sub = g
    codes = _lbp_basic(sub)
    uniformity, entropy, contrast = _lbp_stats(codes)

    homo, contra = _dct_cooccurrence(g, settings.dct_block_size)
    grad_cons = _subband_gradient_consistency(g, settings.wavelet_name, settings.wavelet_levels)

    return FreqTextureFeatures(
        lbp_uniformity=uniformity,
        lbp_entropy=entropy,
        subband_lbp_contrast=contrast,
        dct_cooc_homogeneity=homo,
        dct_cooc_contrast=contra,
        subband_grad_consistency=round(grad_cons, 4),
    )
