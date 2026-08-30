"""Higher-order spectral (bispectrum) features (E-Masterplan E1 #2).

Natural image frequency components have statistically *independent* phase;
generative models introduce phase coupling across frequency triples. The
bispectrum — the Fourier transform of the third-order cumulant — is the
canonical tool for measuring that phase coupling, and it is symmetric under
translation, making it robust to cropping.

Computing the full 2D bispectrum is O(N^4); we reduce to an O(N^2) estimate
by summing ``X(k1 + k2) * conj(X(k1)) * conj(X(k2))`` over a bounded,
deterministically-seeded subset of mid-band frequency pairs and summarizing
the result with scalar statistics that capture "how phase-coupled" the
image is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import fft2

from ..config import Settings

_EPS = 1e-12


@dataclass(frozen=True)
class HOSFeatures:
    """Bispectrum-based phase-coupling statistics (translation-invariant)."""

    bicoherence: float  # 0 = independent phase, 1 = fully coupled
    bispectrum_kurtosis: float
    phase_coupling_entropy: float
    bicoherence_skew: float
    spectral_asymmetry: float  # quadrant energy imbalance (directional probe)

    def as_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.bicoherence,
                self.bispectrum_kurtosis,
                self.phase_coupling_entropy,
                self.bicoherence_skew,
                self.spectral_asymmetry,
            ],
            dtype=np.float64,
        )


def _empty_result() -> dict[str, float]:
    """Degenerate result when the mid-band is too small to sample."""
    return {
        "bicoherence": 0.0,
        "bispectrum_kurtosis": 0.0,
        "phase_coupling_entropy": 0.0,
        "bicoherence_skew": 0.0,
        "spectral_asymmetry": 0.0,
    }


def _phase_coupling_snapshot(img: np.ndarray, max_samples: int) -> dict[str, float]:
    """Estimate bicoherence on a bounded sample of mid-band frequency pairs."""
    # Crop to even dimensions for clean FFT alignment; remove DC.
    h, w = img.shape
    crop_h, crop_w = h - (h % 2), w - (w % 2)
    centered = img[:crop_h, :crop_w] - img[:crop_h, :crop_w].mean()
    spec = fft2(centered)

    # Positive-frequency quadrant, DC masked.
    half = spec[: crop_h // 2, : crop_w // 2]
    hh, ww = half.shape
    if hh < 12 or ww < 12:
        return _empty_result()

    yy, xx = np.mgrid[0:hh, 0:ww]
    radius = np.sqrt(xx.astype(np.float64) ** 2 + yy.astype(np.float64) ** 2)
    rmax = radius.max()
    mid_band = (radius > 4.0) & (radius < rmax * 0.6)
    available = np.flatnonzero(mid_band)
    if available.size < 4:
        return _empty_result()

    rng = np.random.default_rng(42)
    n_pairs = min(max_samples, available.size)
    pairs = rng.choice(available, size=(n_pairs, 2), replace=True)
    k1 = np.unravel_index(pairs[:, 0], (hh, ww))
    k2 = np.unravel_index(pairs[:, 1], (hh, ww))

    sum0 = (k1[0] + k2[0]) % hh
    sum1 = (k1[1] + k2[1]) % ww

    x1 = half[k1]
    x2 = half[k2]
    xsum = half[sum0, sum1]

    triple = xsum * np.conj(x1) * np.conj(x2)

    # Bicoherence: normalized mean bispectral magnitude, bounded [0, 1].
    numerator = np.abs(np.mean(triple))
    denominator = np.mean(np.abs(xsum) * np.abs(x1) * np.abs(x2)) + _EPS
    bicoh = float(np.clip(numerator / denominator, 0.0, 1.0))

    # Kurtosis of the real bispectrum (peakiness of phase coupling).
    real_t = triple.real
    if real_t.std() > _EPS:
        kurt = float(((real_t - real_t.mean()) ** 4).mean() / (real_t.std() ** 4 + _EPS))
    else:
        kurt = 0.0

    # Entropy of the |bispectrum| distribution (uniform = weak coupling).
    mags_t = np.abs(triple)
    total = float(mags_t.sum())
    if total > _EPS:
        p = mags_t / total
        p = p[p > 0]
        entropy = float(-np.sum(p * np.log2(p)) / np.log2(len(p))) if len(p) else 0.0
    else:
        entropy = 0.0

    # Skewness of per-sample bispectral magnitude.
    samples = mags_t
    if samples.std() > _EPS:
        skew = float((((samples - samples.mean()) / (samples.std() + _EPS)) ** 3).mean())
    else:
        skew = 0.0

    # Quadrant energy asymmetry (directional-artifact probe).
    mags_full = np.abs(half)
    q1 = float(mags_full[: hh // 2, : ww // 2].sum())
    q2 = float(mags_full[hh // 2 :, ww // 2 :].sum())
    asym = (q1 - q2) / (q1 + q2 + _EPS)

    return {
        "bicoherence": round(bicoh, 6),
        "bispectrum_kurtosis": round(kurt, 4),
        "phase_coupling_entropy": round(entropy, 4),
        "bicoherence_skew": round(skew, 4),
        "spectral_asymmetry": round(asym, 6),
    }


def extract_hos_features(gray: np.ndarray, settings: Settings) -> HOSFeatures:
    """Extract higher-order spectral (bispectrum) features.

    Args:
        gray: HxW float32 in [0, 1].
        settings: ``image_size`` scales the deterministic sample cap.

    Returns:
        HOSFeatures.
    """
    cap = int(np.clip(settings.image_size * 3, 200, 600))
    snapshot = _phase_coupling_snapshot(np.asarray(gray, dtype=np.float32), max_samples=cap)
    return HOSFeatures(
        bicoherence=snapshot["bicoherence"],
        bispectrum_kurtosis=snapshot["bispectrum_kurtosis"],
        phase_coupling_entropy=snapshot["phase_coupling_entropy"],
        bicoherence_skew=snapshot["bicoherence_skew"],
        spectral_asymmetry=snapshot["spectral_asymmetry"],
    )
