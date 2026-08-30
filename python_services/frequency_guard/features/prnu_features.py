"""PRNU-style camera fingerprint features (E-Masterplan E1 #4).

Real sensors leave a Photo Response Non-Uniformity (PRNU) fingerprint: a
weak, spatially-consistent multiplicative noise pattern. Generated images
lack it. Full PRNU verification requires a reference fingerprint per camera;
here we extract *intra-image* PRNU proxies — the spatial consistency and
scale-invariance of the multiplicative residual — which are strong
authentic-side evidence.

The residual is computed as ``I - F(I)`` where F is a denoising filter
(Wiener-style in the wavelet domain, simplified to a Gaussian/Median hybrid
for CPU speed). We then measure how consistently this residual correlates
across image quadrants and scales.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import Settings

_EPS = 1e-12


@dataclass(frozen=True)
class PRNUFeatures:
    """Intra-image PRNU-proxy statistics (authentic-side evidence)."""

    residual_corr: float  # quadrant residual correlation (consistency)
    residual_energy: float  # mean |residual| relative to signal
    residual_kurtosis: float
    scale_invariance: float  # correlation of residuals across 2 scales
    fingerprint_strength: float  # std of normalized residual pattern

    def as_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.residual_corr,
                self.residual_energy,
                self.residual_kurtosis,
                self.scale_invariance,
                self.fingerprint_strength,
            ],
            dtype=np.float64,
        )


def _denoise(channel: np.ndarray) -> np.ndarray:
    """Lightweight denoiser: Gaussian blur (sigma 1) + median hybrid."""
    blur = cv2.GaussianBlur(channel, (5, 5), 1.0)
    med = cv2.medianBlur(channel, 3)
    return (blur * 0.5 + med * 0.5).astype(np.float32)


def _residual(channel: np.ndarray) -> np.ndarray:
    """Multiplicative residual: (I - F(I)) / (I + eps)."""
    denoised = _denoise(channel)
    return (channel - denoised) / (channel + _EPS)


def _quadrant_correlation(residual: np.ndarray) -> float:
    """Mean pairwise correlation between the four quadrant residuals."""
    h, w = residual.shape
    th, tw = (h // 2) * 2, (w // 2) * 2
    r = residual[:th, :tw]
    quads = [
        r[: th // 2, : tw // 2].ravel(),
        r[: th // 2, tw // 2 :].ravel(),
        r[th // 2 :, : tw // 2].ravel(),
        r[th // 2 :, tw // 2 :].ravel(),
    ]
    # Downsample each quadrant to a common size for correlation.
    target = 32 * 32
    reduced = []
    for q in quads:
        if q.size == 0:
            return 0.0
        idx = np.linspace(0, q.size - 1, target).astype(np.int64)
        reduced.append(q[idx])
    stacked = np.stack(reduced)
    corr = np.corrcoef(stacked)
    upper = corr[np.triu_indices(4, k=1)]
    return float(np.clip(np.mean(upper), -1.0, 1.0))


def _scale_correlation(channel: np.ndarray) -> float:
    """Correlation between residuals computed at two scales."""
    r1 = _residual(channel)
    small = cv2.resize(channel, (channel.shape[1] // 2, channel.shape[0] // 2), interpolation=cv2.INTER_AREA)
    r2 = _residual(small)
    r2_up = cv2.resize(r2, (r1.shape[1], r1.shape[0]), interpolation=cv2.INTER_LINEAR)
    a, b = r1.ravel(), r2_up.ravel()
    if a.std() < _EPS or b.std() < _EPS:
        return 0.0
    return float(np.clip(np.corrcoef(a, b)[0, 1], -1.0, 1.0))


def extract_prnu_features(gray: np.ndarray, settings: Settings) -> PRNUFeatures:
    """Extract intra-image PRNU-proxy features.

    Args:
        gray: HxW float32 in [0, 1].
        settings: unused beyond API consistency.

    Returns:
        PRNUFeatures.
    """
    channel = np.asarray(gray, dtype=np.float32)
    residual = _residual(channel)

    corr = _quadrant_correlation(residual)
    energy = float(np.mean(np.abs(residual)))
    # float64 for moment math: residual values can be extreme where the
    # signal is near zero (I + eps), and float32 ** 4 overflows to inf.
    flat = residual.astype(np.float64).ravel()
    kurt = float(((flat - flat.mean()) ** 4).mean() / (flat.std() ** 4 + _EPS)) if flat.std() > _EPS else 0.0
    if not np.isfinite(kurt):
        kurt = 0.0
    scale_corr = _scale_correlation(channel)
    strength = float(np.std(residual / (np.abs(residual).mean() + _EPS)))

    return PRNUFeatures(
        residual_corr=round(corr, 4),
        residual_energy=round(energy, 6),
        residual_kurtosis=round(kurt, 4),
        scale_invariance=round(scale_corr, 4),
        fingerprint_strength=round(strength, 4),
    )
