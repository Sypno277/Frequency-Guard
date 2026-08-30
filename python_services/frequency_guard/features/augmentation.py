"""Degradation augmentation + test-time augmentation (E-Masterplan E3).

v2 only *tested* robustness post-hoc (JPEG/resize/crop perturbations in the
test suite); v3 *trains* for it. This module provides:

- ``augment_bgr``: random JPEG/resize/blur/noise/crop degradations applied
  during training so the model sees degraded images and stays accurate on
  them (E3.1).
- ``tta_variants``: deterministic multi-crop/multi-scale variants of one
  image whose predictions are averaged at inference time for the opt-in
  high-scrutiny mode (E3.2).

All ops are cv2/numpy, CPU-only, and deterministic under a seed.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import Settings


@dataclass(frozen=True)
class AugmentConfig:
    """Ranges for the degradation pipeline (E3.1)."""

    jpeg_quality_range: tuple[int, int] = (30, 95)
    resize_scale_range: tuple[float, float] = (0.5, 2.0)
    blur_sigma_range: tuple[float, float] = (0.0, 1.5)
    noise_std_range: tuple[float, float] = (0.0, 0.02)
    crop_fraction_range: tuple[float, float] = (0.7, 1.0)
    random_state: int = 42


def _jpeg_degrade(bgr: np.ndarray, quality: int) -> np.ndarray:
    """Re-encode + decode the image as JPEG at ``quality``."""
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return bgr
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else bgr


def _resize_cycle(bgr: np.ndarray, scale: float) -> np.ndarray:
    """Down/up-scale round trip (simulates social-media re-upload)."""
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def augment_bgr(
    bgr: np.ndarray,
    settings: Settings,
    config: AugmentConfig | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Apply one random degradation chain (E3.1 training-time augmentation).

    Args:
        bgr: decoded uint8 BGR image.
        settings: provides the working image size.
        config: augmentation ranges (defaults to :class:`AugmentConfig`).
        seed: per-sample RNG seed. When supplied, the degradation chain is
            *deterministic* for a given seed (so training emits reproducible
            augmented variants keyed by image+fold). When omitted, falls back
            to ``config.random_state`` (preserving the historical fixed-seed
            behavior used by tests).

    Returns:
        Degraded uint8 BGR image of the same dimensions.
    """
    cfg = config or AugmentConfig()
    rng_seed = cfg.random_state if seed is None else seed
    rng = np.random.default_rng(rng_seed)
    out = bgr

    if rng.random() < 0.7:
        q_lo, q_hi = cfg.jpeg_quality_range
        out = _jpeg_degrade(out, int(rng.integers(q_lo, q_hi + 1)))

    if rng.random() < 0.5:
        s_lo, s_hi = cfg.resize_scale_range
        out = _resize_cycle(out, float(rng.uniform(s_lo, s_hi)))

    if rng.random() < 0.4:
        b_lo, b_hi = cfg.blur_sigma_range
        sigma = float(rng.uniform(b_lo, b_hi))
        if sigma > 0.05:
            out = cv2.GaussianBlur(out, (0, 0), sigma)

    if rng.random() < 0.4:
        n_lo, n_hi = cfg.noise_std_range
        std = float(rng.uniform(n_lo, n_hi))
        if std > 0.0005:
            noise = rng.normal(0.0, std, out.shape).astype(np.float32)
            out = np.clip(out.astype(np.float32) + noise * 255.0, 0, 255).astype(np.uint8)

    if rng.random() < 0.4:
        c_lo, c_hi = cfg.crop_fraction_range
        frac = float(rng.uniform(c_lo, c_hi))
        h, w = out.shape[:2]
        ch, cw = max(8, int(h * frac)), max(8, int(w * frac))
        y0 = int(rng.integers(0, h - ch + 1))
        x0 = int(rng.integers(0, w - cw + 1))
        out = out[y0 : y0 + ch, x0 : x0 + cw]
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)

    return out


def tta_variants(bgr: np.ndarray, settings: Settings) -> Iterator[np.ndarray]:
    """Deterministic TTA variants (E3.2): full + 2 scales + 2 off-center crops.

    Yields 5 variants; the caller preprocesses + extracts features for each
    and averages the predictions. Only used by the opt-in ``?tta=true`` mode.
    """
    h, w = bgr.shape[:2]
    yield bgr  # full image

    # Multi-scale resizes (0.75x, 1.25x round-trips).
    for scale in (0.75, 1.25):
        yield _resize_cycle(bgr, scale)

    # Off-center crops at 85% area: top-left and bottom-right biases.
    frac = 0.85
    ch, cw = max(8, int(h * frac)), max(8, int(w * frac))
    yield bgr[0:ch, 0:cw]
    yield bgr[h - ch : h, w - cw : w]
