"""Preprocessing: resize, color-space conversion, and normalization.

All operations are vectorized numpy/OpenCV. The pipeline is designed so the
same code path serves both training-time degradation robustness and
inference-time standardization.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Settings


@dataclass(frozen=True)
class PreprocessedImage:
    """Standardized image tensors for feature extraction."""

    gray: np.ndarray  # HxW float32, range [0, 1], `image_size` square
    rgb: np.ndarray  # HxWx3 float32, range [0, 1], `image_size` square
    y_channel: np.ndarray  # HxW float32 Y (YCbCr), range [0, 1]
    orig_size: tuple[int, int]  # (height, width) before resize


def to_gray(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 to float32 grayscale in [0, 1]."""
    if bgr.ndim == 2:
        return bgr.astype(np.float32) / 255.0
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def to_rgb_float(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 to float32 RGB in [0, 1]."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def to_y_channel(bgr: np.ndarray) -> np.ndarray:
    """Extract the luma (Y) channel from BGR in [0, 1]."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32) / 255.0


def resize_square(img: np.ndarray, size: int) -> np.ndarray:
    """Resize preserving aspect ratio with letterboxing (no distortion).

    Real photos and generated images rarely share the same aspect ratio;
    stretching distorts spectral statistics. Letterboxing adds a neutral
    border that is crop-invariant.
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size) + img.shape[2:], dtype=img.dtype)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def preprocess(bgr: np.ndarray, settings: Settings) -> PreprocessedImage:
    """Run the standard preprocessing chain on a BGR image.

    Args:
        bgr: decoded image (HxWx3 uint8, BGR).
        settings: runtime configuration.

    Returns:
        Standardized tensors plus the original dimensions.
    """
    orig_size = (bgr.shape[0], bgr.shape[1])
    square = resize_square(bgr, settings.image_size)
    return PreprocessedImage(
        gray=to_gray(square),
        rgb=to_rgb_float(square),
        y_channel=to_y_channel(square),
        orig_size=orig_size,
    )
