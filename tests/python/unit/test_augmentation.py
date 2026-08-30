"""Unit tests for degradation augmentation (E3.1) and the per-sample seed."""

from __future__ import annotations

import numpy as np
import pytest

from python_services.frequency_guard.config import Settings
from python_services.frequency_guard.features.augmentation import AugmentConfig, augment_bgr


def _bgr(size: int = 64, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    return img


class TestAugmentBgr:
    def test_deterministic_with_seed(self) -> None:
        """Same seed must produce identical degradation."""
        img = _bgr()
        a = augment_bgr(img, Settings(), seed=1234)
        b = augment_bgr(img, Settings(), seed=1234)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_generally_differs(self) -> None:
        """Different seeds usually produce different degradations."""
        img = _bgr()
        a = augment_bgr(img, Settings(), seed=111)
        b = augment_bgr(img, Settings(), seed=222)
        assert not np.array_equal(a, b)

    def test_preserves_dimensions(self) -> None:
        """Augmentation must return an image of the same HxWx3 shape."""
        img = _bgr(size=80, seed=7)
        out = augment_bgr(img, Settings(), seed=99)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_pristine_when_all_ops_disabled(self) -> None:
        """A config that disables every branch returns the exact input."""
        cfg = AugmentConfig(
            jpeg_quality_range=(100, 100),
            resize_scale_range=(1.0, 1.0),
            blur_sigma_range=(0.0, 0.0),
            noise_std_range=(0.0, 0.0),
            crop_fraction_range=(1.0, 1.0),
        )
        img = _bgr()
        out = augment_bgr(img, Settings(), config=cfg, seed=5)
        np.testing.assert_array_equal(out, img)
