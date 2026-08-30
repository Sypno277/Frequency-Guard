"""Shared pytest fixtures: synthetic images and a fast test settings object.

Synthetic fixtures cover real-ish and grid/checkerboard-like images so
extractor unit tests have deterministic inputs that exercise distinct
spectral signatures without needing an external dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from python_services.frequency_guard.config import Settings


@pytest.fixture(scope="session")
def fast_settings() -> Settings:
    """Settings tuned for fast test runs (small image, few bins)."""
    return Settings(
        image_size=128,
        dct_block_size=8,
        wavelet_name="db4",
        wavelet_levels=2,
        radial_bins=12,
        azimuthal_bins=8,
        tiles_per_side=2,
        n_folds=2,
    )


def make_smooth_gradient(size: int = 128) -> np.ndarray:
    """Natural-photo-like smooth gradient (low-frequency dominant)."""
    yy, xx = np.mgrid[0:size, 0:size]
    value = (0.4 + 0.3 * np.sin(xx / 24.0) * np.cos(yy / 19.0)).astype(np.float32)
    value = np.clip(value, 0.0, 1.0)
    return value


def make_checkerboard_upscaled(size: int = 128, block: int = 8) -> np.ndarray:
    """Simulated GAN-upsampled checkerboard (periodic grid artifact)."""
    base = np.zeros((size // block, size // block), dtype=np.float32)
    base[::2, ::2] = 1.0
    base[1::2, 1::2] = 1.0
    up = np.kron(base, np.ones((block, block), dtype=np.float32))
    # mild anti-alias so it is not a perfect binary image
    return up / up.max()


def make_noisy_photo(size: int = 128, noise_std: float = 0.02) -> np.ndarray:
    """Smooth gradient with sensor-like Gaussian noise."""
    grad = make_smooth_gradient(size)
    noise = np.random.default_rng(7).normal(0.0, noise_std, grad.shape).astype(np.float32)
    return np.clip(grad + noise, 0.0, 1.0)


@pytest.fixture()
def smooth_image() -> np.ndarray:
    return make_smooth_gradient()


@pytest.fixture()
def checker_image() -> np.ndarray:
    return make_checkerboard_upscaled()


@pytest.fixture()
def noisy_image() -> np.ndarray:
    return make_noisy_photo()
