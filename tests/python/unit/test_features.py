"""Unit tests for all frequency-domain feature extractors and the
orchestrator/cache."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from python_services.frequency_guard.features.dct_features import extract_dct_features
from python_services.frequency_guard.features.extractor import (
    FeatureBundle,
    FeatureCache,
    cache_key,
    extract_features,
    settings_signature,
)
from python_services.frequency_guard.features.fft_features import extract_fft_features
from python_services.frequency_guard.features.noise_features import extract_noise_features
from python_services.frequency_guard.features.texture_features import extract_texture_features
from python_services.frequency_guard.features.wavelet_features import extract_wavelet_features
from python_services.frequency_guard.preprocess import preprocess
from tests.python.conftest import (
    make_checkerboard_upscaled,
    make_noisy_photo,
    make_smooth_gradient,
)


def _bgr_from_gray(gray: np.ndarray) -> np.ndarray:
    """Convert a float32 [0,1] gray image into a uint8 BGR triple."""
    g = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


class TestFFTFeatures:
    def test_vector_shapes(self, fast_settings) -> None:
        gray = make_smooth_gradient(fast_settings.image_size)
        feats = extract_fft_features(gray, fast_settings)
        assert len(feats.radial_profile) == fast_settings.radial_bins
        assert len(feats.azimuthal_profile) == fast_settings.azimuthal_bins
        expected = fast_settings.radial_bins + fast_settings.azimuthal_bins + 8
        assert len(feats.as_vector()) == expected

    def test_radial_profile_nonnegative(self, fast_settings) -> None:
        gray = make_noisy_photo(fast_settings.image_size)
        feats = extract_fft_features(gray, fast_settings)
        assert np.all(feats.radial_profile >= 0)
        assert np.all(feats.azimuthal_profile >= 0)

    def test_slope_is_negative_for_natural_like(self, fast_settings) -> None:
        """A smooth photo-like gradient must yield a negative spectral slope."""
        gray = make_noisy_photo(fast_settings.image_size)
        feats = extract_fft_features(gray, fast_settings)
        assert feats.spectral_slope < -0.5

    def test_deterministic(self, fast_settings) -> None:
        gray = make_checkerboard_upscaled(fast_settings.image_size)
        first = extract_fft_features(gray, fast_settings)
        second = extract_fft_features(gray, fast_settings)
        assert np.array_equal(first.radial_profile, second.radial_profile)


class TestDCTFeatures:
    def test_vector(self, fast_settings) -> None:
        gray = make_smooth_gradient(fast_settings.image_size)
        feats = extract_dct_features(gray, fast_settings)
        assert len(feats.as_vector()) == 7

    def test_ratios_in_unit_interval(self, fast_settings) -> None:
        gray = make_checkerboard_upscaled(fast_settings.image_size)
        feats = extract_dct_features(gray, fast_settings)
        assert 0.0 <= feats.high_freq_energy_ratio <= 1.0
        assert 0.0 <= feats.mid_band_energy_ratio <= 1.0
        assert 0.0 <= feats.block_boundary_energy <= 1.0

    def test_low_high_ratio_positive(self, fast_settings) -> None:
        gray = make_smooth_gradient(fast_settings.image_size)
        feats = extract_dct_features(gray, fast_settings)
        assert feats.low_high_ratio > 0


class TestWaveletFeatures:
    def test_vector_shape(self, fast_settings) -> None:
        gray = make_smooth_gradient(fast_settings.image_size)
        feats = extract_wavelet_features(gray, fast_settings)
        assert len(feats.subband_energies) == fast_settings.wavelet_levels * 3
        assert len(feats.subband_entropies) == fast_settings.wavelet_levels * 3

    def test_energies_nonnegative(self, fast_settings) -> None:
        gray = make_noisy_photo(fast_settings.image_size)
        feats = extract_wavelet_features(gray, fast_settings)
        assert np.all(feats.subband_energies >= 0)
        assert np.all(feats.subband_entropies >= 0)

    def test_detail_energy_ratio(self, fast_settings) -> None:
        gray = make_smooth_gradient(fast_settings.image_size)
        feats = extract_wavelet_features(gray, fast_settings)
        assert 0.0 <= feats.detail_energy_ratio <= 1.0

    def test_rejects_non_orthogonal_wavelet(self, fast_settings) -> None:
        bad_settings = replace(fast_settings, wavelet_name="morl")
        with pytest.raises(ValueError):
            extract_wavelet_features(make_smooth_gradient(64), bad_settings)


class TestNoiseFeatures:
    def test_vector(self, fast_settings) -> None:
        gray = make_noisy_photo(fast_settings.image_size)
        feats = extract_noise_features(gray, fast_settings)
        assert len(feats.as_vector()) == 5

    def test_consistency_bounded(self, fast_settings) -> None:
        gray = make_noisy_photo(fast_settings.image_size, noise_std=0.05)
        feats = extract_noise_features(gray, fast_settings)
        assert 0.0 <= feats.block_consistency <= 1.0


class TestTextureFeatures:
    def test_vector(self, fast_settings) -> None:
        gray = make_noisy_photo(fast_settings.image_size)
        feats = extract_texture_features(gray, fast_settings)
        assert len(feats.as_vector()) == 6

    def test_fractal_dimension_textured_exceeds_smooth(self, fast_settings) -> None:
        """Textured images must measure higher edge complexity than flat ones.

        A near-edge-free gradient has box-count dimension ≈ 0; adding
        sensor-like noise creates real edges and raises it measurably.
        """
        smooth = extract_texture_features(make_smooth_gradient(fast_settings.image_size), fast_settings)
        noisy = extract_texture_features(make_noisy_photo(fast_settings.image_size), fast_settings)
        assert np.isfinite(smooth.fractal_dimension)
        assert noisy.fractal_dimension > smooth.fractal_dimension


class TestExtractorOrchestrator:
    def test_features_vector_length_is_stable(self, fast_settings) -> None:
        bgr = _bgr_from_gray(make_noisy_photo(fast_settings.image_size))
        prep = preprocess(bgr, fast_settings)
        bundle = extract_features(prep, fast_settings)
        assert isinstance(bundle, FeatureBundle)
        assert bundle.vector.ndim == 1
        assert len(bundle.vector) > 50

    def test_checkerboard_vs_smooth_differ(self, fast_settings) -> None:
        """Synthetic signatures must produce measurably different vectors."""
        smooth_bgr = _bgr_from_gray(make_smooth_gradient(fast_settings.image_size))
        check_bgr = _bgr_from_gray(make_checkerboard_upscaled(fast_settings.image_size))
        smooth = extract_features(preprocess(smooth_bgr, fast_settings), fast_settings)
        check = extract_features(preprocess(check_bgr, fast_settings), fast_settings)
        assert np.linalg.norm(smooth.vector - check.vector) > 1.0


class TestFeatureCache:
    def test_caches_and_reuses(self, fast_settings) -> None:
        cache = FeatureCache(maxsize=4)
        bgr = _bgr_from_gray(make_smooth_gradient(fast_settings.image_size))
        prep = preprocess(bgr, fast_settings)
        key = cache_key("img.png", prep, fast_settings)

        calls = {"n": 0}

        def compute() -> FeatureBundle:
            calls["n"] += 1
            return extract_features(prep, fast_settings)

        first = cache.get_or_compute(key, compute)
        second = cache.get_or_compute(key, compute)
        assert calls["n"] == 1
        assert first is second

    def test_eviction_bounds_size(self, fast_settings) -> None:
        cache = FeatureCache(maxsize=2)
        bgr = _bgr_from_gray(make_smooth_gradient(fast_settings.image_size))
        prep = preprocess(bgr, fast_settings)
        sig = settings_signature(fast_settings)

        def compute() -> FeatureBundle:
            return extract_features(prep, fast_settings)

        for idx in range(5):
            key = (f"img_{idx}.png", prep.orig_size[0], prep.orig_size[1], sig)
            cache.get_or_compute(key, compute)
        assert len(cache) <= 2

    def test_clear(self) -> None:
        cache = FeatureCache(maxsize=2)
        cache.clear()
        assert len(cache) == 0
