"""Feature extraction orchestrator.

Runs all frequency-domain + spatial extractors in a single pipeline pass
over a preprocessed image and assembles one fixed-size feature vector.
Results are cached with a bounded, thread-safe cache keyed by (source,
width, height, settings signature).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..config import Settings
from ..preprocess import PreprocessedImage
from .chroma_features import ChromaFeatures, extract_chroma_features
from .dct_features import DCTFeatures, extract_dct_features
from .fft_features import FFTFeatures, extract_fft_features
from .freq_texture_features import FreqTextureFeatures, extract_freq_texture_features
from .hos_features import HOSFeatures, extract_hos_features
from .jpeg_features import JPEGFeatures, extract_jpeg_features
from .noise_features import NoiseFeatures, extract_noise_features
from .prnu_features import PRNUFeatures, extract_prnu_features
from .texture_features import TextureFeatures, extract_texture_features
from .wavelet_features import WaveletFeatures, extract_wavelet_features

CacheKey = tuple[str, int, int, str]


@dataclass(frozen=True)
class FeatureBundle:
    """All feature groups plus the merged classifier-ready vector."""

    fft: FFTFeatures
    dct: DCTFeatures
    wavelet: WaveletFeatures
    noise: NoiseFeatures
    texture: TextureFeatures
    chroma: ChromaFeatures
    hos: HOSFeatures
    jpeg: JPEGFeatures
    prnu: PRNUFeatures
    freq_texture: FreqTextureFeatures
    vector: np.ndarray

    @property
    def names(self) -> tuple[str, ...]:
        """Stable feature names, length must equal ``len(self.vector)``."""
        n_radial = len(self.fft.radial_norm)
        n_azimuthal = len(self.fft.azimuthal_profile)
        n_wave_energy = len(self.wavelet.subband_energies)
        n_wave_entropy = len(self.wavelet.subband_entropies)
        return (
            *(f"fft_radial_{i}" for i in range(n_radial)),
            *(f"fft_azimuthal_{i}" for i in range(n_azimuthal)),
            "fft_slope",
            "fft_intercept",
            "fft_flatness",
            "fft_phase_entropy",
            "fft_high_freq_ratio",
            "fft_dc_ratio",
            "fft_peak_prominence",
            "fft_peak_radius_bin",
            "dct_kurtosis",
            "dct_high_freq_ratio",
            "dct_mid_band_ratio",
            "dct_boundary_energy",
            "dct_std",
            "dct_avg_dc",
            "dct_low_high_ratio",
            *(f"wavelet_energy_{i}" for i in range(n_wave_energy)),
            *(f"wavelet_entropy_{i}" for i in range(n_wave_entropy)),
            "wavelet_detail_ratio",
            "wavelet_cross_scale_ratio",
            "wavelet_ll_ratio",
            "noise_kurtosis",
            "noise_std",
            "noise_mean_abs",
            "noise_block_consistency",
            "noise_flatness",
            "texture_contrast",
            "texture_correlation",
            "texture_energy",
            "texture_homogeneity",
            "texture_dissimilarity",
            "texture_fractal_dim",
            *(
                f"chroma_{n}"
                for n in (
                    "flatness",
                    "high_freq_ratio",
                    "slope",
                    "cb_dc_ratio",
                    "cr_dc_ratio",
                    "cb_cr_corr",
                    "rg_corr",
                    "gb_corr",
                    "entropy",
                    "radial_kurtosis",
                )
            ),
            "hos_bicoherence",
            "hos_bispectrum_kurtosis",
            "hos_phase_coupling_entropy",
            "hos_bicoherence_skew",
            "hos_spectral_asymmetry",
            "jpeg_quant_estimate",
            "jpeg_hist_kurtosis",
            "jpeg_hist_periodicity",
            "jpeg_zero_ratio",
            "jpeg_mode_spread",
            "prnu_residual_corr",
            "prnu_residual_energy",
            "prnu_residual_kurtosis",
            "prnu_scale_invariance",
            "prnu_fingerprint_strength",
            "freqtex_lbp_uniformity",
            "freqtex_lbp_entropy",
            "freqtex_lbp_contrast",
            "freqtex_dct_cooc_homogeneity",
            "freqtex_dct_cooc_contrast",
            "freqtex_grad_consistency",
        )


def settings_signature(settings: Settings) -> str:
    """Stable string of the settings values that affect features."""
    return ",".join(
        [
            str(settings.image_size),
            str(settings.dct_block_size),
            settings.wavelet_name,
            str(settings.wavelet_levels),
            str(settings.radial_bins),
            str(settings.azimuthal_bins),
        ]
    )


def cache_key(source: str, preprocessed: PreprocessedImage, settings: Settings) -> CacheKey:
    """Compute the cache key for a preprocessed image."""
    return (
        source,
        preprocessed.orig_size[0],
        preprocessed.orig_size[1],
        settings_signature(settings),
    )


def extract_features(preprocessed: PreprocessedImage, settings: Settings) -> FeatureBundle:
    """Extract the full feature bundle from a preprocessed image."""
    fft = extract_fft_features(preprocessed.y_channel, settings)
    dct = extract_dct_features(preprocessed.y_channel, settings)
    wavelet = extract_wavelet_features(preprocessed.y_channel, settings)
    noise = extract_noise_features(preprocessed.y_channel, settings)
    texture = extract_texture_features(preprocessed.gray, settings)
    chroma = extract_chroma_features(preprocessed, settings)
    hos = extract_hos_features(preprocessed.y_channel, settings)
    jpeg = extract_jpeg_features(preprocessed.y_channel, settings)
    prnu = extract_prnu_features(preprocessed.y_channel, settings)
    freq_texture = extract_freq_texture_features(preprocessed.y_channel, settings)

    vector = np.concatenate(
        (
            fft.as_vector(),
            dct.as_vector(),
            wavelet.as_vector(),
            noise.as_vector(),
            texture.as_vector(),
            chroma.as_vector(),
            hos.as_vector(),
            jpeg.as_vector(),
            prnu.as_vector(),
            freq_texture.as_vector(),
        )
    ).astype(np.float64)
    # Classifiers (esp. sklearn RandomForest) reject NaN/inf inputs. Any
    # extractor that divides by a near-zero signal may emit them; replacing
    # non-finite entries with 0 is the safe, neutral fallback (E-Masterplan
    # E1 hardening).
    if not np.all(np.isfinite(vector)):
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

    return FeatureBundle(
        fft=fft,
        dct=dct,
        wavelet=wavelet,
        noise=noise,
        texture=texture,
        chroma=chroma,
        hos=hos,
        jpeg=jpeg,
        prnu=prnu,
        freq_texture=freq_texture,
        vector=vector,
    )


class FeatureCache:
    """Bounded, thread-safe cache for extracted feature vectors."""

    def __init__(self, maxsize: int = 256) -> None:
        self._maxsize = max(1, maxsize)
        self._store: dict[CacheKey, FeatureBundle] = {}
        self._order: list[CacheKey] = []
        self._lock = threading.Lock()

    def get_or_compute(self, key: CacheKey, compute: Callable[[], FeatureBundle]) -> FeatureBundle:
        """Return the cached bundle for ``key``, computing if absent."""
        with self._lock:
            cached = self._store.get(key)
            if cached is not None:
                return cached
        bundle = compute()
        with self._lock:
            if key in self._store:
                return self._store[key]
            if len(self._store) >= self._maxsize:
                evicted = self._order.pop(0)
                self._store.pop(evicted, None)
            self._store[key] = bundle
            self._order.append(key)
        return bundle

    def clear(self) -> None:
        """Drop all cached entries."""
        with self._lock:
            self._store.clear()
            self._order.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# Default cache used by the API/training entry points.
global_cache = FeatureCache()
