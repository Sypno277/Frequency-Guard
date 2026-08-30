"""FFT-based spectral features.

Implements the core frequency statistics that separate camera images from
AI-generated ones:

- **Radial energy profile**: log-spaced annuli over the 2D magnitude
  spectrum. Natural photos follow a 1/f power law (energy decays smoothly
  with radius); GAN upsampling and diffusion denoising distort this decay.
- **Spectral slope / 1/f deviation**: least-squares fit of log-power vs
  log-frequency in the mid-band; cameras sit in a tight α≈2.0 band.
- **Spectral flatness**: geometric/arithmetic mean ratio of power —
  generator noise floors flatten the spectrum.
- **Phase entropy**: high-frequency phase of natural images is near-random;
  generation introduces phase correlation.
- **Azimuthal energy profile**: angular energy distribution exposes
  directional / checkerboard artifacts invisible to radial-only analysis.

All computations use `scipy.fft.rfft2` on the luma channel, DC removed, and
avoid returning the full complex spectrum (memory-light).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import rfft2
from scipy.stats import entropy as scipy_entropy

from ..config import Settings


@dataclass(frozen=True)
class FFTFeatures:
    """Numerical FFT feature vector and the profiles used for display."""

    radial_profile: np.ndarray  # (n_radial_bins,) mean magnitude per annulus
    radial_norm: np.ndarray  # (n_radial_bins,) radial_profile / total energy
    azimuthal_profile: np.ndarray  # (n_azimuthal_bins,) mean magnitude per angle
    spectral_slope: float
    spectral_intercept: float
    flatness: float
    phase_entropy: float  # normalized to [0, 1]
    high_freq_energy_ratio: float
    dc_energy_ratio: float
    peak_prominence: float  # strongest spectral peak relative to median
    peak_radius_bin: int  # radial bin index of the strongest peak

    def as_vector(self) -> np.ndarray:
        """Flatten to a fixed-size numpy vector for classification."""
        return np.concatenate(
            (
                self.radial_norm,
                self.azimuthal_profile,
                np.asarray(
                    [
                        self.spectral_slope,
                        self.spectral_intercept,
                        self.flatness,
                        self.phase_entropy,
                        self.high_freq_energy_ratio,
                        self.dc_energy_ratio,
                        self.peak_prominence,
                        float(self.peak_radius_bin),
                    ],
                    dtype=np.float64,
                ),
            )
        )


# Guard against log(0): replace zero-power bins with this epsilon.
_EPS = 1e-12


def _log_spaced_radial_bins(shape: tuple[int, int], n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin_index[H,W], distances[H,W]) for log-spaced radial bins.

    Bins are logarithmically spaced so low frequencies (where most energy
    lives) get finer resolution and the 1/f fit has balanced samples.
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.sqrt(xx.astype(np.float64) ** 2 + yy.astype(np.float64) ** 2)
    max_r = radius.max()
    # log-spacing from r=1 to max_r; avoid log(0) with +1 offset
    edges = np.logspace(0.0, np.log10(max_r + 1.0), n_bins + 1) - 1.0
    edges[0] = 0.0
    bin_idx = np.digitize(radius.ravel(), edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1).reshape(radius.shape)
    return bin_idx, radius


def _spectral_slope(radial_profile: np.ndarray, radius_centers: np.ndarray) -> tuple[float, float]:
    """Fit log(P) = a + b*log(r) over the mid-band of the radial profile.

    The fit uses radii 3..max_r*0.7 (skip DC/very-low and the noisy extreme
    high-frequency rim). Returns (slope, intercept).
    """
    r = radius_centers
    mask = (r > 2.0) & (r < r.max() * 0.85)
    if mask.sum() < 5:
        return 0.0, 0.0
    x = np.log(r[mask])
    y = np.log(radial_profile[mask] + _EPS)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _azimuthal_energy(magnitude: np.ndarray, radius: np.ndarray, n_bins: int) -> np.ndarray:
    """Average magnitude over angular sectors of the spectrum.

    Only the annulus between 10% and 90% of the max radius is included;
    the DC corner and high-frequency rim dominate otherwise and drown the
    directional signature. Vectorized with a single ``np.bincount`` pass.
    """
    h, w = magnitude.shape
    yy, xx = np.mgrid[0:h, 0:w]
    angle = np.arctan2(yy.astype(np.float64), xx.astype(np.float64))  # in [-pi, pi]
    rmax = radius.max()
    band = (radius > rmax * 0.1) & (radius < rmax * 0.9)
    angle_bin = np.floor((angle + np.pi) / (2.0 * np.pi) * n_bins).astype(np.int64) % n_bins

    flat_angle = angle_bin[band]
    flat_mag = magnitude[band]
    counts = np.bincount(flat_angle, minlength=n_bins).astype(np.float64)
    counts[counts == 0] = 1.0
    summed = np.bincount(flat_angle, weights=flat_mag, minlength=n_bins)
    return summed / counts


def extract_fft_features(gray: np.ndarray, settings: Settings) -> FFTFeatures:
    """Extract FFT features from a normalized grayscale image.

    Args:
        gray: HxW float32 in [0, 1].
        settings: configuration (radial_bins, azimuthal_bins).

    Returns:
        FFTFeatures with both raw profiles and scalar statistics.
    """
    # One FFT per image — reuse the complex spectrum for both magnitude and
    # phase entropy (avoids a redundant ~10ms transform on 256px inputs).
    spectrum = rfft2(gray - gray.mean())
    magnitude = np.abs(spectrum)
    total_energy = float(np.sum(magnitude**2)) + _EPS

    bin_idx, radius = _log_spaced_radial_bins(magnitude.shape, settings.radial_bins)

    # Radial profile via bincount (replaces slow np.add.at scatter-add).
    flat_bin = bin_idx.ravel()
    flat_mag = magnitude.ravel()
    counts = np.bincount(flat_bin, minlength=settings.radial_bins).astype(np.float64)
    counts[counts == 0] = 1.0
    radial_profile = np.bincount(flat_bin, weights=flat_mag, minlength=settings.radial_bins) / counts

    radial_norm = radial_profile / (radial_profile.sum() + _EPS)

    # radius centers per bin for the slope fit (mean radius of members)
    radius_centers = np.bincount(flat_bin, weights=radius.ravel(), minlength=settings.radial_bins) / counts

    slope, intercept = _spectral_slope(radial_profile, radius_centers)

    # flatness: G/P over mid-band (r>2) — geometric < arithmetic for peaky spectra
    mid_bins = radius_centers > 2.0
    mid_mask = radius > 2.0
    mid_power = magnitude**2
    if mid_bins.sum() > 10 and mid_mask.sum() > 0:
        geo = np.exp(np.mean(np.log(mid_power[mid_mask] + _EPS)))
        arith = np.mean(mid_power[mid_mask])
        flatness = float(geo / (arith + _EPS))
    else:
        flatness = 0.0

    # phase entropy from the same FFT (real/imag → phase histogram)
    phase = np.angle(spectrum)
    phase_sel = phase[mid_mask]
    hist, _ = np.histogram(phase_sel, bins=32, range=(-np.pi, np.pi))
    hist = hist.astype(np.float64)
    hist = hist[hist > 0]
    phase_entropy = float(scipy_entropy(hist / hist.sum(), base=2) / np.log2(32)) if hist.size else 0.0

    nyquist = min(magnitude.shape[0], magnitude.shape[1]) // 2
    high_energy = float(np.sum(magnitude[-nyquist:, -nyquist:] ** 2)) if nyquist > 2 else 0.0
    dc_energy = float(magnitude[0, 0] ** 2)

    # peak prominence: strongest non-DC peak relative to the radial median
    non_dc = magnitude[1:, 1:].copy()
    if non_dc.size:
        is_peak = (
            (non_dc > np.roll(non_dc, 1, axis=0))
            & (non_dc > np.roll(non_dc, -1, axis=0))
            & (non_dc > np.roll(non_dc, 1, axis=1))
            & (non_dc > np.roll(non_dc, -1, axis=1))
        )
        peak_vals = non_dc[is_peak]
        median = float(np.median(non_dc))
        peak_prominence = float(peak_vals.max() / (median + _EPS)) if peak_vals.size else 0.0
        peak_loc = np.unravel_index(int(non_dc.argmax()), non_dc.shape)
        peak_radius_bin = int(bin_idx[peak_loc[0] + 1, peak_loc[1] + 1])
    else:
        peak_prominence = 0.0
        peak_radius_bin = 0

    return FFTFeatures(
        radial_profile=radial_profile,
        radial_norm=radial_norm,
        azimuthal_profile=_azimuthal_energy(magnitude, radius, settings.azimuthal_bins),
        spectral_slope=slope,
        spectral_intercept=intercept,
        flatness=flatness,
        phase_entropy=phase_entropy,
        high_freq_energy_ratio=high_energy / total_energy,
        dc_energy_ratio=dc_energy / total_energy,
        peak_prominence=peak_prominence,
        peak_radius_bin=peak_radius_bin,
    )
