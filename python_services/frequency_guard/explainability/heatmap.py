"""Explainability heatmaps (Masterplan §3.1 POST-PROCESSING, §5.1).

Replaces GradCAM-style backprop explainers with a gradient-free,
CPU-cheap alternative purpose-built for frequency-domain detection:

1. **Spectral-saliency heatmap** — the image is tiled; each tile's local
   FFT deviation from the natural 1/f law (|slope - 2.0|) is measured.
   Tiles whose spectra deviate most from camera-like statistics are where
   synthetic content concentrates.

2. **Patch-inconsistency map** — inter-tile variance of spectral slope.
   Real sensor images are spatially consistent; generators often
   synthesize patches with slightly different spectral character.

Maps are normalized to [0, 1] and encoded as base64 PNG overlays with a
viridis-like colormap for direct compositing onto the original image.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import numpy as np

_EPS = 1e-12

# Natural photography follows power-law decay alpha ~ 1.9-2.2 (Masterplan §4.1).
NATURAL_SLOPE_ALPHA = 2.0


@dataclass(frozen=True)
class ExplainabilityMaps:
    """Per-image explainability artifacts."""

    saliency: np.ndarray  # (tps,tps) float32 in [0,1] — tile 1/f deviation
    inconsistency: np.ndarray  # (tps,tps) float32 in [0,1] — inter-tile drift
    overlay_png_base64: str  # RGBA PNG data URL of the saliency heatmap
    mean_deviation: float
    patch_inconsistency_score: float


def _tile_grid(shape: tuple[int, int], tiles_per_side: int) -> list[tuple[int, int]]:
    """Yield (y0, x0) origins for a square tile grid."""
    h, w = shape
    th, tw = h // tiles_per_side, w // tiles_per_side
    return [(y * th, x * tw) for y in range(tiles_per_side) for x in range(tiles_per_side)]


def _local_slope(tile: np.ndarray) -> tuple[float, float]:
    """Fit log-power vs log-frequency slope + flatness on one tile.

    Returns:
        (spectral_slope, spectral_flatness) for the tile.
    """
    spectrum = np.abs(np.fft.rfft2(tile - tile.mean()))
    h, w = spectrum.shape
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.sqrt(xx.astype(np.float64) ** 2 + yy.astype(np.float64) ** 2)
    rmax = radius.max()

    edges = np.logspace(0.0, np.log10(rmax + 1.0), 13) - 1.0
    edges[0] = 0.0
    bin_idx = np.clip(np.digitize(radius.ravel(), edges) - 1, 0, 11)

    profile = np.zeros(12)
    counts = np.zeros(12)
    np.add.at(profile, bin_idx, spectrum.ravel())
    np.add.at(counts, bin_idx, 1)
    counts[counts == 0] = 1
    profile /= counts

    # Radius center per bin for the log-log fit.
    centers = np.zeros(12)
    np.add.at(centers, bin_idx, radius.ravel())
    centers /= counts

    mask = centers > 2.0
    if mask.sum() < 4:
        return NATURAL_SLOPE_ALPHA, 0.0
    slope, _ = np.polyfit(np.log(centers[mask]), np.log(profile[mask] + _EPS), 1)

    # Flatness over the same mid-band.
    power = spectrum.astype(np.float64) ** 2
    band_power = power[radius > 2.0]
    flatness = 0.0
    if band_power.size >= 4:
        geo = np.exp(np.mean(np.log(band_power + _EPS)))
        arith = np.mean(band_power)
        flatness = float(geo / (arith + _EPS))

    return float(slope), flatness


def _colormap_viridis(values: np.ndarray) -> np.ndarray:
    """Map [0,1] floats to RGB via a viridis anchor LUT (no matplotlib)."""
    anchors = np.asarray(
        [
            [68, 1, 84],
            [72, 40, 120],
            [62, 74, 137],
            [49, 104, 142],
            [38, 130, 142],
            [31, 158, 137],
            [53, 183, 121],
            [109, 205, 89],
            [253, 231, 37],
        ],
        dtype=np.float64,
    )
    flat = np.clip(values.ravel(), 0.0, 1.0)
    idx = flat * (len(anchors) - 1)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(anchors) - 1)
    frac = (idx - lo)[:, None]
    rgb = anchors[lo] * (1 - frac) + anchors[hi] * frac
    return rgb.reshape(values.shape[0], values.shape[1], 3)


def _to_heatmap_png(map_small: np.ndarray, size: int, alpha: float = 0.75) -> str:
    """Upscale, colormap, and encode a map as a base64 RGBA PNG data URL.

    The PNG carries the heatmap colors at ``alpha`` opacity so the dashboard
    can layer it directly over the original photo.
    """
    resized = cv2.resize(map_small.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    resized = np.clip(resized, 0.0, 1.0)

    rgb = _colormap_viridis(resized)
    alpha_channel = np.full(resized.shape[:2], int(255 * alpha), dtype=np.uint8)
    rgba = np.dstack([rgb.astype(np.uint8), alpha_channel])

    ok, buf = cv2.imencode(".png", rgba)
    if not ok:
        raise RuntimeError("heatmap PNG encoding failed")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def compute_explainability(
    gray: np.ndarray, tiles_per_side: int = 4, alpha: float = 0.75
) -> ExplainabilityMaps:
    """Compute spectral-saliency + patch-inconsistency maps for an image.

    Args:
        gray: HxW float32 grayscale in [0, 1] (preprocessed luma).
        tiles_per_side: grid resolution of the analysis tiles.
        alpha: heatmap opacity for the PNG overlay.

    Returns:
        ExplainabilityMaps with both maps and a dashboard-ready overlay.
    """
    gray = np.asarray(gray, dtype=np.float64)
    h, w = gray.shape
    tps = max(2, min(tiles_per_side, min(h, w) // 16))
    if tps < 2:
        raise ValueError("image too small for tiled explainability")

    slopes: list[float] = []
    flatnesses: list[float] = []
    th, tw = h // tps, w // tps

    for y0, x0 in _tile_grid(gray.shape, tps):
        tile = gray[y0 : y0 + th, x0 : x0 + tw]
        slope, flatness = _local_slope(tile)
        slopes.append(slope)
        flatnesses.append(flatness)

    slope_arr = np.asarray(slopes, dtype=np.float64)
    flat_arr = np.asarray(flatnesses, dtype=np.float64)

    # --- saliency: per-tile deviation from the natural 1/f law ----------
    deviations = np.abs(slope_arr - NATURAL_SLOPE_ALPHA)
    dev_norm = deviations / (deviations.max() + _EPS)

    # --- inconsistency: dispersion across tiles -------------------------
    slope_std = float(slope_arr.std())
    flat_std = float(flat_arr.std())
    inconsistency_score = float(
        np.clip(slope_std / (abs(float(slope_arr.mean())) + _EPS) + flat_std, 0.0, 1.0)
    )

    # Per-tile inconsistency contribution: distance from the mean slope.
    slope_z = np.abs(slope_arr - slope_arr.mean()) / (slope_std + _EPS)
    inc_norm = slope_z / (slope_z.max() + _EPS)

    saliency_map = dev_norm.reshape(tps, tps).astype(np.float32)
    inconsistency_map = inc_norm.reshape(tps, tps).astype(np.float32)
    overlay_b64 = _to_heatmap_png(saliency_map, size=min(h, w), alpha=alpha)

    return ExplainabilityMaps(
        saliency=saliency_map,
        inconsistency=inconsistency_map,
        overlay_png_base64=overlay_b64,
        mean_deviation=float(deviations.mean()),
        patch_inconsistency_score=inconsistency_score,
    )
