"""DCT-based block-artifact features.

JPEG and most codecs process images in 8x8 blocks; AI generators that
inherit or mimic these pipelines (or that are upsampled through conv layers
with block-like structure) leave fingerprints in DCT coefficient
statistics:

- **High-frequency coefficient anomaly**: diffusion models often produce
  flattened or unusually structured high-wavenumber coefficients.
- **Block-boundary energy**: periodic energy at block boundaries is the
  classic JPEG double-compression / re-encoding signal.
- **Coefficient kurtosis**: the distribution of DCT coefficients (heavy
  tails vs parametric Gaussian) differs between natural noise and
  synthetic textures.

Implementation uses a strided windowed DCT via `scipy.fftpack.dctn` on
8x8 tiles — vectorized, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fftpack import dctn

from ..config import Settings

_EPS = 1e-12


@dataclass(frozen=True)
class DCTFeatures:
    """DCT statistics feature vector."""

    coeff_kurtosis: float
    high_freq_energy_ratio: float
    mid_band_energy_ratio: float
    block_boundary_energy: float
    coeff_std: float
    avg_dc: float
    low_high_ratio: float

    def as_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.coeff_kurtosis,
                self.high_freq_energy_ratio,
                self.mid_band_energy_ratio,
                self.block_boundary_energy,
                self.coeff_std,
                self.avg_dc,
                self.low_high_ratio,
            ],
            dtype=np.float64,
        )


def _block_dct_energy_map(gray: np.ndarray, block: int) -> np.ndarray:
    """Return (Bx, By, block, block) DCT energy map per tile.

    Crops to a multiple of ``block`` (center crop for stability) and applies
    an orthonormal 2D DCT-II per tile, returning squared coefficients.
    """
    h, w = gray.shape
    th, tw = (h // block) * block, (w // block) * block
    y0, x0 = (h - th) // 2, (w - tw) // 2
    crop = gray[y0 : y0 + th, x0 : x0 + tw]

    tiles = crop.reshape(th // block, block, tw // block, block)
    tiles = tiles.transpose(0, 2, 1, 3)  # (rows, cols, block, block)
    tiles = tiles.reshape(-1, block, block)  # (n_tiles, block, block)

    # One vectorized 2D DCT across each tile's last two axes. Replaces
    # ~1024 per-tile Python-loop scipy calls — the dominant per-image cost.
    dcts = dctn(tiles, axes=(-2, -1), norm="ortho")
    return (dcts**2).reshape(th // block, tw // block, block, block)


def extract_dct_features(gray: np.ndarray, settings: Settings) -> DCTFeatures:
    """Extract DCT features from a normalized grayscale image.

    Args:
        gray: HxW float32 in [0, 1].
        settings: configuration (dct_block_size).

    Returns:
        DCTFeatures.
    """
    block = settings.dct_block_size
    energy = _block_dct_energy_map(gray, block)

    # --- coefficient statistics across all tiles ----------------------
    coeffs = np.sqrt(energy)  # magnitude coefficients
    # exclude DC (0,0) for kurtosis/std: DC is a per-tile offset, not texture
    ac = coeffs[..., 1:, 1:].ravel()
    std_ac = float(ac.std()) if ac.size else 0.0
    # excess kurtosis (Fisher): 0 for Gaussian
    if ac.size > 4 and std_ac > _EPS:
        mean = float(ac.mean())
        m4 = float(((ac - mean) ** 4).mean())
        m2 = float(((ac - mean) ** 2).mean())
        kurtosis = m4 / (m2**2 + _EPS) - 3.0
    else:
        kurtosis = 0.0

    # --- band energy ratios -------------------------------------------
    # frequencies (u,v) in [0, block-1]; split into low/mid/high by radius
    uu, vv = np.mgrid[0:block, 0:block]
    radius = np.sqrt(uu.astype(np.float64) ** 2 + vv.astype(np.float64) ** 2)
    low = radius <= block / 4.0
    mid = (radius > block / 4.0) & (radius <= block / 2.0)
    high = radius > block / 2.0
    low[0, 0] = False  # exclude DC from all band ratios

    low_energy = float(energy[..., low].sum())
    mid_energy = float(energy[..., mid].sum())
    high_energy = float(energy[..., high].sum())
    total_ac = low_energy + mid_energy + high_energy + 1e-12

    # --- block-boundary energy (grid artifact strength) ----------------
    # Energy concentrated along tile borders at u==0 or v==0 rows that are
    # not the DC → periodic boundary signal. Sum boundary edge coefficients.
    boundary_mask = np.zeros((block, block), dtype=bool)
    boundary_mask[0, :] = True  # top edge
    boundary_mask[-1, :] = True  # bottom edge
    boundary_mask[:, 0] = True  # left edge
    boundary_mask[:, -1] = True  # right edge
    boundary_mask[0, 0] = False  # DC row/col are legitimately large
    boundary_energy = float((energy * boundary_mask).sum())
    total_energy = float(energy.sum()) + _EPS

    avg_dc = float(energy[..., 0, 0].mean())

    return DCTFeatures(
        coeff_kurtosis=kurtosis,
        high_freq_energy_ratio=high_energy / total_ac,
        mid_band_energy_ratio=mid_energy / total_ac,
        block_boundary_energy=boundary_energy / total_energy,
        coeff_std=std_ac,
        avg_dc=avg_dc,
        low_high_ratio=(low_energy + _EPS) / (high_energy + _EPS),
    )
