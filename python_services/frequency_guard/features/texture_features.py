"""Spatial texture features: GLCM and fractal dimension.

These complement the frequency-domain features by capturing spatial
regularity. GLCM statistics quantify contrast/correlation/energy/homogeneity
of gray-level co-occurrence; fractal dimension (box-counting on a Sobel
edge map) measures the complexity of image structure. Natural photography
occupies a characteristic band of these values that generative models drift
from (over-smooth textures, unnaturally regular microstructure).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

from ..config import Settings

_EPS = 1e-12

_GLCM_DISTANCES = (1, 2)
_GLCM_ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)


@dataclass(frozen=True)
class TextureFeatures:
    """GLCM + fractal-dimension feature vector."""

    glcm_contrast: float
    glcm_correlation: float
    glcm_energy: float
    glcm_homogeneity: float
    glcm_dissimilarity: float
    fractal_dimension: float

    def as_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.glcm_contrast,
                self.glcm_correlation,
                self.glcm_energy,
                self.glcm_homogeneity,
                self.glcm_dissimilarity,
                self.fractal_dimension,
            ],
            dtype=np.float64,
        )


def _glcm_stats(gray: np.ndarray) -> tuple[float, float, float, float, float]:
    """Compute GLCM properties from a quantized 8-bit grayscale image."""
    levels = 64
    img = np.clip(gray / (1.0 / levels), 0, levels - 1).astype(np.uint8)

    glcm = graycomatrix(
        img,
        distances=_GLCM_DISTANCES,
        angles=_GLCM_ANGLES,
        levels=levels,
        symmetric=True,
        normed=True,
    )

    contrast = float(np.mean(graycoprops(glcm, "contrast")))
    correlation = float(np.mean(graycoprops(glcm, "correlation")))
    energy = float(np.mean(graycoprops(glcm, "energy")))
    homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
    dissimilarity = float(np.mean(graycoprops(glcm, "dissimilarity")))
    return contrast, correlation, energy, homogeneity, dissimilarity


def _fractal_dimension(gray: np.ndarray) -> float:
    """Box-counting fractal dimension of the Sobel edge map (2D).

    Counts boxes (edge or no edge) at logarithmically spaced scales and
    returns -slope of log(count) vs log(1/scale).
    """
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    edges = magnitude > (magnitude.mean() + magnitude.std() * 0.5)

    h, w = edges.shape
    max_scale = min(h, w) // 2
    scales = [s for s in (2, 4, 8, 16, 32, 64) if s <= max_scale]
    if not scales:
        return 0.0

    # Vectorized box counting: for each scale, crop the edge map to a whole
    # number of boxes and reduce each block with a single max via reshape.
    # This replaces the per-box Python loop (O(n_boxes) interpreted calls)
    # with one vectorized reduction per scale.
    th = (h // max(scales)) * max(scales)
    tw = (w // max(scales)) * max(scales)
    edges_c = edges[:th, :tw]
    counts = []
    for scale in scales:
        blocks = edges_c.reshape(th // scale, scale, tw // scale, scale).transpose(0, 2, 1, 3)
        n_blocks = int(np.count_nonzero(blocks.any(axis=(1, 3))))
        counts.append(max(n_blocks, 1))

    x = np.log(1.0 / np.asarray(scales, dtype=np.float64))
    y = np.log(np.asarray(counts, dtype=np.float64))
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def extract_texture_features(gray: np.ndarray, settings: Settings) -> TextureFeatures:
    """Extract GLCM + fractal-dimension features from normalized grayscale.

    Args:
        gray: HxW float32 in [0, 1].
        settings: unused beyond API consistency (kept for symmetry).

    Returns:
        TextureFeatures.
    """
    contrast, correlation, energy, homogeneity, dissimilarity = _glcm_stats(gray)
    return TextureFeatures(
        glcm_contrast=contrast,
        glcm_correlation=correlation,
        glcm_energy=energy,
        glcm_homogeneity=homogeneity,
        glcm_dissimilarity=dissimilarity,
        fractal_dimension=_fractal_dimension(gray),
    )
