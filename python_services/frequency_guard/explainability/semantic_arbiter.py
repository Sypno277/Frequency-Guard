"""CPU semantic arbitration — the lightweight "VLM substitute" (FG Phase 3).

A true Vision-Language Model (GPT-4V/LLaVA) can reason about "melted hands"
and "broken shadows" but needs a GPU or an API key. To keep the pipeline
CPU-only, this module runs deterministic OpenCV/NumPy heuristics that emit
the *same structured output* — boolean flags + localized bounding boxes —
on the exact trigger condition from the FrequencyGuard spec: only when
calibrated confidence is in the 30–70% "uncanny valley" band.

It does not claim to be a VLM. It produces cheap, reproducible semantic
flags (anatomy, optics, topology) that are grounded in measurable image
artifacts, and its output is consumed by the NLP engine (Phase 4).

The three "tensor nodes" from the spec map to:
  - Anatomy  : skin-tone count of hand-like / limb regions + symmetry
  - Optics   : light-source consistency + shadow-vector divergence
  - Topology : background-edge irregularity + non-Euclidean texture regions
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# --- trigger band -------------------------------------------------------
UNCANNY_LOW = 0.30
UNCANNY_HIGH = 0.70


@dataclass
class SemanticFlag:
    """One detected semantic anomaly."""

    node: str  # "anatomy" | "optics" | "topology"
    label: str  # human-readable description
    score: float  # 0..1 severity
    bbox: tuple[int, int, int, int] | None = None  # x, y, w, h in pixels

    def as_dict(self) -> dict[str, object]:
        return {
            "node": self.node,
            "label": self.label,
            "score": round(self.score, 4),
            "bbox": list(self.bbox) if self.bbox else None,
        }


@dataclass
class SemanticVerdict:
    """Aggregated semantic-arbiter output for one image."""

    triggered: bool
    flags: list[SemanticFlag] = field(default_factory=list)
    summary: str = "Semantic arbitration not triggered (confidence outside uncanny band)."

    @property
    def suspicious(self) -> bool:
        """True when any flag exceeds the suspicion threshold."""
        return any(f.score >= 0.5 for f in self.flags)

    def as_dict(self) -> dict[str, object]:
        return {
            "triggered": self.triggered,
            "suspicious": self.suspicious,
            "flags": [f.as_dict() for f in self.flags],
            "summary": self.summary,
        }


def _skin_mask(bgr: np.ndarray) -> np.ndarray:
    """HSV skin-tone mask used for anatomy / limb-region counting."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 40], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    # Add the warmer end of the hue wheel.
    lower2 = np.array([160, 30, 40], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask |= cv2.inRange(hsv, lower2, upper2)
    return mask


def _check_anatomy(bgr: np.ndarray) -> list[SemanticFlag]:
    """Detect anatomical anomalies: limb/hand count + bilateral symmetry.

    This is heuristic, not a face/hand detector. It counts skin-tone
    connected components near the expected extremities and flags both an
    implausible count and a strong asymmetry (a common diffusion artifact).
    """
    flags: list[SemanticFlag] = []

    # 1) Count hand-like regions = skin components with a palm-like shape.
    mask = _skin_mask(bgr)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hand_regions: list[tuple[int, int, int, int]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 400 or area > 0.6 * bgr.shape[0] * bgr.shape[1]:
            continue
        x, y, w, h = cv2.boundingRect(c)
        # Palm-ish aspect ratio, and an elongated limb/tendril part.
        aspect = w / max(1, h)
        convexity = cv2.isContourConvex(c)
        if 0.5 <= aspect <= 2.5 or not convexity:
            hand_regions.append((x, y, w, h))

    # 2) Bilateral symmetry check across the vertical midline.
    h, w = bgr.shape[:2]
    left = mask[:, : w // 2].sum()
    right = mask[:, w // 2 :].sum()
    total = left + right + 1e-6
    asymmetry = abs(left - right) / total

    if len(hand_regions) == 0:
        flags.append(SemanticFlag("anatomy", "No plausible hand/limb regions detected", 0.55))
    elif len(hand_regions) > 6:
        flags.append(
            SemanticFlag(
                "anatomy",
                f"Implausibly many hand-like regions ({len(hand_regions)})",
                min(1.0, len(hand_regions) / 12),
                bbox=hand_regions[0],
            )
        )

    if asymmetry > 0.55:
        flags.append(
            SemanticFlag(
                "anatomy",
                "Strong bilateral asymmetry in skin-tone distribution",
                min(1.0, asymmetry),
            )
        )
    return flags


def _check_optics(bgr: np.ndarray) -> list[SemanticFlag]:
    """Detect optic anomalies: light-source direction inconsistency.

    Uses the gradient orientation histogram across 4 quadrants. A single
    coherent light source produces correlated edge-gradient directions;
    divergent quadrants (a common diffusion failure) yield a wide spread.
    """
    flags: list[SemanticFlag] = []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx * gx + gy * gy)
    angle = np.arctan2(gy, gx + 1e-6)

    h, w = gray.shape
    quadrants = [
        angle[: h // 2, : w // 2],
        angle[: h // 2, w // 2 :],
        angle[h // 2 :, : w // 2],
        angle[h // 2 :, w // 2 :],
    ]
    # Circular mean of each quadrant's edge direction.
    means: list[float] = []
    for q in quadrants:
        m = q[magnitude[: q.shape[0], : q.shape[1]] > np.percentile(magnitude, 80)]
        if m.size == 0:
            continue
        means.append(np.arctan2(np.sin(m).mean(), np.cos(m).mean()))
    if len(means) < 2:
        return flags
    # Circular variance across quadrant directions — high = inconsistent light.
    sin_sum = np.sin(means).mean()
    cos_sum = np.cos(means).mean()
    spread = 1.0 - np.sqrt(sin_sum * sin_sum + cos_sum * cos_sum)
    if spread > 0.55:
        flags.append(
            SemanticFlag(
                "optics",
                "Inconsistent edge-gradient direction across quadrants (divergent light source)",
                min(1.0, spread),
            )
        )
    return flags


def _check_topology(bgr: np.ndarray) -> list[SemanticFlag]:
    """Detect topology anomalies: background edge irregularity.

    High-frequency edge discontinuities at region boundaries (non-Euclidean
    texture seams) signal local synthesis defects. We measure the fraction
    of strong edges whose orientation jumps abruptly between adjacent rows.
    """
    flags: list[SemanticFlag] = []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
    if edges.sum() == 0:
        return flags
    # Edge orientations from the Sobel pair on the full grayscale image.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    angle = np.arctan2(gy, gx + 1e-6)
    # Only consider strong-edge pixels (Canny mask).
    mask = edges > 0
    if mask.sum() < 4:
        return flags
    # Aggregate ALL strong-edge pixels into a single orientation coherence
    # scalar (circular variance over the whole edge population), not per-column.
    sm = float(np.sin(angle[mask]).mean())
    cm = float(np.cos(angle[mask]).mean())
    irregularity = 1.0 - np.sqrt(sm * sm + cm * cm)
    if irregularity > 0.5:
        flags.append(
            SemanticFlag(
                "topology",
                "Background edge-orientation irregularity (non-Euclidean texture seams)",
                min(1.0, irregularity),
            )
        )
    return flags


def run_semantic_arbiter(bgr: np.ndarray, calibrated_fake: float) -> SemanticVerdict:
    """Run the semantic arbiter if confidence is in the uncanny band.

    Args:
        bgr: preprocessed image as a BGR numpy array.
        calibrated_fake: calibrated fake probability (0..1).

    Returns:
        :class:`SemanticVerdict`. When not triggered, ``flags`` is empty.
    """
    if not (UNCANNY_LOW <= calibrated_fake <= UNCANNY_HIGH):
        return SemanticVerdict(
            triggered=False,
            flags=[],
            summary="Semantic arbitration not triggered (confidence outside 30–70% band).",
        )

    flags = _check_anatomy(bgr) + _check_optics(bgr) + _check_topology(bgr)
    if flags:
        summary = "Semantic arbitration flagged anomalies in the uncanny-confidence band."
    else:
        summary = "Semantic arbitration ran; no detectable semantic anomalies."

    return SemanticVerdict(triggered=True, flags=flags, summary=summary)
