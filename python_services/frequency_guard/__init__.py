"""Frequency Guard v2 — CPU-only frequency-domain AI-image detector.

Classical signal-processing feature extraction (numpy/scipy/pywavelets/
scikit-image) feeding a calibrated scikit-learn ensemble. No GPU, no torch.
"""

from .config import Settings
from .logging import get_logger

__all__ = ["Settings", "get_logger"]
