"""Image loading and validation.

Decodes images with OpenCV, applies EXIF orientation when available, and
raises typed errors for unsupported formats or corrupt payloads so callers
(the API layer, batch jobs, tests) can respond with meaningful 4xx errors
instead of 500s.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np

from ..config import Settings

SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


class LoadError(ValueError):
    """Raised when an image cannot be decoded or is not supported."""


@dataclass(frozen=True)
class LoadedImage:
    """Decoded image plus provenance metadata."""

    bgr: np.ndarray  # HxWx3 uint8, BGR (OpenCV convention)
    source: str
    width: int
    height: int
    channels: int


def validate_extension(filename: str, settings: Settings) -> None:
    """Ensure ``filename`` has an allowed image extension.

    Raises:
        LoadError: if the extension is not in ``settings.supported_formats``.
    """
    ext = Path(filename).suffix.lower()
    if ext not in settings.supported_formats:
        allowed = ", ".join(settings.supported_formats)
        raise LoadError(f"Unsupported format '{ext or '<none>'}'. Allowed: {allowed}")


def _apply_exif_orientation(img: np.ndarray, exif_orientation: int | None) -> np.ndarray:
    """Apply EXIF orientation (1-8) to an already-decoded image (numpy-only)."""
    if exif_orientation is None or exif_orientation == 1 or img.size == 0:
        return img

    if exif_orientation == 2:
        out = cv2.flip(img, 1)
    elif exif_orientation == 3:
        out = cv2.rotate(img, cv2.ROTATE_180)
    elif exif_orientation == 4:
        out = cv2.flip(img, 0)
    elif exif_orientation == 5:
        out = cv2.rotate(cv2.flip(img, 1), cv2.ROTATE_90_CLOCKWISE)
    elif exif_orientation == 6:
        out = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif exif_orientation == 7:
        out = cv2.rotate(cv2.flip(img, 1), cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif exif_orientation == 8:
        out = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        out = img
    return out


def _read_exif_orientation(buffer: bytes) -> int | None:
    """Extract EXIF orientation (tag 0x0112) from a JPEG/TIFF buffer, if present."""
    orientation = None
    try:
        from PIL import Image as PILImage

        with PILImage.open(io.BytesIO(buffer)) as pil_img:
            exif = pil_img.getexif()
            raw = exif.get(0x0112, 1)
            if isinstance(raw, tuple | list):
                raw = raw[0]
            orientation = int(raw)
            if orientation == 1:
                return None  # no rotation needed
    except Exception:
        return None
    return orientation


def load_image_bytes(buffer: bytes, source: str, settings: Settings) -> LoadedImage:
    """Decode an image from an in-memory byte buffer.

    Args:
        buffer: raw file bytes.
        source: human-readable origin (filename or "upload").
        settings: runtime configuration.

    Raises:
        LoadError: if the payload cannot be decoded or exceeds size limits.
    """
    if len(buffer) == 0:
        raise LoadError("Empty file payload")
    if len(buffer) > settings.max_upload_bytes:
        raise LoadError(f"File too large: {len(buffer)} bytes > {settings.max_upload_bytes} limit")

    orientation = _read_exif_orientation(buffer)
    arr = np.frombuffer(buffer, dtype=np.uint8)
    # Decode raw BGR pixels with EXIF ignored: newer OpenCV applies EXIF at
    # decode time, so orientation is applied exactly once by us below for
    # consistent behavior across OpenCV versions. IMREAD_IGNORE_ORIENTATION
    # (128) must be OR'd with IMREAD_COLOR (1) or the mode bits are 0
    # (= grayscale).
    flags = int(cv2.IMREAD_COLOR) | int(cv2.IMREAD_IGNORE_ORIENTATION)
    img = cv2.imdecode(arr, flags)
    if img is None:
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise LoadError("Could not decode image payload (corrupted or unsupported codec)")

    img = _apply_exif_orientation(img, orientation)

    height, width = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1
    return LoadedImage(bgr=img, source=source, width=width, height=height, channels=channels)


def load_image_file(path: str | Path, settings: Settings) -> LoadedImage:
    """Decode an image from disk, validating format first."""
    p = Path(path)
    validate_extension(p.name, settings)
    buffer = p.read_bytes()
    return load_image_bytes(buffer, str(p), settings)


def load_image_stream(stream: BinaryIO, source: str, settings: Settings) -> LoadedImage:
    """Decode an image from a binary stream."""
    buffer = stream.read()
    return load_image_bytes(buffer, source, settings)
