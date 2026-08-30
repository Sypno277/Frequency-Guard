"""Unit tests for image loading and validation."""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from PIL import Image as PILImage

from python_services.frequency_guard.config import Settings
from python_services.frequency_guard.io.loader import (
    LoadError,
    load_image_bytes,
    load_image_file,
    load_image_stream,
    validate_extension,
)


def _png_bytes() -> bytes:
    img = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    ok, enc = cv2.imencode(".png", img)
    assert ok
    return enc.tobytes()


def _jpg_bytes() -> bytes:
    img = np.zeros((64, 64, 3), dtype=np.uint8) + 120
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    assert ok
    return enc.tobytes()


def _exif_rotated_jpg_bytes(orientation: int) -> bytes:
    """Encode a JPEG with the given EXIF orientation (0x0112)."""
    img = np.zeros((32, 48, 3), dtype=np.uint8) + 90
    pil = PILImage.fromarray(img)
    exif = pil.getexif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90, exif=exif)
    return buf.getvalue()


class TestValidateExtension:
    def test_accepts_supported(self) -> None:
        settings = Settings()
        for name in ("a.jpg", "b.JPEG", "c.png", "d.webp", "e.bmp"):
            validate_extension(name, settings)  # must not raise

    def test_rejects_unsupported(self) -> None:
        settings = Settings()
        with pytest.raises(LoadError):
            validate_extension("a.gif", settings)
        with pytest.raises(LoadError):
            validate_extension("noext", settings)


class TestLoadImageBytes:
    def test_loads_png_and_reports_dims(self) -> None:
        settings = Settings()
        loaded = load_image_bytes(_png_bytes(), "test.png", settings)
        assert loaded.bgr.shape[2] == 3
        assert loaded.width == 64
        assert loaded.height == 64
        assert loaded.channels == 3

    def test_empty_payload_raises(self) -> None:
        with pytest.raises(LoadError):
            load_image_bytes(b"", "empty.png", Settings())

    def test_corrupt_payload_raises(self) -> None:
        with pytest.raises(LoadError):
            load_image_bytes(b"\x89PNG-not-a-real-image", "corrupt.png", Settings())

    def test_oversized_payload_raises(self) -> None:
        settings = Settings(max_upload_bytes=10)
        with pytest.raises(LoadError):
            load_image_bytes(b"x" * 20, "big.png", settings)


class TestLoadImageFile:
    def test_roundtrip_file(self, tmp_path) -> None:
        (tmp_path / "img.png").write_bytes(_png_bytes())
        loaded = load_image_file(tmp_path / "img.png", Settings())
        assert loaded.width == 64

    def test_unsupported_extension_fails_before_read(self, tmp_path) -> None:
        (tmp_path / "img.txt").write_bytes(b"data")
        with pytest.raises(LoadError):
            load_image_file(tmp_path / "img.txt", Settings())


class TestExifOrientation:
    def test_no_exif_unchanged(self) -> None:
        settings = Settings()
        loaded = load_image_bytes(_jpg_bytes(), "img.jpg", settings)
        assert (loaded.width, loaded.height) == (64, 64)

    @pytest.mark.parametrize("orientation", [3, 6, 8])
    def test_exif_rotation_applied(self, orientation: int) -> None:
        """Oriented JPEGs must decode to corrected dimensions."""
        settings = Settings()
        payload = _exif_rotated_jpg_bytes(orientation)
        loaded = load_image_bytes(payload, "oriented.jpg", settings)
        # source is 32(H)x48(W); 6/8 rotate 90° -> 48(H)x32(W); 3 -> unchanged
        if orientation == 3:
            assert (loaded.width, loaded.height) == (48, 32)
        else:
            assert (loaded.width, loaded.height) == (32, 48)


class TestLoadImageStream:
    def test_stream_loading(self) -> None:
        stream = io.BytesIO(_png_bytes())
        loaded = load_image_stream(stream, "stream.png", Settings())
        assert loaded.width == 64
