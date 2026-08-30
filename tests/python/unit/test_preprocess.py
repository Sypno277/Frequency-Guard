"""Unit tests for preprocessing: color conversion, letterbox resize, and the
full preprocess pipeline."""

from __future__ import annotations

import numpy as np

from python_services.frequency_guard.preprocess import (
    PreprocessedImage,
    preprocess,
    resize_square,
    to_gray,
    to_rgb_float,
    to_y_channel,
)


def _bgr_image(height: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


class TestColorConversion:
    def test_to_gray_shape_and_range(self) -> None:
        bgr = _bgr_image(32, 48)
        gray = to_gray(bgr)
        assert gray.shape == (32, 48)
        assert gray.dtype == np.float32
        assert gray.min() >= 0.0
        assert gray.max() <= 1.0

    def test_to_gray_accepts_2d(self) -> None:
        gray_in = _bgr_image(16, 16)[:, :, 0]
        gray_out = to_gray(gray_in)
        assert gray_out.shape == (16, 16)

    def test_to_rgb_float_range(self) -> None:
        rgb = to_rgb_float(_bgr_image(16, 16))
        assert rgb.shape == (16, 16, 3)
        assert rgb.min() >= 0.0 and rgb.max() <= 1.0

    def test_to_y_channel(self) -> None:
        y = to_y_channel(_bgr_image(16, 16))
        assert y.shape == (16, 16)
        assert y.min() >= 0.0 and y.max() <= 1.0


class TestResizeSquare:
    def test_landscape_letterboxed(self) -> None:
        img = _bgr_image(100, 200)
        out = resize_square(img, 128)
        assert out.shape == (128, 128, 3)
        # content is placed inside, border zeros
        assert out[0, 0].sum() == 0  # top-left corner is letterbox
        assert out[127, 127].sum() == 0  # bottom-right corner is letterbox

    def test_portrait_letterboxed(self) -> None:
        img = _bgr_image(200, 100)
        out = resize_square(img, 128)
        assert out.shape == (128, 128, 3)
        assert out[0, 0].sum() == 0
        assert out[127, 127].sum() == 0

    def test_square_no_letterbox(self) -> None:
        img = _bgr_image(64, 64)
        out = resize_square(img, 128)
        assert out.shape == (128, 128, 3)
        # center pixel on content, should be non-zero for random image
        assert out[64, 64].sum() > 0

    def test_invariant_under_small_input(self) -> None:
        img = _bgr_image(2, 2)
        out = resize_square(img, 128)
        assert out.shape == (128, 128, 3)


class TestPreprocess:
    def test_preprocess_output_fields(self, fast_settings) -> None:
        bgr = _bgr_image(80, 120)
        result = preprocess(bgr, fast_settings)
        assert isinstance(result, PreprocessedImage)
        assert result.gray.shape == (fast_settings.image_size, fast_settings.image_size)
        assert result.rgb.shape == (fast_settings.image_size, fast_settings.image_size, 3)
        assert result.y_channel.shape == result.gray.shape
        assert result.orig_size == (80, 120)

    def test_preprocess_is_deterministic(self, fast_settings) -> None:
        bgr = _bgr_image(80, 120)
        first = preprocess(bgr, fast_settings)
        second = preprocess(bgr, fast_settings)
        assert np.array_equal(first.gray, second.gray)
