"""Tests for image processing utilities."""

import cv2
import numpy as np
import pytest

from src.utils.image_processing import (
    compute_contrast,
    compute_sharpness,
    match_histogram_luminance,
)


class TestComputeSharpness:
    def test_sharp_image_higher_score(self):
        """A sharp edge should score higher than a blurred one."""
        sharp = np.zeros((100, 100, 3), dtype=np.uint8)
        sharp[:, 50:] = 255

        blurry = cv2.GaussianBlur(sharp, (31, 31), 10)

        sharp_score = compute_sharpness(sharp)
        blurry_score = compute_sharpness(blurry)
        assert sharp_score > blurry_score

    def test_score_in_range(self, synthetic_frame):
        score = compute_sharpness(synthetic_frame)
        assert 0 <= score <= 1

    def test_uniform_image_low_sharpness(self):
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        score = compute_sharpness(img)
        assert score < 0.1


class TestComputeContrast:
    def test_high_contrast(self):
        """Black and white checkerboard should have high contrast."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:50, :] = 255
        score = compute_contrast(img)
        assert score > 0.5

    def test_uniform_low_contrast(self):
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        score = compute_contrast(img)
        assert score < 0.1

    def test_black_image(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        score = compute_contrast(img)
        assert score == 0.0

    def test_score_in_range(self, synthetic_frame):
        score = compute_contrast(synthetic_frame)
        assert 0 <= score <= 1


class TestMatchHistogramLuminance:
    def test_output_shape_preserved(self):
        source = np.random.randint(0, 255, (50, 80, 3), dtype=np.uint8)
        reference = np.random.randint(0, 255, (50, 80, 3), dtype=np.uint8)
        result = match_histogram_luminance(source, reference)
        assert result.shape == source.shape
        assert result.dtype == np.uint8

    def test_identity_match(self):
        """Matching an image to itself should produce approximately the same image."""
        img = np.random.randint(50, 200, (40, 60, 3), dtype=np.uint8)
        result = match_histogram_luminance(img, img)
        # Allow some rounding error from color space conversion
        diff = np.abs(result.astype(int) - img.astype(int))
        assert diff.mean() < 5

    def test_brightness_transfer(self):
        """A dark source matched to a bright reference should become brighter."""
        dark = np.full((40, 60, 3), 50, dtype=np.uint8)
        bright = np.full((40, 60, 3), 200, dtype=np.uint8)
        result = match_histogram_luminance(dark, bright)
        # The result should be brighter than the original dark image
        assert result.mean() > dark.mean()

    def test_invalid_color_space(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported color space"):
            match_histogram_luminance(img, img, color_space="RGB")
