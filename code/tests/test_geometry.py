"""Tests for geometry utilities."""

import numpy as np
import pytest

from src.data_types import Quad
from src.utils.geometry import (
    canonical_rect_from_quad,
    compute_homography,
    quad_area,
    quad_bbox_area_ratio,
    quad_frontality_score,
    warp_points,
)


class TestComputeHomography:
    def test_identity_same_points(self):
        """Same src and dst should give identity homography."""
        pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        H_fwd, H_inv, valid = compute_homography(pts, pts)
        assert valid
        assert H_fwd is not None
        np.testing.assert_allclose(H_fwd, np.eye(3), atol=0.1)

    def test_known_translation(self):
        """Translation should be encoded in the homography."""
        src = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
        dst = src + np.array([10, 20], dtype=np.float32)
        H_fwd, H_inv, valid = compute_homography(src, dst)
        assert valid

        # Warp src points through H_fwd and verify they match dst
        warped = warp_points(src, H_fwd)
        np.testing.assert_allclose(warped, dst, atol=1.0)

    def test_inverse_round_trip(self):
        """H_fwd composed with H_inv should approximate identity."""
        src = np.array([[10, 10], [110, 15], [105, 80], [5, 75]], dtype=np.float32)
        dst = np.array([[20, 20], [120, 20], [120, 70], [20, 70]], dtype=np.float32)
        H_fwd, H_inv, valid = compute_homography(src, dst)
        assert valid
        composed = H_inv @ H_fwd
        np.testing.assert_allclose(
            composed / composed[2, 2], np.eye(3), atol=0.1
        )

    def test_too_few_points(self):
        """Less than 4 points should return invalid."""
        src = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)
        dst = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)
        _, _, valid = compute_homography(src, dst)
        assert not valid


class TestQuadFrontalityScore:
    def test_perfect_rectangle_high_score(self):
        quad = Quad(points=np.array([
            [0, 0], [200, 0], [200, 50], [0, 50]
        ], dtype=np.float32))
        score = quad_frontality_score(quad)
        assert score > 0.6

    def test_trapezoid_lower_score(self, trapezoid_quad):
        rect_quad = Quad(points=np.array([
            [200, 150], [440, 150], [440, 250], [200, 250]
        ], dtype=np.float32))
        trap_score = quad_frontality_score(trapezoid_quad)
        rect_score = quad_frontality_score(rect_quad)
        assert rect_score > trap_score

    def test_score_in_range(self, rect_quad):
        score = quad_frontality_score(rect_quad)
        assert 0 <= score <= 1


class TestQuadArea:
    def test_rectangle(self):
        """100x50 rectangle should have area 5000."""
        quad = Quad(points=np.array([
            [0, 0], [100, 0], [100, 50], [0, 50]
        ], dtype=np.float32))
        assert quad_area(quad) == pytest.approx(5000.0, abs=0.1)

    def test_unit_square(self):
        quad = Quad(points=np.array([
            [0, 0], [1, 0], [1, 1], [0, 1]
        ], dtype=np.float32))
        assert quad_area(quad) == pytest.approx(1.0, abs=0.01)

    def test_degenerate_line(self):
        """Collinear points should have zero area."""
        quad = Quad(points=np.array([
            [0, 0], [10, 0], [20, 0], [30, 0]
        ], dtype=np.float32))
        assert quad_area(quad) == pytest.approx(0.0, abs=0.01)


class TestQuadBboxAreaRatio:
    def test_perfect_rectangle_ratio_one(self):
        """A perfect axis-aligned rectangle fills its bbox completely."""
        quad = Quad(points=np.array([
            [0, 0], [200, 0], [200, 50], [0, 50]
        ], dtype=np.float32))
        ratio = quad_bbox_area_ratio(quad)
        assert ratio == pytest.approx(1.0, abs=0.01)

    def test_trapezoid_less_than_one(self, trapezoid_quad):
        ratio = quad_bbox_area_ratio(trapezoid_quad)
        assert 0.0 < ratio < 1.0

    def test_rectangle_higher_than_trapezoid(self, trapezoid_quad):
        rect = Quad(points=np.array([
            [200, 150], [440, 150], [440, 250], [200, 250]
        ], dtype=np.float32))
        assert quad_bbox_area_ratio(rect) > quad_bbox_area_ratio(trapezoid_quad)

    def test_score_in_range(self, rect_quad):
        ratio = quad_bbox_area_ratio(rect_quad)
        assert 0 <= ratio <= 1

    def test_degenerate_quad(self):
        """Zero-area quad should return 0."""
        quad = Quad(points=np.zeros((4, 2), dtype=np.float32))
        ratio = quad_bbox_area_ratio(quad)
        assert ratio == 0.0


class TestCanonicalRectFromQuad:
    def test_rectangle_preserves_dimensions(self):
        """A 200x50 rectangle should produce a 200x50 canonical rect."""
        quad = Quad(points=np.array([
            [0, 0], [200, 0], [200, 50], [0, 50]
        ], dtype=np.float32))
        rect, size = canonical_rect_from_quad(quad)
        assert size == (200, 50)
        expected = np.array([[0, 0], [200, 0], [200, 50], [0, 50]], dtype=np.float32)
        np.testing.assert_allclose(rect, expected, atol=1.0)

    def test_trapezoid_averages_edges(self):
        """Trapezoid: top=200, bottom=300, sides are diagonal (~71px)."""
        quad = Quad(points=np.array([
            [50, 0], [250, 0], [300, 50], [0, 50]
        ], dtype=np.float32))
        rect, size = canonical_rect_from_quad(quad)
        assert size[0] == 250  # avg of top(200) and bottom(300)
        # Height is avg of left and right diagonal edges (~70.7)
        assert 70 <= size[1] <= 72
        assert rect.shape == (4, 2)
        # Starts at origin, axis-aligned
        np.testing.assert_allclose(rect[0], [0, 0], atol=1e-5)

    def test_output_starts_at_origin(self, rect_quad):
        rect, _ = canonical_rect_from_quad(rect_quad)
        np.testing.assert_allclose(rect[0], [0, 0], atol=1e-5)

    def test_output_is_axis_aligned(self, rect_quad):
        rect, (w, h) = canonical_rect_from_quad(rect_quad)
        # TL, TR, BR, BL ordering
        np.testing.assert_allclose(rect[0], [0, 0], atol=1e-5)
        np.testing.assert_allclose(rect[1], [w, 0], atol=1e-5)
        np.testing.assert_allclose(rect[2], [w, h], atol=1e-5)
        np.testing.assert_allclose(rect[3], [0, h], atol=1e-5)

    def test_degenerate_quad_raises(self):
        """Near-zero width or height should raise ValueError."""
        quad = Quad(points=np.array([
            [0, 0], [0.5, 0], [0.5, 50], [0, 50]
        ], dtype=np.float32))
        with pytest.raises(ValueError, match="Degenerate quad"):
            canonical_rect_from_quad(quad)

    def test_degenerate_zero_height_raises(self):
        quad = Quad(points=np.array([
            [0, 0], [200, 0], [200, 0.5], [0, 0.5]
        ], dtype=np.float32))
        with pytest.raises(ValueError, match="Degenerate quad"):
            canonical_rect_from_quad(quad)


class TestWarpPoints:
    def test_identity(self):
        pts = np.array([[10, 20], [30, 40]], dtype=np.float32)
        H = np.eye(3, dtype=np.float64)
        warped = warp_points(pts, H)
        np.testing.assert_allclose(warped, pts, atol=1e-5)

    def test_translation(self):
        pts = np.array([[0, 0], [10, 10]], dtype=np.float32)
        H = np.array([[1, 0, 5], [0, 1, 10], [0, 0, 1]], dtype=np.float64)
        warped = warp_points(pts, H)
        expected = np.array([[5, 10], [15, 20]], dtype=np.float32)
        np.testing.assert_allclose(warped, expected, atol=1e-5)
