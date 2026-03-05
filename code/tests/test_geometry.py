"""Tests for geometry utilities."""

import numpy as np
import pytest

from src.data_types import Quad
from src.utils.geometry import compute_homography, quad_frontality_score, warp_points


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
