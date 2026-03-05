"""Geometry utilities: homography computation, quad metrics, point warping."""

from __future__ import annotations

import cv2
import numpy as np

from src.data_types import Quad


def compute_homography(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    method: str = "RANSAC",
    ransac_threshold: float = 5.0,
) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
    """Compute homography from src_points to dst_points.

    Args:
        src_points: (N, 2) source points.
        dst_points: (N, 2) destination points.
        method: "RANSAC" or "LMEDS".
        ransac_threshold: Reprojection threshold for RANSAC.

    Returns:
        (H_forward, H_inverse, is_valid):
            H_forward maps src -> dst,
            H_inverse maps dst -> src,
            is_valid indicates success.
    """
    if len(src_points) < 4 or len(dst_points) < 4:
        return None, None, False

    cv_method = cv2.RANSAC if method == "RANSAC" else cv2.LMEDS
    H_forward, mask = cv2.findHomography(
        src_points.astype(np.float32),
        dst_points.astype(np.float32),
        cv_method,
        ransac_threshold,
    )

    if H_forward is None:
        return None, None, False

    if mask is not None and mask.sum() < 4:
        return None, None, False

    try:
        H_inverse = np.linalg.inv(H_forward)
    except np.linalg.LinAlgError:
        return None, None, False

    return H_forward, H_inverse, True


def quad_frontality_score(quad: Quad) -> float:
    """Estimate how frontal (rectangular) a text quad is.

    A perfectly frontal text region has parallel sides of equal length.
    We measure:
    1. Ratio of opposite side lengths (closer to 1.0 = more parallel)
    2. Angle between diagonals (perpendicular diags = more rectangular)

    Returns:
        Score in [0, 1], where 1.0 = perfectly rectangular/frontal.
    """
    pts = quad.points
    top = np.linalg.norm(pts[1] - pts[0])
    bottom = np.linalg.norm(pts[2] - pts[3])
    left = np.linalg.norm(pts[3] - pts[0])
    right = np.linalg.norm(pts[2] - pts[1])

    h_ratio = min(top, bottom) / max(top, bottom, 1e-6)
    v_ratio = min(left, right) / max(left, right, 1e-6)
    side_score = (h_ratio + v_ratio) / 2

    diag1 = pts[2] - pts[0]
    diag2 = pts[3] - pts[1]
    cos_angle = abs(np.dot(diag1, diag2)) / (
        np.linalg.norm(diag1) * np.linalg.norm(diag2) + 1e-6
    )
    angle_score = 1.0 - cos_angle

    return float(np.clip(0.6 * side_score + 0.4 * angle_score, 0, 1))


def warp_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply homography H to a set of 2D points.

    Args:
        points: (N, 2) array of (x, y) points.
        H: 3x3 homography matrix.

    Returns:
        (N, 2) warped points.
    """
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H)
    return warped.reshape(-1, 2)
