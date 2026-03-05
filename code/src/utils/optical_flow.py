"""Optical flow tracking utilities for quad point propagation."""

from __future__ import annotations

import cv2
import numpy as np


def track_points_farneback(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_points: np.ndarray,
    config,
) -> np.ndarray | None:
    """Track points using Farneback dense optical flow.

    Computes a dense flow field, then samples at quad corner locations.

    Args:
        prev_gray: Previous frame (grayscale, uint8).
        curr_gray: Current frame (grayscale, uint8).
        prev_points: (N, 2) points to track, (x, y) format.
        config: FrontalizationConfig with Farneback parameters.

    Returns:
        (N, 2) tracked points, or None if tracking failed.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=config.farneback_pyr_scale,
        levels=config.farneback_levels,
        winsize=config.farneback_winsize,
        iterations=config.farneback_iterations,
        poly_n=config.farneback_poly_n,
        poly_sigma=config.farneback_poly_sigma,
        flags=0,
    )

    h, w = prev_gray.shape
    new_points = []
    for pt in prev_points:
        x = int(np.clip(round(pt[0]), 0, w - 1))
        y = int(np.clip(round(pt[1]), 0, h - 1))
        dx, dy = flow[y, x]
        new_points.append([pt[0] + dx, pt[1] + dy])

    return np.array(new_points, dtype=np.float32)


def track_points_lucas_kanade(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_points: np.ndarray,
    config,
) -> np.ndarray | None:
    """Track points using Lucas-Kanade sparse optical flow.

    Directly tracks individual points — efficient for 4 corner points.

    Args:
        prev_gray: Previous frame (grayscale, uint8).
        curr_gray: Current frame (grayscale, uint8).
        prev_points: (N, 2) points to track.
        config: FrontalizationConfig with LK parameters.

    Returns:
        (N, 2) tracked points, or None if any point lost track.
    """
    pts = prev_points.reshape(-1, 1, 2).astype(np.float32)

    lk_params = dict(
        winSize=tuple(config.lk_win_size),
        maxLevel=config.lk_max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    next_pts, status, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts, None, **lk_params
    )

    if next_pts is None or status is None:
        return None

    if not np.all(status):
        return None

    return next_pts.reshape(-1, 2)
