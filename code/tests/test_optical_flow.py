"""Tests for optical flow tracking utilities."""

import cv2
import numpy as np
import pytest

from src.config import FrontalizationConfig
from src.utils.optical_flow import track_points_farneback, track_points_lucas_kanade


@pytest.fixture
def flow_config():
    return FrontalizationConfig()


@pytest.fixture
def shifted_frame_pair():
    """Create two grayscale frames where the second is shifted by (10, 5)."""
    frame1 = np.zeros((200, 300), dtype=np.uint8)
    # Draw a bright rectangle as trackable feature
    cv2.rectangle(frame1, (100, 60), (200, 120), 255, -1)

    frame2 = np.zeros((200, 300), dtype=np.uint8)
    cv2.rectangle(frame2, (110, 65), (210, 125), 255, -1)

    return frame1, frame2


class TestFarneback:
    def test_tracks_shifted_points(self, shifted_frame_pair, flow_config):
        prev, curr = shifted_frame_pair
        points = np.array([
            [100, 60], [200, 60], [200, 120], [100, 120]
        ], dtype=np.float32)

        tracked = track_points_farneback(prev, curr, points, flow_config)
        assert tracked is not None
        assert tracked.shape == (4, 2)

        # Points should have moved approximately (+10, +5)
        displacement = tracked - points
        mean_dx = displacement[:, 0].mean()
        mean_dy = displacement[:, 1].mean()
        assert abs(mean_dx - 10) < 5  # Tolerant due to flow estimation noise
        assert abs(mean_dy - 5) < 5


class TestLucasKanade:
    def test_tracks_shifted_points(self, shifted_frame_pair, flow_config):
        prev, curr = shifted_frame_pair
        points = np.array([
            [100, 60],  # Corner of rectangle — strong gradient for LK
        ], dtype=np.float32)

        tracked = track_points_lucas_kanade(prev, curr, points, flow_config)
        assert tracked is not None
        assert tracked.shape == (1, 2)

        dx = tracked[0, 0] - points[0, 0]
        dy = tracked[0, 1] - points[0, 1]
        assert abs(dx - 10) < 5
        assert abs(dy - 5) < 5
