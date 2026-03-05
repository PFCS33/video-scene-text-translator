"""Tests for Stage 5: Revert (De-Frontalization + Compositing)."""

import numpy as np
import pytest

from src.config import PipelineConfig
from src.data_types import FrameHomography, PropagatedROI, Quad
from src.stages.s5_revert import RevertStage


@pytest.fixture
def revert_stage(default_config):
    return RevertStage(default_config)


class TestWarpRoiToFrame:
    def test_identity_homography(self, revert_stage):
        roi = np.full((100, 200, 3), 128, dtype=np.uint8)
        alpha = np.ones((100, 200), dtype=np.float32)
        quad = Quad(points=np.array([
            [0, 0], [200, 0], [200, 100], [0, 100]
        ], dtype=np.float32))
        prop = PropagatedROI(
            frame_idx=0, track_id=0,
            roi_image=roi, alpha_mask=alpha, target_quad=quad,
        )
        hom = FrameHomography(
            frame_idx=0,
            H_to_ref=np.eye(3),
            H_from_ref=np.eye(3),
            is_valid=True,
        )
        result = revert_stage.warp_roi_to_frame(prop, hom, (100, 200))
        assert result is not None
        warped_roi, warped_alpha = result
        assert warped_roi.shape == (100, 200, 3)
        assert warped_alpha.shape == (100, 200)

    def test_invalid_homography_returns_none(self, revert_stage):
        roi = np.zeros((50, 50, 3), dtype=np.uint8)
        alpha = np.ones((50, 50), dtype=np.float32)
        quad = Quad(points=np.zeros((4, 2), dtype=np.float32))
        prop = PropagatedROI(
            frame_idx=0, track_id=0,
            roi_image=roi, alpha_mask=alpha, target_quad=quad,
        )
        hom = FrameHomography(frame_idx=0, is_valid=False)
        result = revert_stage.warp_roi_to_frame(prop, hom, (50, 50))
        assert result is None


class TestCompositeRoiIntoFrame:
    def test_full_alpha_replaces_frame(self, revert_stage):
        frame = np.full((100, 100, 3), 0, dtype=np.uint8)
        roi = np.full((100, 100, 3), 200, dtype=np.uint8)
        alpha = np.ones((100, 100), dtype=np.float32)
        result = revert_stage.composite_roi_into_frame(frame, roi, alpha)
        np.testing.assert_array_equal(result, roi)

    def test_zero_alpha_preserves_frame(self, revert_stage):
        frame = np.full((100, 100, 3), 100, dtype=np.uint8)
        roi = np.full((100, 100, 3), 200, dtype=np.uint8)
        alpha = np.zeros((100, 100), dtype=np.float32)
        result = revert_stage.composite_roi_into_frame(frame, roi, alpha)
        np.testing.assert_array_equal(result, frame)

    def test_half_alpha_blends(self, revert_stage):
        frame = np.full((100, 100, 3), 0, dtype=np.uint8)
        roi = np.full((100, 100, 3), 200, dtype=np.uint8)
        alpha = np.full((100, 100), 0.5, dtype=np.float32)
        result = revert_stage.composite_roi_into_frame(frame, roi, alpha)
        expected = 100  # 0 * 0.5 + 200 * 0.5
        assert np.abs(result.mean() - expected) < 2


class TestRevertRun:
    def test_no_rois_returns_original_frames(self, revert_stage):
        frames = {
            0: np.zeros((100, 100, 3), dtype=np.uint8),
            1: np.ones((100, 100, 3), dtype=np.uint8) * 255,
        }
        output = revert_stage.run(frames, {}, {}, [])
        assert len(output) == 2
        np.testing.assert_array_equal(output[0], frames[0])
        np.testing.assert_array_equal(output[1], frames[1])
