"""Tests for Stage 4: Propagation."""

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from src.data_types import Quad, TextDetection, TextTrack
from src.stages.s4_propagation import PropagationStage


@pytest.fixture
def propagation_stage(default_config):
    return PropagationStage(default_config)


class TestPropagateToFrame:
    def test_output_shape_preserved(self, propagation_stage):
        edited = np.random.randint(0, 255, (50, 100, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (50, 100, 3), dtype=np.uint8)
        result = propagation_stage.propagate_to_frame(edited, target)
        assert result.shape == edited.shape

    def test_different_size_target_resized(self, propagation_stage):
        edited = np.random.randint(0, 255, (50, 100, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (80, 150, 3), dtype=np.uint8)
        result = propagation_stage.propagate_to_frame(edited, target)
        assert result.shape == edited.shape


class TestAlphaMask:
    def test_shape(self, propagation_stage):
        mask = propagation_stage._create_alpha_mask((50, 100))
        assert mask.shape == (50, 100)
        assert mask.dtype == np.float32

    def test_center_is_one(self, propagation_stage):
        mask = propagation_stage._create_alpha_mask((100, 200))
        assert mask[50, 100] == 1.0

    def test_edges_less_than_center(self, propagation_stage):
        mask = propagation_stage._create_alpha_mask((100, 200))
        assert mask[0, 0] < mask[50, 100]
        assert mask[0, 100] < mask[50, 100]

    def test_values_in_range(self, propagation_stage):
        mask = propagation_stage._create_alpha_mask((80, 120))
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0


class TestPropagationRun:
    def test_produces_propagated_rois(self, propagation_stage, synthetic_frame):
        quad = Quad(points=np.array([
            [200, 150], [440, 150], [440, 250], [200, 250]
        ], dtype=np.float32))
        det = TextDetection(
            frame_idx=0, quad=quad, bbox=quad.to_bbox(),
            text="HELLO", ocr_confidence=0.9,
        )
        edited_roi = np.full((100, 240, 3), 128, dtype=np.uint8)
        track = TextTrack(
            track_id=0, source_text="HELLO", target_text="HOLA",
            source_lang="en", target_lang="es",
            detections={0: det},
            reference_frame_idx=0,

            edited_roi=edited_roi,
        )
        frames = {0: synthetic_frame}
        result = propagation_stage.run([track], frames)
        assert 0 in result
        assert len(result[0]) == 1
        assert result[0][0].roi_image.shape[:2] == edited_roi.shape[:2]

    def test_target_canonical_roi_default_off(self, propagation_stage, synthetic_frame):
        """By default, PropagatedROI.target_roi_canonical should be None."""
        quad = Quad(points=np.array([
            [200, 150], [440, 150], [440, 250], [200, 250]
        ], dtype=np.float32))
        det = TextDetection(
            frame_idx=0, quad=quad, bbox=quad.to_bbox(),
            text="HELLO", ocr_confidence=0.9,
        )
        edited_roi = np.full((100, 240, 3), 128, dtype=np.uint8)
        track = TextTrack(
            track_id=0, source_text="HELLO", target_text="HOLA",
            source_lang="en", target_lang="es",
            detections={0: det},
            reference_frame_idx=0,
            edited_roi=edited_roi,
        )
        result = propagation_stage.run([track], {0: synthetic_frame})
        assert result[0][0].target_roi_canonical is None

    def test_target_canonical_roi_populated_when_flag_on(
        self, default_config, synthetic_frame,
    ):
        """When save_target_canonical_roi=True, S4 must populate the field
        with the same canonical ROI it fed to LCM/histogram matching."""
        default_config.propagation.save_target_canonical_roi = True
        stage = PropagationStage(default_config)

        quad = Quad(points=np.array([
            [50, 50], [250, 50], [250, 150], [50, 150]
        ], dtype=np.float32))
        H = np.eye(3, dtype=np.float64)
        det = TextDetection(
            frame_idx=0, quad=quad, bbox=quad.to_bbox(),
            text="HELLO", ocr_confidence=0.9,
            H_to_frontal=H, H_from_frontal=H, homography_valid=True,
        )
        edited_roi = np.full((80, 180, 3), 150, dtype=np.uint8)
        track = TextTrack(
            track_id=0, source_text="HELLO", target_text="HOLA",
            source_lang="en", target_lang="es",
            detections={0: det},
            reference_frame_idx=0,
            canonical_size=(180, 80),
            edited_roi=edited_roi,
        )
        # Frame needs to be at least 180x80 for H=identity warp to not go out
        # of bounds and yield an all-zero canonical.
        frame = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        result = stage.run([track], {0: frame})

        prop = result[0][0]
        assert prop.target_roi_canonical is not None
        # canonical_size is (W, H) in the track but the warp output from
        # cv2.warpPerspective with dsize=(w, h) returns (h, w, 3).
        assert prop.target_roi_canonical.shape == (80, 180, 3)
        assert prop.target_roi_canonical.dtype == np.uint8
        # It should be a copy, not a view onto the frame.
        prop.target_roi_canonical[:] = 0
        assert result[0][0].target_roi_canonical[0, 0, 0] == 0  # our write hit
        # And the source frame should be untouched.
        assert frame[0, 0, 0] != 0 or frame.mean() > 0  # original noise intact

    def test_skips_track_without_edited_roi(self, propagation_stage):
        track = TextTrack(
            track_id=0, source_text="A", target_text="B",
            source_lang="en", target_lang="es",
            detections={},
            edited_roi=None,
        )
        result = propagation_stage.run([track], {})
        assert result == {}

    def test_uses_frontalized_roi_when_homography_available(self, default_config):
        """When H_to_frontal is set, S4 should warp frame to canonical via warpPerspective."""
        stage = PropagationStage(default_config)
        frame = np.full((200, 300, 3), 100, dtype=np.uint8)

        quad = Quad(points=np.array([
            [50, 50], [250, 50], [250, 150], [50, 150]
        ], dtype=np.float32))
        H = np.eye(3, dtype=np.float64)
        det = TextDetection(
            frame_idx=0, quad=quad, bbox=quad.to_bbox(),
            text="HELLO", ocr_confidence=0.9,
            H_to_frontal=H,
            H_from_frontal=H,
            homography_valid=True,
        )
        edited_roi = np.full((80, 180, 3), 150, dtype=np.uint8)
        track = TextTrack(
            track_id=0, source_text="HELLO", target_text="HOLA",
            source_lang="en", target_lang="es",
            detections={0: det},
            reference_frame_idx=0,

            canonical_size=(180, 80),
            edited_roi=edited_roi,
        )
        with patch("src.stages.s4_propagation.stage.cv2.warpPerspective",
                    wraps=cv2.warpPerspective) as mock_warp:
            result = stage.run([track], {0: frame})

        # Verify warpPerspective was called for the frontalization path.
        # S4 may warp twice when the reference detection is also a target
        # (once to set up LCM ref background, once during the per-detection
        # loop), so we just check that every call used the right H + size.
        assert mock_warp.call_count >= 1
        for call in mock_warp.call_args_list:
            np.testing.assert_array_equal(call.args[1], H)
            assert call.args[2] == (180, 80)

        assert 0 in result
        assert len(result[0]) == 1
        assert result[0][0].roi_image.shape == (80, 180, 3)

    def test_falls_back_to_bbox_when_no_homography(self, default_config, synthetic_frame):
        """Without homography, S4 should fall back to bbox crop."""
        stage = PropagationStage(default_config)
        quad = Quad(points=np.array([
            [200, 150], [440, 150], [440, 250], [200, 250]
        ], dtype=np.float32))
        det = TextDetection(
            frame_idx=0, quad=quad, bbox=quad.to_bbox(),
            text="HELLO", ocr_confidence=0.9,
            # No homography fields set — defaults to None/False
        )
        edited_roi = np.full((100, 240, 3), 128, dtype=np.uint8)
        track = TextTrack(
            track_id=0, source_text="HELLO", target_text="HOLA",
            source_lang="en", target_lang="es",
            detections={0: det},
            reference_frame_idx=0,

            edited_roi=edited_roi,
        )
        result = stage.run([track], {0: synthetic_frame})
        assert 0 in result
        assert len(result[0]) == 1
        assert result[0][0].roi_image.shape[:2] == edited_roi.shape[:2]

    def test_falls_back_when_homography_invalid(self, default_config, synthetic_frame):
        """Even with H_to_frontal set, if homography_valid=False, fall back to bbox."""
        stage = PropagationStage(default_config)
        quad = Quad(points=np.array([
            [200, 150], [440, 150], [440, 250], [200, 250]
        ], dtype=np.float32))
        det = TextDetection(
            frame_idx=0, quad=quad, bbox=quad.to_bbox(),
            text="HELLO", ocr_confidence=0.9,
            H_to_frontal=np.eye(3, dtype=np.float64),
            H_from_frontal=np.eye(3, dtype=np.float64),
            homography_valid=False,  # marked invalid
        )
        edited_roi = np.full((100, 240, 3), 128, dtype=np.uint8)
        track = TextTrack(
            track_id=0, source_text="HELLO", target_text="HOLA",
            source_lang="en", target_lang="es",
            detections={0: det},
            reference_frame_idx=0,

            canonical_size=(240, 100),
            edited_roi=edited_roi,
        )
        result = stage.run([track], {0: synthetic_frame})
        assert 0 in result
        assert len(result[0]) == 1
        # Should still match edited_roi shape via bbox fallback
        assert result[0][0].roi_image.shape[:2] == edited_roi.shape[:2]
