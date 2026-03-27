"""Tests for Stage 5: Revert (De-Frontalization + Compositing)."""

import numpy as np
import pytest

from src.data_types import BBox, PropagatedROI, Quad, TextDetection, TextTrack
from src.stages.s5_revert import RevertStage


@pytest.fixture
def revert_stage(default_config):
    return RevertStage(default_config)


class TestWarpRoiToFrame:
    def test_identity_homography(self, revert_stage):
        """Warp with identity H returns bbox-sized output, not full-frame."""
        roi = np.full((100, 200, 3), 128, dtype=np.uint8)
        alpha = np.ones((100, 200), dtype=np.float32)
        quad = Quad(points=np.array([
            [0, 0], [200, 0], [200, 100], [0, 100]
        ], dtype=np.float32))
        prop = PropagatedROI(
            frame_idx=0, track_id=0,
            roi_image=roi, alpha_mask=alpha, target_quad=quad,
        )
        H_from_frontal = np.eye(3)
        result = revert_stage.warp_roi_to_frame(
            prop, H_from_frontal, (480, 640)
        )
        assert result is not None
        warped_roi, warped_alpha, target_bbox = result
        # Output should be bbox-sized (200x100), not full-frame (640x480)
        assert warped_roi.shape == (100, 200, 3)
        assert warped_alpha.shape == (100, 200)
        assert target_bbox == BBox(x=0, y=0, width=200, height=100)

    def test_none_homography_returns_none(self, revert_stage):
        """Passing None as H_from_frontal returns None."""
        roi = np.zeros((50, 50, 3), dtype=np.uint8)
        alpha = np.ones((50, 50), dtype=np.float32)
        quad = Quad(points=np.zeros((4, 2), dtype=np.float32))
        prop = PropagatedROI(
            frame_idx=0, track_id=0,
            roi_image=roi, alpha_mask=alpha, target_quad=quad,
        )
        result = revert_stage.warp_roi_to_frame(prop, None, (50, 50))
        assert result is None

    def test_bounded_warp_size(self, revert_stage):
        """Warped ROI should be bbox-sized, not full-frame-sized."""
        roi = np.full((50, 100, 3), 200, dtype=np.uint8)
        alpha = np.ones((50, 100), dtype=np.float32)
        # Quad placed at (300, 200) in frame — far from origin
        quad = Quad(points=np.array([
            [300, 200], [400, 200], [400, 250], [300, 250]
        ], dtype=np.float32))
        prop = PropagatedROI(
            frame_idx=0, track_id=0,
            roi_image=roi, alpha_mask=alpha, target_quad=quad,
        )
        # Identity homography — ROI should land exactly in quad's bbox
        H_from_frontal = np.eye(3)
        result = revert_stage.warp_roi_to_frame(
            prop, H_from_frontal, (1080, 1920)
        )
        assert result is not None
        warped_roi, warped_alpha, target_bbox = result
        # Key assertion: output is quad-bbox-sized, not 1920x1080
        assert warped_roi.shape == (50, 100, 3)
        assert warped_alpha.shape == (50, 100)
        assert target_bbox == BBox(x=300, y=200, width=100, height=50)

    def test_bbox_clamped_to_frame_bounds(self, revert_stage):
        """Quad partially outside frame should be clamped to frame bounds."""
        roi = np.full((100, 100, 3), 128, dtype=np.uint8)
        alpha = np.ones((100, 100), dtype=np.float32)
        # Quad extends past the frame edge (frame is 200x200)
        quad = Quad(points=np.array([
            [150, 150], [250, 150], [250, 250], [150, 250]
        ], dtype=np.float32))
        prop = PropagatedROI(
            frame_idx=0, track_id=0,
            roi_image=roi, alpha_mask=alpha, target_quad=quad,
        )
        H_from_frontal = np.eye(3)
        result = revert_stage.warp_roi_to_frame(
            prop, H_from_frontal, (200, 200)
        )
        assert result is not None
        warped_roi, warped_alpha, target_bbox = result
        # Clamped: x2=min(250,200)=200, y2=min(250,200)=200
        assert target_bbox == BBox(x=150, y=150, width=50, height=50)
        assert warped_roi.shape == (50, 50, 3)
        assert warped_alpha.shape == (50, 50)

    def test_zero_area_bbox_returns_none(self, revert_stage):
        """If clamped bbox has zero area, return None."""
        roi = np.full((50, 50, 3), 128, dtype=np.uint8)
        alpha = np.ones((50, 50), dtype=np.float32)
        # Quad entirely outside frame bounds
        quad = Quad(points=np.array([
            [300, 300], [350, 300], [350, 350], [300, 350]
        ], dtype=np.float32))
        prop = PropagatedROI(
            frame_idx=0, track_id=0,
            roi_image=roi, alpha_mask=alpha, target_quad=quad,
        )
        H_from_frontal = np.eye(3)
        result = revert_stage.warp_roi_to_frame(
            prop, H_from_frontal, (100, 100)
        )
        assert result is None


class TestCompositeRoiIntoFrame:
    def test_full_alpha_replaces_region(self, revert_stage):
        """Full alpha compositing replaces only the target bbox region."""
        frame = np.full((200, 200, 3), 0, dtype=np.uint8)
        roi = np.full((50, 50, 3), 200, dtype=np.uint8)
        alpha = np.ones((50, 50), dtype=np.float32)
        target_bbox = BBox(x=10, y=20, width=50, height=50)
        result = revert_stage.composite_roi_into_frame(
            frame, roi, alpha, target_bbox
        )
        # Target region should be replaced
        np.testing.assert_array_equal(result[20:70, 10:60], roi)
        # Outside region should be untouched
        assert result[0, 0, 0] == 0

    def test_zero_alpha_preserves_frame(self, revert_stage):
        frame = np.full((200, 200, 3), 100, dtype=np.uint8)
        roi = np.full((50, 50, 3), 200, dtype=np.uint8)
        alpha = np.zeros((50, 50), dtype=np.float32)
        target_bbox = BBox(x=10, y=20, width=50, height=50)
        result = revert_stage.composite_roi_into_frame(
            frame, roi, alpha, target_bbox
        )
        np.testing.assert_array_equal(result, frame)

    def test_half_alpha_blends(self, revert_stage):
        frame = np.full((200, 200, 3), 0, dtype=np.uint8)
        roi = np.full((50, 50, 3), 200, dtype=np.uint8)
        alpha = np.full((50, 50), 0.5, dtype=np.float32)
        target_bbox = BBox(x=10, y=20, width=50, height=50)
        result = revert_stage.composite_roi_into_frame(
            frame, roi, alpha, target_bbox
        )
        region = result[20:70, 10:60]
        expected = 100  # 0 * 0.5 + 200 * 0.5
        assert np.abs(region.mean() - expected) < 2


class TestRevertRun:
    def test_no_rois_returns_original_frames(self, revert_stage):
        frames = {
            0: np.zeros((100, 100, 3), dtype=np.uint8),
            1: np.ones((100, 100, 3), dtype=np.uint8) * 255,
        }
        output = revert_stage.run(frames, {}, [])
        assert len(output) == 2
        np.testing.assert_array_equal(output[0], frames[0])
        np.testing.assert_array_equal(output[1], frames[1])

    def test_run_reads_homography_from_detection(self, revert_stage):
        """run() should read H_from_frontal from TextDetection, not a separate dict."""
        frame = np.full((200, 200, 3), 0, dtype=np.uint8)
        frames = {0: frame}

        roi = np.full((50, 100, 3), 200, dtype=np.uint8)
        alpha = np.ones((50, 100), dtype=np.float32)
        quad = Quad(points=np.array([
            [10, 20], [110, 20], [110, 70], [10, 70]
        ], dtype=np.float32))
        prop = PropagatedROI(
            frame_idx=0, track_id=0,
            roi_image=roi, alpha_mask=alpha, target_quad=quad,
        )

        det = TextDetection(
            frame_idx=0,
            quad=quad,
            bbox=quad.to_bbox(),
            text="HELLO",
            ocr_confidence=0.95,
            H_from_frontal=np.eye(3),
            homography_valid=True,
        )
        track = TextTrack(
            track_id=0,
            source_text="HELLO",
            target_text="HOLA",
            source_lang="en",
            target_lang="es",
            detections={0: det},
            reference_frame_idx=0,

        )

        output = revert_stage.run(frames, {0: [prop]}, [track])
        assert len(output) == 1
        # Region should be modified (ROI composited)
        region = output[0][20:70, 10:110]
        assert region.mean() > 100  # was 0, now blended with 200

    def test_run_skips_invalid_homography(self, revert_stage):
        """run() should skip ROIs where homography_valid is False."""
        frame = np.full((200, 200, 3), 50, dtype=np.uint8)
        frames = {0: frame}

        roi = np.full((50, 100, 3), 200, dtype=np.uint8)
        alpha = np.ones((50, 100), dtype=np.float32)
        quad = Quad(points=np.array([
            [10, 20], [110, 20], [110, 70], [10, 70]
        ], dtype=np.float32))
        prop = PropagatedROI(
            frame_idx=0, track_id=0,
            roi_image=roi, alpha_mask=alpha, target_quad=quad,
        )

        det = TextDetection(
            frame_idx=0,
            quad=quad,
            bbox=quad.to_bbox(),
            text="HELLO",
            ocr_confidence=0.95,
            H_from_frontal=np.eye(3),
            homography_valid=False,  # invalid
        )
        track = TextTrack(
            track_id=0,
            source_text="HELLO",
            target_text="HOLA",
            source_lang="en",
            target_lang="es",
            detections={0: det},
            reference_frame_idx=0,

        )

        output = revert_stage.run(frames, {0: [prop]}, [track])
        # Frame should be unmodified since homography is invalid
        np.testing.assert_array_equal(output[0], frame)

    def test_run_skips_missing_track(self, revert_stage):
        """run() should skip ROIs whose track_id has no matching track."""
        frame = np.full((200, 200, 3), 50, dtype=np.uint8)
        frames = {0: frame}

        roi = np.full((50, 100, 3), 200, dtype=np.uint8)
        alpha = np.ones((50, 100), dtype=np.float32)
        quad = Quad(points=np.array([
            [10, 20], [110, 20], [110, 70], [10, 70]
        ], dtype=np.float32))
        prop = PropagatedROI(
            frame_idx=0, track_id=99,  # no track with this ID
            roi_image=roi, alpha_mask=alpha, target_quad=quad,
        )

        output = revert_stage.run(frames, {0: [prop]}, [])
        np.testing.assert_array_equal(output[0], frame)
