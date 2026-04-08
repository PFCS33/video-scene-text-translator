"""Tests for Stage 3: Text Editing."""

import numpy as np
import pytest

from src.config import PipelineConfig
from src.data_types import Quad, TextDetection, TextTrack
from src.models.placeholder_editor import PlaceholderTextEditor
from src.stages.s3_text_editing import (
    TextEditingStage,
    _clamp_expansion_ratio,
    _expanded_warp,
)


class TestPlaceholderEditor:
    def test_output_shape_matches_input(self):
        editor = PlaceholderTextEditor()
        roi = np.full((50, 150, 3), 200, dtype=np.uint8)
        result = editor.edit_text(roi, "TEST")
        assert result.shape == roi.shape
        assert result.dtype == np.uint8

    def test_output_differs_from_input(self):
        editor = PlaceholderTextEditor()
        roi = np.full((50, 150, 3), 200, dtype=np.uint8)
        result = editor.edit_text(roi, "HELLO")
        # The result should not be identical to input (text was drawn)
        assert not np.array_equal(result, roi)

    def test_tiny_roi_does_not_crash(self):
        editor = PlaceholderTextEditor()
        roi = np.zeros((3, 3, 3), dtype=np.uint8)
        result = editor.edit_text(roi, "X")
        assert result.shape == (3, 3, 3)


class TestTextEditingStage:
    def test_edits_reference_roi(self, default_config, synthetic_frame):
        stage = TextEditingStage(default_config)
        quad = Quad(points=np.array([
            [200, 150], [440, 150], [440, 250], [200, 250]
        ], dtype=np.float32))
        det = TextDetection(
            frame_idx=0, quad=quad, bbox=quad.to_bbox(),
            text="HELLO", ocr_confidence=0.9,
        )
        track = TextTrack(
            track_id=0, source_text="HELLO", target_text="HOLA",
            source_lang="en", target_lang="es",
            detections={0: det},
            reference_frame_idx=0,
        )
        frames = {0: synthetic_frame}
        result = stage.run([track], frames)
        assert result[0].edited_roi is not None
        assert result[0].edited_roi.shape[2] == 3

    def test_unknown_backend_raises(self):
        config = PipelineConfig()
        config.text_editor.backend = "nonexistent"
        stage = TextEditingStage(config)
        with pytest.raises(ValueError, match="Unknown text editor backend"):
            stage._init_editor()

    def test_uses_frontalized_roi_when_homography_available(self, default_config):
        """When H_to_frontal is set, S3 should warp to canonical before editing."""
        stage = TextEditingStage(default_config)
        # Create a 200x300 frame with known pattern
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        frame[50:150, 100:250, :] = 200  # bright rectangle in center

        quad = Quad(points=np.array([
            [100, 50], [250, 50], [250, 150], [100, 150]
        ], dtype=np.float32))
        det = TextDetection(
            frame_idx=0, quad=quad, bbox=quad.to_bbox(),
            text="HELLO", ocr_confidence=0.9,
            H_to_frontal=np.eye(3, dtype=np.float64),  # identity for simplicity
            H_from_frontal=np.eye(3, dtype=np.float64),
            homography_valid=True,
        )
        # canonical_size differs from bbox (150x100) to prove warp path is used
        # bbox would give 150x100, canonical_size is 180x120
        track = TextTrack(
            track_id=0, source_text="HELLO", target_text="HOLA",
            source_lang="en", target_lang="es",
            detections={0: det},
            reference_frame_idx=0,
            canonical_size=(180, 120),  # width, height — differs from bbox dims
        )
        result = stage.run([track], {0: frame})
        assert result[0].edited_roi is not None
        # With identity H and canonical_size (180, 120), the ROI should be 120x180x3
        # This differs from bbox crop which would be 100x150x3
        assert result[0].edited_roi.shape == (120, 180, 3)

    def test_stage_a_not_implemented(self):
        config = PipelineConfig()
        config.text_editor.backend = "stage_a"
        stage = TextEditingStage(config)
        with pytest.raises(NotImplementedError):
            stage._init_editor()

    def test_expanded_roi_crops_back_to_canonical(self, default_config):
        """With expansion enabled, edited_roi should match canonical_size."""
        default_config.text_editor.roi_context_expansion = 0.3
        stage = TextEditingStage(default_config)

        frame = np.full((300, 500, 3), 128, dtype=np.uint8)
        quad = Quad(points=np.array([
            [100, 50], [300, 50], [300, 150], [100, 150]
        ], dtype=np.float32))
        det = TextDetection(
            frame_idx=0, quad=quad, bbox=quad.to_bbox(),
            text="HELLO", ocr_confidence=0.9,
            H_to_frontal=np.eye(3, dtype=np.float64),
            H_from_frontal=np.eye(3, dtype=np.float64),
            homography_valid=True,
        )
        track = TextTrack(
            track_id=0, source_text="HELLO", target_text="HOLA",
            source_lang="en", target_lang="es",
            detections={0: det}, reference_frame_idx=0,
            canonical_size=(200, 100),
        )
        result = stage.run([track], {0: frame})
        # edited_roi should be canonical size (100, 200) not expanded
        assert result[0].edited_roi is not None
        assert result[0].edited_roi.shape == (100, 200, 3)

    def test_no_expansion_when_ratio_zero(self, default_config):
        """With expansion=0.0, behavior is identical to before."""
        default_config.text_editor.roi_context_expansion = 0.0
        stage = TextEditingStage(default_config)

        frame = np.full((300, 500, 3), 128, dtype=np.uint8)
        quad = Quad(points=np.array([
            [100, 50], [300, 50], [300, 150], [100, 150]
        ], dtype=np.float32))
        det = TextDetection(
            frame_idx=0, quad=quad, bbox=quad.to_bbox(),
            text="TEST", ocr_confidence=0.9,
            H_to_frontal=np.eye(3, dtype=np.float64),
            H_from_frontal=np.eye(3, dtype=np.float64),
            homography_valid=True,
        )
        track = TextTrack(
            track_id=0, source_text="TEST", target_text="PRUEBA",
            source_lang="en", target_lang="es",
            detections={0: det}, reference_frame_idx=0,
            canonical_size=(200, 100),
        )
        result = stage.run([track], {0: frame})
        assert result[0].edited_roi is not None
        assert result[0].edited_roi.shape == (100, 200, 3)


class TestClampExpansionRatio:
    def test_zero_ratio_returns_zero(self):
        assert _clamp_expansion_ratio(0.0, 200, 100) == 0.0

    def test_negative_ratio_returns_zero(self):
        assert _clamp_expansion_ratio(-0.5, 200, 100) == 0.0

    def test_small_roi_allows_full_ratio(self):
        # 200x100, max=200, ratio=0.3 → expanded max = 200*1.6 = 320 < 1024
        result = _clamp_expansion_ratio(0.3, 200, 100)
        assert result == pytest.approx(0.3)

    def test_large_roi_caps_ratio(self):
        # 800x400, max=800, ratio=0.3 → expanded max = 800*1.6 = 1280 > 1024
        # max_ratio = (1024/800 - 1) / 2 = 0.14
        result = _clamp_expansion_ratio(0.3, 800, 400)
        assert result == pytest.approx((1024 / 800 - 1) / 2)
        assert result < 0.3

    def test_at_max_returns_zero(self):
        # 1024x500 — already at max, no expansion possible
        assert _clamp_expansion_ratio(0.3, 1024, 500) == 0.0

    def test_above_max_returns_zero(self):
        assert _clamp_expansion_ratio(0.3, 2000, 500) == 0.0


class TestExpandedWarp:
    def test_output_size_includes_margins(self):
        frame = np.zeros((300, 500, 3), dtype=np.uint8)
        H = np.eye(3, dtype=np.float64)
        roi, edit_region = _expanded_warp(frame, H, 200, 100, 0.3)
        margin_x = round(200 * 0.3)  # 60
        margin_y = round(100 * 0.3)  # 30
        assert roi.shape == (100 + 2 * margin_y, 200 + 2 * margin_x, 3)

    def test_edit_region_matches_canonical_area(self):
        frame = np.zeros((300, 500, 3), dtype=np.uint8)
        H = np.eye(3, dtype=np.float64)
        roi, edit_region = _expanded_warp(frame, H, 200, 100, 0.3)
        margin_x = round(200 * 0.3)
        margin_y = round(100 * 0.3)
        et, eb, el, er = edit_region
        assert et == margin_y
        assert eb == margin_y + 100
        assert el == margin_x
        assert er == margin_x + 200

    def test_scene_context_is_real_pixels(self):
        """Margins should contain real scene pixels, not zeros."""
        frame = np.full((300, 500, 3), 42, dtype=np.uint8)
        H = np.eye(3, dtype=np.float64)
        roi, (et, eb, el, er) = _expanded_warp(frame, H, 200, 100, 0.3)
        # Top margin should have real pixel value (42)
        if et > 0:
            assert np.all(roi[0, el:er] == 42)
