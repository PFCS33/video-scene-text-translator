"""Tests for Stage 3: Text Editing."""

import numpy as np
import pytest

from src.config import PipelineConfig
from src.data_types import Quad, TextDetection, TextTrack
from src.models.placeholder_editor import PlaceholderTextEditor
from src.stages.s3_text_editing import TextEditingStage


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
