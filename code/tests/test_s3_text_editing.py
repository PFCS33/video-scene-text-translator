"""Tests for Stage 3: Text Editing."""

import numpy as np
import pytest

from src.config import PipelineConfig
from src.data_types import Quad, TextDetection, TextTrack
from src.stages.s3_text_editing import TextEditingStage
from src.models.placeholder_editor import PlaceholderTextEditor


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
            reference_quad=quad,
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

    def test_stage_a_not_implemented(self):
        config = PipelineConfig()
        config.text_editor.backend = "stage_a"
        stage = TextEditingStage(config)
        with pytest.raises(NotImplementedError):
            stage._init_editor()
