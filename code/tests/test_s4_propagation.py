"""Tests for Stage 4: Propagation."""

import numpy as np
import pytest

from src.config import PipelineConfig
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
            reference_quad=quad,
            edited_roi=edited_roi,
        )
        frames = {0: synthetic_frame}
        result = propagation_stage.run([track], frames)
        assert 0 in result
        assert len(result[0]) == 1
        assert result[0][0].roi_image.shape[:2] == edited_roi.shape[:2]

    def test_skips_track_without_edited_roi(self, propagation_stage):
        track = TextTrack(
            track_id=0, source_text="A", target_text="B",
            source_lang="en", target_lang="es",
            detections={},
            edited_roi=None,
        )
        result = propagation_stage.run([track], {})
        assert result == {}
