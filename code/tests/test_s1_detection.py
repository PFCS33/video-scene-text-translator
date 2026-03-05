"""Tests for Stage 1: Detection.

Tests grouping, scoring, and reference selection logic.
OCR and translation are mocked to avoid external dependencies.
"""

import numpy as np
import pytest

from src.config import PipelineConfig
from src.data_types import BBox, Quad, TextDetection, TextTrack
from src.stages.s1_detection import DetectionStage, _bbox_iou


class TestBBoxIoU:
    def test_identical_boxes(self):
        bbox = BBox(x=0, y=0, width=100, height=100)
        assert _bbox_iou(bbox, bbox) == pytest.approx(1.0, abs=0.01)

    def test_no_overlap(self):
        a = BBox(x=0, y=0, width=50, height=50)
        b = BBox(x=100, y=100, width=50, height=50)
        assert _bbox_iou(a, b) == pytest.approx(0.0, abs=0.01)

    def test_partial_overlap(self):
        a = BBox(x=0, y=0, width=100, height=100)
        b = BBox(x=50, y=50, width=100, height=100)
        # Intersection: 50x50=2500, Union: 10000+10000-2500=17500
        assert _bbox_iou(a, b) == pytest.approx(2500 / 17500, abs=0.01)

    def test_contained(self):
        outer = BBox(x=0, y=0, width=100, height=100)
        inner = BBox(x=25, y=25, width=50, height=50)
        # Intersection = inner area = 2500, Union = 10000
        assert _bbox_iou(outer, inner) == pytest.approx(2500 / 10000, abs=0.01)


class TestCompositeScore:
    def test_weighted_sum(self, default_config):
        stage = DetectionStage(default_config)
        det = TextDetection(
            frame_idx=0,
            quad=Quad(points=np.zeros((4, 2), dtype=np.float32)),
            bbox=BBox(x=0, y=0, width=10, height=10),
            text="test",
            ocr_confidence=1.0,
            sharpness_score=1.0,
            contrast_score=1.0,
            frontality_score=1.0,
        )
        score = stage._compute_composite_score(det)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_zero_scores(self, default_config):
        stage = DetectionStage(default_config)
        det = TextDetection(
            frame_idx=0,
            quad=Quad(points=np.zeros((4, 2), dtype=np.float32)),
            bbox=BBox(x=0, y=0, width=10, height=10),
            text="test",
            ocr_confidence=0.0,
        )
        score = stage._compute_composite_score(det)
        assert score == 0.0


class TestGroupDetections:
    def _make_det(self, frame_idx, x, text="HELLO"):
        quad = Quad(points=np.array([
            [x, 100], [x + 100, 100], [x + 100, 150], [x, 150]
        ], dtype=np.float32))
        return TextDetection(
            frame_idx=frame_idx, quad=quad, bbox=quad.to_bbox(),
            text=text, ocr_confidence=0.9, composite_score=0.8,
        )

    def test_same_position_grouped(self, default_config):
        """Detections at same position across frames should be one track."""
        stage = DetectionStage(default_config)
        # Override translate to avoid external call
        stage.translate_text = lambda text: "HOLA"

        all_dets = {
            0: [self._make_det(0, 200)],
            1: [self._make_det(1, 205)],  # Slight shift, should still match
            2: [self._make_det(2, 210)],
        }
        tracks = stage.group_detections_into_tracks(all_dets)
        assert len(tracks) == 1
        assert len(tracks[0].detections) == 3

    def test_different_positions_separate_tracks(self, default_config):
        """Detections at different positions should be separate tracks."""
        stage = DetectionStage(default_config)
        stage.translate_text = lambda text: "TRANSLATED"

        all_dets = {
            0: [self._make_det(0, 50), self._make_det(0, 400)],
        }
        tracks = stage.group_detections_into_tracks(all_dets)
        assert len(tracks) == 2

    def test_empty_detections(self, default_config):
        stage = DetectionStage(default_config)
        tracks = stage.group_detections_into_tracks({})
        assert tracks == []


class TestSelectReferenceFrames:
    def test_selects_highest_score(self, default_config):
        stage = DetectionStage(default_config)
        quad = Quad(points=np.array([
            [0, 0], [100, 0], [100, 50], [0, 50]
        ], dtype=np.float32))

        det_low = TextDetection(
            frame_idx=0, quad=quad, bbox=quad.to_bbox(),
            text="A", ocr_confidence=0.5, composite_score=0.3,
        )
        det_high = TextDetection(
            frame_idx=5, quad=quad, bbox=quad.to_bbox(),
            text="A", ocr_confidence=0.95, composite_score=0.9,
        )
        track = TextTrack(
            track_id=0, source_text="A", target_text="B",
            source_lang="en", target_lang="es",
            detections={0: det_low, 5: det_high},
        )
        result = stage.select_reference_frames([track])
        assert result[0].reference_frame_idx == 5
