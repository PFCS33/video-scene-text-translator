"""Tests for Stage 1: Detection.

Tests grouping, scoring, and reference selection logic.
OCR and translation are mocked to avoid external dependencies.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data_types import BBox, Quad, TextDetection, TextTrack
from src.stages.s1_detection import DetectionStage
from src.stages.s1_detection.detector import TextDetector
from src.stages.s1_detection.tracker import TextTracker, bbox_iou


class TestBBoxIoU:
    def test_identical_boxes(self):
        bbox = BBox(x=0, y=0, width=100, height=100)
        assert bbox_iou(bbox, bbox) == pytest.approx(1.0, abs=0.01)

    def test_no_overlap(self):
        a = BBox(x=0, y=0, width=50, height=50)
        b = BBox(x=100, y=100, width=50, height=50)
        assert bbox_iou(a, b) == pytest.approx(0.0, abs=0.01)

    def test_partial_overlap(self):
        a = BBox(x=0, y=0, width=100, height=100)
        b = BBox(x=50, y=50, width=100, height=100)
        # Intersection: 50x50=2500, Union: 10000+10000-2500=17500
        assert bbox_iou(a, b) == pytest.approx(2500 / 17500, abs=0.01)

    def test_contained(self):
        outer = BBox(x=0, y=0, width=100, height=100)
        inner = BBox(x=25, y=25, width=50, height=50)
        # Intersection = inner area = 2500, Union = 10000
        assert bbox_iou(outer, inner) == pytest.approx(2500 / 10000, abs=0.01)


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
        score = stage.detector.compute_composite_score(det)
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
        score = stage.detector.compute_composite_score(det)
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
        tracker = TextTracker(default_config.detection)

        all_dets = {
            0: [self._make_det(0, 200)],
            1: [self._make_det(1, 205)],  # Slight shift, should still match
            2: [self._make_det(2, 210)],
        }
        tracks = tracker.group_detections_into_tracks(
            all_dets, lambda text: "HOLA"
        )
        assert len(tracks) == 1
        assert len(tracks[0].detections) == 3

    def test_different_positions_separate_tracks(self, default_config):
        """Detections at different positions should be separate tracks."""
        tracker = TextTracker(default_config.detection)

        all_dets = {
            0: [self._make_det(0, 50), self._make_det(0, 400)],
        }
        tracks = tracker.group_detections_into_tracks(
            all_dets, lambda text: "TRANSLATED"
        )
        assert len(tracks) == 2

    def test_empty_detections(self, default_config):
        tracker = TextTracker(default_config.detection)
        tracks = tracker.group_detections_into_tracks(
            {}, lambda text: "TRANSLATED"
        )
        assert tracks == []


class TestSelectReferenceFrames:
    def _make_det(self, frame_idx, ocr_conf=0.9, sharpness=0.5,
                  contrast=0.5, frontality=0.9):
        """Helper to create a detection with controllable scores."""
        quad = Quad(points=np.array([
            [0, 0], [100, 0], [100, 50], [0, 50]
        ], dtype=np.float32))
        return TextDetection(
            frame_idx=frame_idx, quad=quad, bbox=quad.to_bbox(),
            text="A", ocr_confidence=ocr_conf,
            sharpness_score=sharpness,
            contrast_score=contrast,
            frontality_score=frontality,
            composite_score=0.5,
        )

    def test_selects_by_contrast_and_frontality(self, default_config):
        """After pre-filters, should pick highest 0.7*contrast + 0.3*frontality."""
        stage = DetectionStage(default_config)
        # Both pass OCR filter (>= 0.7 default)
        det_low = self._make_det(0, ocr_conf=0.8, contrast=0.3, frontality=0.5)
        det_high = self._make_det(5, ocr_conf=0.9, contrast=0.9, frontality=0.8)
        track = TextTrack(
            track_id=0, source_text="A", target_text="B",
            source_lang="en", target_lang="es",
            detections={0: det_low, 5: det_high},
        )
        result = stage.selector.select_reference_frames([track])
        assert result[0].reference_frame_idx == 5

    def test_ocr_filter_excludes_low_confidence(self, default_config):
        """Detection below ref_ocr_min_confidence should be skipped."""
        stage = DetectionStage(default_config)
        # det at frame 0: high contrast but low OCR (filtered out)
        det_bad_ocr = self._make_det(0, ocr_conf=0.3, contrast=0.95, frontality=0.95)
        # det at frame 5: lower contrast but passes OCR filter
        det_ok_ocr = self._make_det(5, ocr_conf=0.8, contrast=0.5, frontality=0.5)
        track = TextTrack(
            track_id=0, source_text="A", target_text="B",
            source_lang="en", target_lang="es",
            detections={0: det_bad_ocr, 5: det_ok_ocr},
        )
        result = stage.selector.select_reference_frames([track])
        assert result[0].reference_frame_idx == 5

    def test_sharpness_top_k_filter(self, default_config):
        """Only top-K sharpest candidates should remain for scoring."""
        default_config.detection.ref_sharpness_top_k = 2
        stage = DetectionStage(default_config)
        # 3 detections all pass OCR, but only top-2 sharpness kept
        det_a = self._make_det(0, ocr_conf=0.9, sharpness=0.9, contrast=0.2, frontality=0.2)
        det_b = self._make_det(1, ocr_conf=0.9, sharpness=0.1, contrast=0.99, frontality=0.99)
        det_c = self._make_det(2, ocr_conf=0.9, sharpness=0.8, contrast=0.5, frontality=0.5)
        track = TextTrack(
            track_id=0, source_text="A", target_text="B",
            source_lang="en", target_lang="es",
            detections={0: det_a, 1: det_b, 2: det_c},
        )
        result = stage.selector.select_reference_frames([track])
        # det_b has highest contrast/frontality but lowest sharpness → filtered
        # Between det_a (0.7*0.2 + 0.3*0.2 = 0.20) and det_c (0.7*0.5 + 0.3*0.5 = 0.50)
        assert result[0].reference_frame_idx == 2

    def test_fallback_when_all_filtered(self, default_config):
        """When no candidates pass pre-filters, fall back to all detections."""
        default_config.detection.ref_ocr_min_confidence = 0.99
        stage = DetectionStage(default_config)
        # No detection meets 0.99 OCR threshold
        det_a = self._make_det(0, ocr_conf=0.5, contrast=0.3, frontality=0.3)
        det_b = self._make_det(1, ocr_conf=0.6, contrast=0.8, frontality=0.8)
        track = TextTrack(
            track_id=0, source_text="A", target_text="B",
            source_lang="en", target_lang="es",
            detections={0: det_a, 1: det_b},
        )
        result = stage.selector.select_reference_frames([track])
        # Falls back to all, det_b has higher contrast+frontality
        assert result[0].reference_frame_idx == 1

    def test_empty_detections_skipped(self, default_config):
        """Track with no detections should be left unchanged."""
        stage = DetectionStage(default_config)
        track = TextTrack(
            track_id=0, source_text="A", target_text="B",
            source_lang="en", target_lang="es",
            detections={},
        )
        result = stage.selector.select_reference_frames([track])
        assert result[0].reference_frame_idx == -1


class TestFillGaps:
    def test_fills_missing_frames(
        self, default_config, synthetic_frame, synthetic_frame_shifted, rect_quad
    ):
        """Optical flow should fill gaps between detected frames."""
        tracker = TextTracker(default_config.detection)
        track = TextTrack(
            track_id=0,
            source_text="HELLO",
            target_text="HOLA",
            source_lang="en",
            target_lang="es",
            detections={
                0: TextDetection(
                    frame_idx=0,
                    quad=rect_quad,
                    bbox=rect_quad.to_bbox(),
                    text="HELLO",
                    ocr_confidence=0.9,
                ),
                2: TextDetection(
                    frame_idx=2,
                    quad=rect_quad,
                    bbox=rect_quad.to_bbox(),
                    text="HELLO",
                    ocr_confidence=0.9,
                ),
            },
            reference_frame_idx=0,
        )
        frames = {
            0: synthetic_frame,
            1: synthetic_frame_shifted,
            2: synthetic_frame,
        }
        result = tracker.fill_gaps([track], frames)
        # Should now have detections for frames 0, 1, and 2
        assert 0 in result[0].detections
        assert 1 in result[0].detections
        assert 2 in result[0].detections
        # Gap-filled frame 1 should have ocr_confidence=0.0
        assert result[0].detections[1].ocr_confidence == 0.0

    def test_no_gaps_unchanged(self, default_config, rect_quad):
        """If all frames have detections, nothing should change."""
        tracker = TextTracker(default_config.detection)
        det0 = TextDetection(
            frame_idx=0,
            quad=rect_quad,
            bbox=rect_quad.to_bbox(),
            text="HELLO",
            ocr_confidence=0.9,
        )
        det1 = TextDetection(
            frame_idx=1,
            quad=rect_quad,
            bbox=rect_quad.to_bbox(),
            text="HELLO",
            ocr_confidence=0.9,
        )
        track = TextTrack(
            track_id=0,
            source_text="HELLO",
            target_text="HOLA",
            source_lang="en",
            target_lang="es",
            detections={0: det0, 1: det1},
            reference_frame_idx=0,
        )
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = tracker.fill_gaps([track], {0: frame, 1: frame})
        assert len(result[0].detections) == 2


class TestTranslateText:
    def test_blank_text_short_circuits_translation(self, default_config):
        stage = DetectionStage(default_config)
        with patch(
            "deep_translator.GoogleTranslator"
        ) as mock_cls:
            mock_cls.side_effect = AssertionError("should not be called")
            result = stage.selector.translate_text("   ")
            assert result == "   "

    def test_deep_translator_google_success(self, default_config):
        stage = DetectionStage(default_config)
        with patch(
            "deep_translator.GoogleTranslator"
        ) as mock_cls:
            mock_cls.return_value.translate.return_value = "PELIGRO"
            result = stage.selector.translate_text("DANGER")
            assert result == "PELIGRO"
            mock_cls.assert_called_once_with(source="en", target="es")

    def test_deep_translator_google_fails_mymemory_fallback(self, default_config):
        stage = DetectionStage(default_config)
        with (
            patch(
                "deep_translator.GoogleTranslator"
            ) as mock_google,
            patch(
                "deep_translator.MyMemoryTranslator"
            ) as mock_mm,
        ):
            mock_google.return_value.translate.side_effect = RuntimeError("blocked")
            mock_mm.return_value.translate.return_value = "PELIGRO"
            result = stage.selector.translate_text("DANGER")
            assert result == "PELIGRO"
            mock_mm.assert_called_once_with(source="en-GB", target="es-ES")

    def test_both_backends_fail_returns_source_text(self, default_config):
        stage = DetectionStage(default_config)
        with (
            patch(
                "deep_translator.GoogleTranslator"
            ) as mock_google,
            patch(
                "deep_translator.MyMemoryTranslator"
            ) as mock_mm,
        ):
            mock_google.return_value.translate.side_effect = RuntimeError("blocked")
            mock_mm.return_value.translate.side_effect = RuntimeError("also down")
            result = stage.selector.translate_text("COFFEE")
            assert result == "COFFEE"


class TestPaddleOCRBackend:
    """Tests for PaddleOCR detection backend (mocked to avoid dependency)."""

    def _make_frame(self, h=200, w=300):
        """Create a non-uniform test frame so sharpness/contrast are computable."""
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Add some variation so quality metrics don't degenerate
        frame[50:150, 50:250] = 200
        frame[80:120, 100:200] = 50
        return frame

    def _make_paddle_result(self, texts, polys, scores):
        """Build a fake PaddleOCR result dict matching the real API shape."""
        return [{"rec_texts": texts, "rec_polys": polys, "rec_scores": scores}]

    def test_detect_paddleocr_basic(self, default_config):
        """PaddleOCR backend should produce TextDetections from mocked results."""
        default_config.detection.ocr_backend = "paddleocr"
        detector = TextDetector(default_config.detection)

        quad_points = np.array(
            [[50, 50], [200, 50], [200, 100], [50, 100]], dtype=np.float32
        )
        mock_ocr = MagicMock()
        mock_ocr.predict.return_value = self._make_paddle_result(
            texts=["HELLO"],
            polys=[quad_points],
            scores=[0.95],
        )
        detector._paddle_ocr = mock_ocr

        frame = self._make_frame()
        dets = detector.detect_text_in_frame(frame, frame_idx=0)

        assert len(dets) == 1
        assert dets[0].text == "HELLO"
        assert dets[0].ocr_confidence == pytest.approx(0.95)
        assert dets[0].frame_idx == 0
        assert dets[0].quad.points.shape == (4, 2)

    def test_detect_paddleocr_filters_low_confidence(self, default_config):
        """Detections below ocr_confidence_threshold should be filtered."""
        default_config.detection.ocr_backend = "paddleocr"
        default_config.detection.ocr_confidence_threshold = 0.5
        detector = TextDetector(default_config.detection)

        quad_points = np.array(
            [[50, 50], [200, 50], [200, 100], [50, 100]], dtype=np.float32
        )
        mock_ocr = MagicMock()
        mock_ocr.predict.return_value = self._make_paddle_result(
            texts=["LOW", "HIGH"],
            polys=[quad_points, quad_points],
            scores=[0.3, 0.8],
        )
        detector._paddle_ocr = mock_ocr

        frame = self._make_frame()
        dets = detector.detect_text_in_frame(frame, frame_idx=0)

        assert len(dets) == 1
        assert dets[0].text == "HIGH"

    def test_detect_paddleocr_filters_small_area(self, default_config):
        """Detections with area below min_text_area should be filtered."""
        default_config.detection.ocr_backend = "paddleocr"
        default_config.detection.min_text_area = 5000
        detector = TextDetector(default_config.detection)

        # Small quad: 20x10 = 200 area (below 5000 threshold)
        small_quad = np.array(
            [[10, 10], [30, 10], [30, 20], [10, 20]], dtype=np.float32
        )
        mock_ocr = MagicMock()
        mock_ocr.predict.return_value = self._make_paddle_result(
            texts=["TINY"],
            polys=[small_quad],
            scores=[0.9],
        )
        detector._paddle_ocr = mock_ocr

        frame = self._make_frame()
        dets = detector.detect_text_in_frame(frame, frame_idx=0)

        assert len(dets) == 0

    def test_detect_paddleocr_multiple_results(self, default_config):
        """Multiple text regions should all be detected."""
        default_config.detection.ocr_backend = "paddleocr"
        detector = TextDetector(default_config.detection)

        quad1 = np.array(
            [[10, 10], [100, 10], [100, 50], [10, 50]], dtype=np.float32
        )
        quad2 = np.array(
            [[10, 100], [100, 100], [100, 150], [10, 150]], dtype=np.float32
        )
        mock_ocr = MagicMock()
        mock_ocr.predict.return_value = self._make_paddle_result(
            texts=["HELLO", "WORLD"],
            polys=[quad1, quad2],
            scores=[0.9, 0.85],
        )
        detector._paddle_ocr = mock_ocr

        frame = self._make_frame()
        dets = detector.detect_text_in_frame(frame, frame_idx=5)

        assert len(dets) == 2
        assert dets[0].text == "HELLO"
        assert dets[1].text == "WORLD"
        assert all(d.frame_idx == 5 for d in dets)

    def test_unknown_backend_raises(self, default_config):
        """Unknown ocr_backend should raise ValueError."""
        default_config.detection.ocr_backend = "unknown_ocr"
        detector = TextDetector(default_config.detection)
        frame = self._make_frame()
        with pytest.raises(ValueError, match="Unknown ocr_backend"):
            detector.detect_text_in_frame(frame, frame_idx=0)
