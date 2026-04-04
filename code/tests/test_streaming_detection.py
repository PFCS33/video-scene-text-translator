"""Tests for streaming detection stage and streaming tracker."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config import DetectionConfig, PipelineConfig
from src.data_types import BBox, Quad, TextDetection, TextTrack
from src.stages.s1_detection.streaming_tracker import StreamingTextTracker


@pytest.fixture
def default_config():
    return DetectionConfig()


@pytest.fixture
def rect_quad():
    return Quad(
        points=np.array(
            [[200, 150], [440, 150], [440, 250], [200, 250]], dtype=np.float32
        )
    )


@pytest.fixture
def synthetic_frame():
    """480p frame with a white rectangle where text would be."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[150:250, 200:440] = 255
    return frame


@pytest.fixture
def synthetic_frame_shifted():
    """Same rectangle shifted slightly right."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[150:250, 210:450] = 255
    return frame


class TestStreamingTrackerPairwise:
    """Test streaming tracker with pairwise optical flow."""

    def test_fills_gap_between_detections(
        self, default_config, rect_quad, synthetic_frame, synthetic_frame_shifted
    ):
        """Streaming tracker should fill frame 1 between detections at 0 and 2."""
        tracker = StreamingTextTracker(default_config)
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

        # Mock VideoReader
        video_reader = MagicMock()
        video_reader.read_frame.side_effect = lambda idx: {
            0: synthetic_frame,
            1: synthetic_frame_shifted,
            2: synthetic_frame,
        }.get(idx)

        result = tracker.fill_gaps_streaming([track], video_reader)

        assert 0 in result[0].detections
        assert 1 in result[0].detections
        assert 2 in result[0].detections
        assert result[0].detections[1].ocr_confidence == 0.0

    def test_no_gaps_unchanged(self, default_config, rect_quad):
        """If all frames have detections, nothing should change."""
        tracker = StreamingTextTracker(default_config)
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

        video_reader = MagicMock()
        result = tracker.fill_gaps_streaming([track], video_reader)

        assert len(result[0].detections) == 2
        assert result[0].detections[0].ocr_confidence == 0.9
        assert result[0].detections[1].ocr_confidence == 0.9

    def test_skips_track_without_reference(self, default_config, rect_quad):
        """Tracks with invalid reference should be skipped."""
        tracker = StreamingTextTracker(default_config)
        track = TextTrack(
            track_id=0,
            source_text="HELLO",
            target_text="HOLA",
            source_lang="en",
            target_lang="es",
            detections={},
            reference_frame_idx=-1,
        )

        video_reader = MagicMock()
        result = tracker.fill_gaps_streaming([track], video_reader)

        assert len(result[0].detections) == 0

    def test_cotracker_short_track_uses_pairwise_fallback(
        self, rect_quad, synthetic_frame, synthetic_frame_shifted
    ):
        """Tracks shorter than CoTracker min window fall back to pairwise in the tracker."""
        config = DetectionConfig(optical_flow_method="cotracker")
        tracker = StreamingTextTracker(config)

        # Mock cotracker_online to report step=8 (min_frames=16)
        tracker._cotracker_online = MagicMock()
        tracker._cotracker_online.step = 8

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

        video_reader = MagicMock()
        video_reader.read_frame.side_effect = lambda idx: {
            0: synthetic_frame,
            1: synthetic_frame_shifted,
            2: synthetic_frame,
        }.get(idx)

        result = tracker.fill_gaps_streaming([track], video_reader)

        # Should have used pairwise fallback and filled the gap
        assert 1 in result[0].detections
        assert result[0].detections[1].ocr_confidence == 0.0
        # CoTracker online should NOT have been called
        tracker._cotracker_online.track_points_online.assert_not_called()
