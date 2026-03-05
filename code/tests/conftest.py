"""Shared test fixtures: synthetic frames, quads, detections, tracks."""

from __future__ import annotations

import numpy as np
import pytest
import cv2

from src.data_types import BBox, Quad, TextDetection, TextTrack
from src.config import PipelineConfig


@pytest.fixture
def synthetic_frame():
    """640x480 BGR frame with a white rectangle containing black text."""
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    cv2.rectangle(frame, (200, 150), (440, 250), (255, 255, 255), -1)
    cv2.putText(
        frame, "HELLO", (220, 220),
        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2,
    )
    return frame


@pytest.fixture
def synthetic_frame_shifted():
    """Same as synthetic_frame but text shifted right by 20px and down by 10px."""
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    cv2.rectangle(frame, (220, 160), (460, 260), (255, 255, 255), -1)
    cv2.putText(
        frame, "HELLO", (240, 230),
        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2,
    )
    return frame


@pytest.fixture
def rect_quad():
    """A perfect rectangle quad for the text region."""
    return Quad(points=np.array([
        [200, 150], [440, 150], [440, 250], [200, 250]
    ], dtype=np.float32))


@pytest.fixture
def shifted_quad():
    """Quad shifted to match synthetic_frame_shifted."""
    return Quad(points=np.array([
        [220, 160], [460, 160], [460, 260], [220, 260]
    ], dtype=np.float32))


@pytest.fixture
def trapezoid_quad():
    """A trapezoidal (non-frontal) quad."""
    return Quad(points=np.array([
        [210, 155], [430, 150], [440, 250], [200, 248]
    ], dtype=np.float32))


@pytest.fixture
def sample_detection(rect_quad):
    return TextDetection(
        frame_idx=0,
        quad=rect_quad,
        bbox=rect_quad.to_bbox(),
        text="HELLO",
        ocr_confidence=0.95,
        sharpness_score=0.8,
        contrast_score=0.7,
        frontality_score=0.9,
        composite_score=0.85,
    )


@pytest.fixture
def sample_track(sample_detection, rect_quad):
    return TextTrack(
        track_id=0,
        source_text="HELLO",
        target_text="HOLA",
        source_lang="en",
        target_lang="es",
        detections={0: sample_detection},
        reference_frame_idx=0,
        reference_quad=rect_quad,
    )


@pytest.fixture
def default_config():
    """PipelineConfig with sensible test defaults."""
    config = PipelineConfig()
    config.input_video = "test_input.mp4"
    config.output_video = "test_output.mp4"
    return config
