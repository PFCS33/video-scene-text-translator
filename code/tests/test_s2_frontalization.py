"""Tests for Stage 2: Frontalization."""

import numpy as np
import pytest

from src.config import PipelineConfig
from src.data_types import BBox, FrameHomography, Quad, TextDetection, TextTrack
from src.stages.s2_frontalization import FrontalizationStage


@pytest.fixture
def frontalization_stage(default_config):
    return FrontalizationStage(default_config)


class TestComputeHomographies:
    def test_reference_frame_gets_identity(self, frontalization_stage):
        quad = Quad(points=np.array([
            [100, 100], [200, 100], [200, 150], [100, 150]
        ], dtype=np.float32))

        track = TextTrack(
            track_id=0, source_text="A", target_text="B",
            source_lang="en", target_lang="es",
            detections={
                0: TextDetection(
                    frame_idx=0, quad=quad, bbox=quad.to_bbox(),
                    text="A", ocr_confidence=0.9,
                ),
            },
            reference_frame_idx=0,
            reference_quad=quad,
        )

        homographies = frontalization_stage.compute_homographies(
            track, {0: quad}
        )
        assert 0 in homographies
        assert homographies[0].is_valid
        np.testing.assert_allclose(
            homographies[0].H_to_ref, np.eye(3), atol=0.01
        )

    def test_shifted_quad_valid_homography(self, frontalization_stage):
        ref_quad = Quad(points=np.array([
            [100, 100], [200, 100], [200, 150], [100, 150]
        ], dtype=np.float32))
        shifted_quad = Quad(points=np.array([
            [110, 110], [210, 110], [210, 160], [110, 160]
        ], dtype=np.float32))

        track = TextTrack(
            track_id=0, source_text="A", target_text="B",
            source_lang="en", target_lang="es",
            detections={
                0: TextDetection(
                    frame_idx=0, quad=ref_quad, bbox=ref_quad.to_bbox(),
                    text="A", ocr_confidence=0.9,
                ),
                1: TextDetection(
                    frame_idx=1, quad=shifted_quad, bbox=shifted_quad.to_bbox(),
                    text="A", ocr_confidence=0.9,
                ),
            },
            reference_frame_idx=0,
            reference_quad=ref_quad,
        )

        homographies = frontalization_stage.compute_homographies(
            track, {0: ref_quad, 1: shifted_quad}
        )
        assert homographies[1].is_valid
        assert homographies[1].H_to_ref is not None
        assert homographies[1].H_from_ref is not None


class TestFrontalizeROI:
    def test_identity_homography_preserves_image(self, frontalization_stage):
        frame = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        hom = FrameHomography(
            frame_idx=0,
            H_to_ref=np.eye(3),
            H_from_ref=np.eye(3),
            is_valid=True,
        )
        ref_bbox = BBox(x=0, y=0, width=300, height=200)
        result = frontalization_stage.frontalize_roi(frame, hom, ref_bbox)
        assert result is not None
        assert result.shape == frame.shape

    def test_invalid_homography_returns_none(self, frontalization_stage):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        hom = FrameHomography(frame_idx=0, is_valid=False)
        ref_bbox = BBox(x=0, y=0, width=100, height=100)
        result = frontalization_stage.frontalize_roi(frame, hom, ref_bbox)
        assert result is None
