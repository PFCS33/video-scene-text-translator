"""Tests for core data structures."""

import numpy as np
import pytest

from src.data_types import BBox, Quad, TextDetection, TextTrack


class TestBBox:
    def test_x2_y2(self):
        bbox = BBox(x=10, y=20, width=100, height=50)
        assert bbox.x2 == 110
        assert bbox.y2 == 70

    def test_area(self):
        bbox = BBox(x=0, y=0, width=100, height=50)
        assert bbox.area() == 5000

    def test_area_zero(self):
        bbox = BBox(x=0, y=0, width=0, height=50)
        assert bbox.area() == 0

    def test_to_slice(self):
        bbox = BBox(x=10, y=20, width=100, height=50)
        row_slice, col_slice = bbox.to_slice()
        assert row_slice == slice(20, 70)
        assert col_slice == slice(10, 110)

    def test_to_slice_indexing(self):
        """Verify to_slice actually works for numpy indexing."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        bbox = BBox(x=10, y=20, width=30, height=40)
        roi = img[bbox.to_slice()]
        assert roi.shape == (40, 30, 3)


class TestQuad:
    def test_to_bbox_rectangle(self, rect_quad):
        bbox = rect_quad.to_bbox()
        assert bbox.x == 200
        assert bbox.y == 150
        assert bbox.width == 240
        assert bbox.height == 100

    def test_to_bbox_trapezoid(self, trapezoid_quad):
        bbox = trapezoid_quad.to_bbox()
        assert bbox.x == 200
        assert bbox.y == 150
        assert bbox.width == 240
        assert bbox.height == 100

    def test_aspect_ratio_wide_rectangle(self):
        quad = Quad(points=np.array([
            [0, 0], [200, 0], [200, 50], [0, 50]
        ], dtype=np.float32))
        ar = quad.aspect_ratio()
        assert ar == pytest.approx(4.0, abs=0.01)

    def test_aspect_ratio_square(self):
        quad = Quad(points=np.array([
            [0, 0], [100, 0], [100, 100], [0, 100]
        ], dtype=np.float32))
        ar = quad.aspect_ratio()
        assert ar == pytest.approx(1.0, abs=0.01)


class TestTextTrack:
    def test_detections_dict_access(self, sample_detection):
        track = TextTrack(
            track_id=0,
            source_text="TEST",
            target_text="PRUEBA",
            source_lang="en",
            target_lang="es",
            detections={0: sample_detection, 5: sample_detection},
        )
        assert 0 in track.detections
        assert 5 in track.detections
        assert 3 not in track.detections
