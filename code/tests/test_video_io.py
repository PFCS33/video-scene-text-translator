"""Tests for video I/O utilities."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.video_io import VideoReader, VideoWriter


@pytest.fixture
def synthetic_video_path():
    """Create a small 5-frame synthetic video and return its path."""
    path = tempfile.mktemp(suffix=".mp4")
    fps = 30.0
    size = (320, 240)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    for i in range(5):
        frame = np.full((240, 320, 3), i * 50, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


class TestVideoReader:
    def test_open_valid(self, synthetic_video_path):
        reader = VideoReader(synthetic_video_path)
        assert reader.frame_count == 5
        reader.close()

    def test_open_invalid(self):
        with pytest.raises(FileNotFoundError):
            VideoReader("/nonexistent/video.mp4")

    def test_properties(self, synthetic_video_path):
        with VideoReader(synthetic_video_path) as reader:
            assert reader.fps == pytest.approx(30.0, abs=1.0)
            assert reader.frame_size == (320, 240)
            assert reader.frame_count == 5

    def test_read_frame(self, synthetic_video_path):
        with VideoReader(synthetic_video_path) as reader:
            frame = reader.read_frame(0)
            assert frame is not None
            assert frame.shape == (240, 320, 3)

    def test_read_frame_out_of_range(self, synthetic_video_path):
        with VideoReader(synthetic_video_path) as reader:
            frame = reader.read_frame(100)
            assert frame is None

    def test_iter_frames(self, synthetic_video_path):
        with VideoReader(synthetic_video_path) as reader:
            frames = list(reader.iter_frames())
            assert len(frames) == 5
            assert frames[0][0] == 0
            assert frames[4][0] == 4

    def test_context_manager(self, synthetic_video_path):
        with VideoReader(synthetic_video_path) as reader:
            assert reader.frame_count == 5


class TestVideoWriter:
    def test_write_and_read_back(self):
        path = tempfile.mktemp(suffix=".mp4")
        size = (160, 120)
        with VideoWriter(path, 25.0, size) as writer:
            for _ in range(3):
                frame = np.zeros((120, 160, 3), dtype=np.uint8)
                writer.write_frame(frame)

        with VideoReader(path) as reader:
            assert reader.frame_count == 3
            assert reader.frame_size == size
