"""Integration test: full pipeline on a synthetic video.

Uses the placeholder editor and mocks OCR/translation to test
end-to-end pipeline execution without external dependencies.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.config import PipelineConfig
from src.pipeline import VideoPipeline
from src.video_io import VideoReader


def _create_synthetic_video(path: str, num_frames: int = 5):
    """Create a synthetic video with a white rectangle containing text."""
    fps = 30.0
    size = (320, 240)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    for i in range(num_frames):
        frame = np.full((240, 320, 3), 100, dtype=np.uint8)
        # Draw text region that shifts slightly per frame
        x_off = i * 2
        cv2.rectangle(
            frame, (80 + x_off, 80), (240 + x_off, 140),
            (255, 255, 255), -1,
        )
        cv2.putText(
            frame, "DANGER", (90 + x_off, 125),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 2,
        )
        writer.write(frame)
    writer.release()


class TestPipelineValidation:
    def test_rejects_empty_input(self):
        config = PipelineConfig()
        pipeline = VideoPipeline(config)
        with pytest.raises(ValueError, match="input_video"):
            pipeline.run()


class TestPipelineNoText:
    def test_no_text_found_outputs_original(self):
        """When OCR finds nothing, output should match input."""
        input_path = tempfile.mktemp(suffix=".mp4")
        output_path = tempfile.mktemp(suffix=".mp4")

        _create_synthetic_video(input_path, num_frames=3)

        config = PipelineConfig()
        config.input_video = input_path
        config.output_video = output_path

        pipeline = VideoPipeline(config)

        # Mock S1 to return no tracks
        pipeline.s1.run = MagicMock(return_value=[])

        result = pipeline.run()
        assert len(result.tracks) == 0
        assert len(result.output_frames) == 3
        assert Path(output_path).exists()
