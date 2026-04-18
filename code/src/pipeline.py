"""Pipeline orchestrator: wires S1 through S5 together."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from src.config import PipelineConfig
from src.data_types import PipelineResult
from src.stages.s1_detection import DetectionStage
from src.stages.s2_frontalization import FrontalizationStage
from src.stages.s3_text_editing import TextEditingStage
from src.stages.s4_propagation import PropagationStage
from src.stages.s5_revert import RevertStage
from src.video_io import VideoReader, VideoWriter

logger = logging.getLogger(__name__)


class VideoPipeline:
    """Orchestrates the 5-stage video text replacement pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
        progress_callback: Callable[[str], None] | None = None,
    ):
        self.config = config
        self.progress_callback = progress_callback
        self.s1 = DetectionStage(config)
        self.s2 = FrontalizationStage(config)
        self.s3 = TextEditingStage(config)
        self.s4 = PropagationStage(config)
        self.s5 = RevertStage(config)

    def _emit(self, event: str) -> None:
        """Emit a stage-transition event to the progress callback, if set."""
        if self.progress_callback is not None:
            self.progress_callback(event)

    def run(self) -> PipelineResult:
        """Execute the full pipeline: S1 -> S2 -> S3 -> S4 -> S5."""
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid config: {'; '.join(errors)}")

        # Load all frames into memory.
        # NOTE: For long videos, refactor to sliding-window loading.
        logger.info("Loading video: %s", self.config.input_video)
        with VideoReader(self.config.input_video) as reader:
            fps = reader.fps
            frame_size = reader.frame_size
            frames_list = list(reader.iter_frames())

        frames: dict[int, np.ndarray] = {idx: f for idx, f in frames_list}
        logger.info(
            "Loaded %d frames (%.1f fps, %dx%d)",
            len(frames), fps, frame_size[0], frame_size[1],
        )

        # S1: Detection & Selection
        logger.info("=== Stage 1: Detection & Selection ===")
        self._emit("stage_1_start")
        tracks = self.s1.run(frames_list)
        self._emit("stage_1_done")
        if not tracks:
            logger.warning("No text tracks found. Outputting original video.")
            output_frames = [frames[i] for i in sorted(frames.keys())]
            with VideoWriter(self.config.output_video, fps, frame_size) as writer:
                for frame in output_frames:
                    writer.write_frame(frame)
            return PipelineResult(
                tracks=[],
                output_frames=output_frames,
                fps=fps,
                frame_size=frame_size,
            )

        # S2: Frontalization (computes homographies, writes into TextDetection)
        logger.info("=== Stage 2: Frontalization ===")
        self._emit("stage_2_start")
        tracks = self.s2.run(tracks)
        self._emit("stage_2_done")

        # S3: Text Editing
        logger.info("=== Stage 3: Text Editing ===")
        self._emit("stage_3_start")
        tracks = self.s3.run(tracks, frames)
        self._emit("stage_3_done")

        # S4: Propagation
        logger.info("=== Stage 4: Propagation ===")
        self._emit("stage_4_start")
        propagated_rois = self.s4.run(tracks, frames)
        self._emit("stage_4_done")

        # S5: Revert
        logger.info("=== Stage 5: Revert ===")
        self._emit("stage_5_start")
        output_frames = self.s5.run(frames, propagated_rois, tracks)
        self._emit("stage_5_done")

        # Write output video
        logger.info("Writing output video: %s", self.config.output_video)
        with VideoWriter(self.config.output_video, fps, frame_size) as writer:
            for frame in output_frames:
                writer.write_frame(frame)

        result = PipelineResult(
            tracks=tracks,
            output_frames=output_frames,
            fps=fps,
            frame_size=frame_size,
        )
        logger.info(
            "Pipeline complete. %d tracks processed, %d frames output.",
            len(tracks), len(output_frames),
        )
        return result
