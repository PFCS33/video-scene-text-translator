""" TPM training data generation pipeline orchestrator: uses S1 and S2 to generate aligned canonical ROI text images."""

from __future__ import annotations

import logging

import numpy as np

from src.config import PipelineConfig
from src.data_types import PipelineResult, TextTrack
from src.stages.s1_detection import DetectionStage
from src.stages.s2_frontalization import FrontalizationStage
from src.video_io import VideoReader, VideoWriter
import cv2
import os

logger = logging.getLogger(__name__)


class CananicalROIExtractor:
    """Extracts aligned canonical ROIs for each track's reference frame.

    Uses the homographies computed in S2 to warp each reference frame's
    quad to a canonical rectangle, and saves these warped ROIs to disk.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, tracks: list[TextTrack], frames: dict[int, np.ndarray]) -> None:
        """Extract and save canonical ROIs for each track."""

        # Include detected text, frame range, canonical size, etc. in metadata for potential future use
        extraction_info = []

        for track in tracks:
            ref_idx = track.reference_frame_idx
            if ref_idx < 0 or ref_idx not in frames:
                logger.warning(
                    "Track %d has no valid reference frame, skipping",
                    track.track_id,
                )
                continue

            canonical_size = track.canonical_size
            extraction_info.append({
                "track_id": track.track_id,
                "detected_text": track.source_text,
                "reference_frame_idx": ref_idx,
                "canonical_size": canonical_size,
                "begin_frame_idx": min(track.detections.keys()),
                "end_frame_idx": max(track.detections.keys()),
            })

            track_output_dir = f"{self.config.output_dir}/track_{track.track_id:02d}_{track.source_text}"
            os.makedirs(track_output_dir, exist_ok=True)

            # extract all frontalized ROIs and save to disk
            for frame_idx, det in track.detections.items():
                if not det.homography_valid:
                    logger.warning(
                        "Track %d frame %d has invalid homography, skipping",
                        track.track_id, frame_idx,
                    )
                    continue

                frame = frames[frame_idx]
                H_to_frontal = det.H_to_frontal
                warped_roi = cv2.warpPerspective(
                    frame,
                    H_to_frontal,
                    (canonical_size[0], canonical_size[1]),
                    flags=cv2.INTER_LINEAR,
                )

                output_path = (
                    f"{track_output_dir}/frame_{frame_idx:06d}.png"
                )
                cv2.imwrite(output_path, warped_roi)
                #logger.info(
                #    "Saved canonical ROI for track %d frame %d to %s",
                #    track.track_id, frame_idx, output_path,
                #)
            logger.info(
                "Extracted canonical ROIs for track %d (text: '%s', frames: %d-%d, size: %dx%d) to %s",
                track.track_id, track.source_text, min(track.detections.keys()), max(track.detections.keys()), canonical_size[0], canonical_size[1], track_output_dir
            )
        return extraction_info

class TPMDataGenPipeline:
    """Orchestrates the 5-stage video text replacement pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.s1 = DetectionStage(config)
        self.s2 = FrontalizationStage(config)

    def run(self) -> PipelineResult:
        """Execute: S1 -> S2."""
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
        tracks = self.s1.run(frames_list)
        if not tracks:
            logger.warning("No text tracks found. Quitting.")
            return None

        # S2: Frontalization (computes homographies, writes into TextDetection)
        logger.info("=== Stage 2: Frontalization ===")
        tracks = self.s2.run(tracks)

        # Extract canonical ROIs and save to disk
        logger.info("=== Extracting canonical ROIs ===")
        roi_extractor = CananicalROIExtractor(self.config)
        extraction_info = roi_extractor.run(tracks, frames)

        return extraction_info
