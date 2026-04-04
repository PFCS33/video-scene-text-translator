"""TPM training data generation pipeline orchestrator (streaming).

Uses StreamingDetectionStage (S1) and S2 to generate aligned canonical
ROI text images.  Processes video in two passes with bounded memory:

  Pass 1 (streaming): OCR detection -> grouping -> reference selection
  Pass 2 (per-track): optical flow gap-fill -> homography -> ROI extraction
"""

from __future__ import annotations

import json
import logging
import os

import cv2
import numpy as np

from src.config import PipelineConfig
from src.data_types import TextTrack
from src.stages.s1_detection.streaming_stage import StreamingDetectionStage
from src.stages.s2_frontalization import FrontalizationStage
from src.video_io import VideoReader

logger = logging.getLogger(__name__)


class TPMDataGenPipeline:
    """Orchestrates streaming TPM data generation: S1 (streaming) -> S2 -> ROI extraction."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.s1 = StreamingDetectionStage(config)
        self.s2 = FrontalizationStage(config)

    def run(self):
        """Execute the 2-pass streaming pipeline."""
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid config: {'; '.join(errors)}")

        # --- Pass 1: Streaming detection ---
        if not self.config.tpm_data_gen.load_detected_tracks:
            logger.info("=== Pass 1: Streaming Detection ===")
            logger.info("Input video: %s", self.config.input_video)
            with VideoReader(self.config.input_video) as reader:
                tracks = self.s1.run(reader)
            
            if self.config.detection.optical_flow_method == "cotracker":
                min_frames = self.s1.tracker._get_cotracker_min_frames()
                before = len(tracks)
                #tracks = [
                #    t for t in tracks
                #    if t.detections and (max(t.detections) - min(t.detections) + 1) >= min_frames
                #]
                tracks = [
                    t for t in tracks
                    if t.detections and len(t.detections) >= min_frames
                ]
                dropped = before - len(tracks)
                if dropped:
                    logger.info("Dropped %d tracks shorter than %d frames (CoTracker minimum)", dropped, min_frames)

            if not tracks:
                logger.warning("No text tracks found. Quitting.")
                return None
            
            # save detected tracks in case we want to do another run without redoing S1
            if self.config.tpm_data_gen.save_detected_tracks:
                os.makedirs(self.config.output_dir, exist_ok=True)
                track_json_path = os.path.join(self.config.output_dir, "s1_tracks.json")
                with open(track_json_path, "w") as f:
                    json.dump([track.to_json_serializable() for track in tracks], f, indent=2)
                logger.info("Saved %d S1 tracks to %s", len(tracks), track_json_path)
        else:
            # Load detected tracks from JSON instead of re-running S1
            track_json_path = os.path.join(self.config.output_dir, "s1_tracks.json")
            if not os.path.exists(track_json_path):
                logger.error("Track JSON file not found at %s", track_json_path)
                return None
            with open(track_json_path, "r") as f:
                tracks_data = json.load(f)
            tracks = [TextTrack.from_json_serializable(data) for data in tracks_data]
            logger.info("Loaded %d S1 tracks from %s", len(tracks), track_json_path)

        # --- Pass 2: Per-track gap-fill + frontalization + ROI extraction ---
        logger.info("=== Pass 2: Gap-fill + Frontalization + ROI Extraction ===")
        with VideoReader(self.config.input_video) as reader:
            # Fill gaps via streaming optical flow
            logger.info("Filling gaps via optical flow (%s)", self.config.detection.optical_flow_method)
            tracks = self.s1.fill_gaps_streaming(tracks, reader)

            # Compute homographies (no frames needed)
            logger.info("Computing frontalization homographies")
            tracks = self.s2.run(tracks)

            # Extract canonical ROIs per track
            logger.info("Extracting canonical ROIs")
            extraction_info = self._extract_rois(tracks, reader)

        return extraction_info

    def _extract_rois(
        self,
        tracks: list[TextTrack],
        video_reader: VideoReader,
    ) -> list[dict]:
        """Extract and save canonical ROIs for each track, reading frames on demand.

        Processes tracks sorted by frame range start to minimize video seeking.
        """
        extraction_info = []

        # Sort tracks by start frame to read video roughly forward
        sorted_tracks = sorted(
            tracks,
            key=lambda t: min(t.detections.keys()) if t.detections else 0,
        )

        for track in sorted_tracks:
            ref_idx = track.reference_frame_idx
            if ref_idx < 0 or not track.detections:
                logger.warning(
                    "Track %d has no valid reference frame, skipping",
                    track.track_id,
                )
                continue

            canonical_size = track.canonical_size
            if canonical_size is None:
                logger.warning(
                    "Track %d has no canonical size, skipping",
                    track.track_id,
                )
                continue

            track_start = min(track.detections.keys())
            track_end = max(track.detections.keys())

            extraction_info.append({
                "track_id": track.track_id,
                "detected_text": track.source_text,
                "reference_frame_idx": ref_idx,
                "canonical_size": canonical_size,
                "begin_frame_idx": track_start,
                "end_frame_idx": track_end,
            })

            track_output_dir = (
                f"{self.config.output_dir}/track_{track.track_id:02d}_{track.source_text}"
            )
            os.makedirs(track_output_dir, exist_ok=True)

            # Read frames sequentially through the track's range
            extracted_count = 0
            for frame_idx in range(track_start, track_end + 1):
                det = track.detections.get(frame_idx)
                if det is None or not det.homography_valid:
                    continue

                frame = video_reader.read_frame(frame_idx)
                if frame is None:
                    logger.warning(
                        "Track %d frame %d: failed to read, skipping",
                        track.track_id, frame_idx,
                    )
                    continue

                warped_roi = cv2.warpPerspective(
                    frame,
                    det.H_to_frontal,
                    (canonical_size[0], canonical_size[1]),
                    flags=cv2.INTER_LINEAR,
                )

                output_path = f"{track_output_dir}/frame_{frame_idx:06d}.png"
                cv2.imwrite(output_path, warped_roi)
                extracted_count += 1

            logger.info(
                "Extracted %d canonical ROIs for track %d (text: '%s', frames: %d-%d, size: %dx%d) to %s",
                extracted_count, track.track_id, track.source_text,
                track_start, track_end,
                canonical_size[0], canonical_size[1],
                track_output_dir,
            )

        return extraction_info
