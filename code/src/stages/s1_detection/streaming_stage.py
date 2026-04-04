"""Streaming Stage 1: Detection & Selection with bounded memory.

Processes video frames via a VideoReader without loading all frames
into memory.  Drop-in replacement for DetectionStage when memory is
a constraint.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from src.config import PipelineConfig
from src.data_types import TextDetection, TextTrack
from src.stages.s1_detection.detector import TextDetector
from src.stages.s1_detection.selector import ReferenceSelector
from src.stages.s1_detection.streaming_tracker import StreamingTextTracker
from src.video_io import VideoReader

logger = logging.getLogger(__name__)


class StreamingDetectionStage:
    """Streaming S1: detect -> group -> translate -> select -> fill gaps.

    Unlike ``DetectionStage``, this reads frames on-demand from a
    ``VideoReader`` and never holds more than a small window in memory.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config.detection
        self.detector = TextDetector(config.detection)
        self.tracker = StreamingTextTracker(config.detection)
        self.selector = ReferenceSelector(config.detection, config.translation)
        self.translation_config = config.translation

    def run(self, video_reader: VideoReader) -> list[TextTrack]:
        """Full streaming S1: detect -> group -> translate -> select -> fill gaps.

        Pass 1 (streaming): iterate all frames for OCR detection, group into
        tracks, and select reference frames.  No frames are retained.

        Gap-filling is deferred — the caller (pipeline) handles it alongside
        ROI extraction in Pass 2 to avoid re-reading the video a third time.

        Returns:
            list[TextTrack] with OCR detections and reference frames selected,
            but gaps NOT yet filled.  Call ``fill_gaps_streaming`` separately.
        """
        logger.info("StreamingS1: Starting detection")
        sample_rate = self.config.frame_sample_rate
        all_detections: dict[int, list[TextDetection]] = {}

        t0_ocr = time.perf_counter()
        frame_count = 0
        for frame_idx, frame in video_reader.iter_frames():
            frame_count += 1
            if frame_idx % sample_rate != 0:
                continue
            dets = self.detector.detect_text_in_frame(frame, frame_idx)
            if dets:
                all_detections[frame_idx] = dets
            logger.debug(
                "StreamingS1: Frame %d -> %d detections",
                frame_idx, len(dets),
            )
        t_ocr = time.perf_counter() - t0_ocr
        logger.info(
            "StreamingS1: OCR detection took %.2fs (%d frames scanned, %d with detections)",
            t_ocr, frame_count, len(all_detections),
        )

        # Group into tracks (no frames needed — operates on detection metadata)
        from src.stages.s1_detection.tracker import TextTracker
        grouping_tracker = TextTracker(self.config)
        tracks = grouping_tracker.group_detections_into_tracks(
            all_detections,
            self.selector.translate_text,
            source_lang=self.translation_config.source_lang,
            target_lang=self.translation_config.target_lang,
        )

        # Select reference frames (no frames needed)
        tracks = self.selector.select_reference_frames(tracks)

        # Update source/target text from reference frame's OCR
        for track in tracks:
            ref_det = track.detections.get(track.reference_frame_idx)
            if ref_det is not None and ref_det.text != track.source_text:
                logger.debug(
                    "Track %d: updating text '%s' -> '%s' (from reference frame)",
                    track.track_id, track.source_text, ref_det.text,
                )
                track.source_text = ref_det.text
                if self.translation_config.target_lang:
                    try:
                        track.target_text = self.selector.translate_text(ref_det.text)
                    except Exception:
                        logger.warning(
                            "Track %d: re-translation failed, keeping original",
                            track.track_id,
                        )
                else:
                    track.target_text = track.source_text

        logger.info("StreamingS1: Found %d text tracks", len(tracks))
        return tracks

    def fill_gaps_streaming(
        self,
        tracks: list[TextTrack],
        video_reader: VideoReader,
    ) -> list[TextTrack]:
        """Fill detection gaps via streaming optical flow.

        Reads frames on-demand from *video_reader* per track.
        """
        t0 = time.perf_counter()
        tracks = self.tracker.fill_gaps_streaming(tracks, video_reader)
        t_flow = time.perf_counter() - t0
        logger.info(
            "StreamingS1: Optical flow (%s) took %.2fs (%d tracks)",
            self.config.optical_flow_method, t_flow, len(tracks),
        )
        return tracks
