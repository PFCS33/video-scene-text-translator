"""Streaming optical flow gap-filling using VideoReader instead of in-memory frames."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.config import DetectionConfig
from src.data_types import Quad, TextDetection, TextTrack
from src.utils.optical_flow import track_points_farneback, track_points_lucas_kanade

logger = logging.getLogger(__name__)


class StreamingTextTracker:
    """Fills detection gaps via optical flow, reading frames on-demand from a VideoReader."""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self._cotracker_online = None

    def fill_gaps_streaming(
        self,
        tracks: list[TextTrack],
        video_reader,
    ) -> list[TextTrack]:
        """Fill missing frames in each track using optical flow, streaming from video.

        Behaviour depends on config.flow_fill_strategy:
        - "gaps_only": only create synthetic detections for missing frames.
        - "full_propagation": propagate reference quad to every frame in range.

        For CoTracker: falls back to pairwise flow for tracks shorter than
        the online model's minimum window size.
        """
        full = self.config.flow_fill_strategy == "full_propagation"

        for track in tracks:
            if track.reference_frame_idx < 0 or not track.detections:
                continue

            ref_idx = track.reference_frame_idx
            track_start = min(track.detections.keys())
            track_end = max(track.detections.keys())
            track_frame_idxs = list(range(track_start, track_end + 1))
            track_length = len(track_frame_idxs)

            use_cotracker = (
                self.config.optical_flow_method == "cotracker"
                and track_length >= self._get_cotracker_min_frames()
            )

            if use_cotracker:
                tracked_quads = self._track_cotracker_online(
                    track, video_reader, track_frame_idxs, ref_idx, ref_only=full,
                )
            else:
                tracked_quads = self._track_pairwise(
                    track, video_reader, track_frame_idxs, ref_idx, ref_only=full,
                )

            for frame_idx, quad in tracked_quads.items():
                if frame_idx == ref_idx:
                    continue
                if full or frame_idx not in track.detections:
                    track.detections[frame_idx] = TextDetection(
                        frame_idx=frame_idx,
                        quad=quad,
                        bbox=quad.to_bbox(),
                        text=track.source_text,
                        ocr_confidence=0.0,
                    )

        return tracks

    def _get_cotracker_min_frames(self) -> int:
        """Minimum track length to use CoTracker online (step * 2)."""
        if self._cotracker_online is None:
            self._init_cotracker_online()
        return self._cotracker_online.step * 2

    def _init_cotracker_online(self):
        if self._cotracker_online is None:
            from src.utils.cotracker_online import CoTrackerOnlineFlowTracker
            self._cotracker_online = CoTrackerOnlineFlowTracker(self.config)

    def _track_cotracker_online(
        self,
        track: TextTrack,
        video_reader,
        track_frame_idxs: list[int],
        ref_idx: int,
        ref_only: bool = False,
    ) -> dict[int, Quad]:
        """Track using CoTracker online mode."""
        self._init_cotracker_online()

        ref_det = track.detections.get(ref_idx)
        if ref_det is None:
            return {}

        tracked_points = self._cotracker_online.track_points_online(
            video_reader, track_frame_idxs, ref_idx, ref_det.quad.points,
        )
        return {idx: Quad(points=pts) for idx, pts in tracked_points.items()}

    def _track_pairwise(
        self,
        track: TextTrack,
        video_reader,
        track_frame_idxs: list[int],
        ref_idx: int,
        ref_only: bool = False,
    ) -> dict[int, Quad]:
        """Track using pairwise optical flow (Farneback or Lucas-Kanade)."""
        tracked_quads: dict[int, Quad] = {}
        if ref_only:
            ref_det = track.detections.get(ref_idx)
            if ref_det is not None:
                tracked_quads[ref_idx] = ref_det.quad
        else:
            for idx, det in track.detections.items():
                tracked_quads[idx] = det.quad

        # Forward pass: ref_idx -> end
        forward_idxs = [i for i in track_frame_idxs if i >= ref_idx]
        self._propagate_quads_streaming(forward_idxs, video_reader, tracked_quads)

        # Backward pass: ref_idx -> start
        backward_idxs = [i for i in reversed(track_frame_idxs) if i <= ref_idx]
        self._propagate_quads_streaming(backward_idxs, video_reader, tracked_quads)

        return tracked_quads

    def _propagate_quads_streaming(
        self,
        ordered_idxs: list[int],
        video_reader,
        tracked_quads: dict[int, Quad],
    ) -> None:
        """Fill missing quads by propagating optical flow, reading frames on demand.

        Only reads frames when there is actually a gap to fill — avoids
        unnecessary I/O when all quads are already present.
        """
        prev_gray = None
        prev_idx = None

        for i, curr_idx in enumerate(ordered_idxs):
            if curr_idx in tracked_quads:
                # Already have a quad — lazily load frame only if the NEXT
                # index needs flow (i.e., is missing from tracked_quads)
                needs_load = any(
                    ordered_idxs[j] not in tracked_quads
                    for j in range(i + 1, min(i + 2, len(ordered_idxs)))
                )
                if needs_load:
                    frame = video_reader.read_frame(curr_idx)
                    if frame is not None:
                        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else:
                        prev_gray = None
                else:
                    prev_gray = None  # will be loaded lazily if needed later
                prev_idx = curr_idx
                continue

            # Gap: need to compute flow from prev frame
            if prev_idx is None or prev_idx not in tracked_quads or prev_gray is None:
                prev_idx = curr_idx
                continue

            curr_frame = video_reader.read_frame(curr_idx)
            if curr_frame is None:
                prev_idx = curr_idx
                continue

            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            prev_points = tracked_quads[prev_idx].points

            if self.config.optical_flow_method in ("farneback", "cotracker"):
                # CoTracker short-track fallback uses farneback
                new_points = track_points_farneback(
                    prev_gray, curr_gray, prev_points, self.config
                )
            else:
                new_points = track_points_lucas_kanade(
                    prev_gray, curr_gray, prev_points, self.config
                )

            if new_points is not None:
                tracked_quads[curr_idx] = Quad(points=new_points)

            prev_gray = curr_gray
            prev_idx = curr_idx
