"""Track text detections across frames and fill gaps via optical flow."""

from __future__ import annotations

import logging
from collections.abc import Callable

import cv2
import numpy as np

from src.config import DetectionConfig
from src.data_types import BBox, Quad, TextDetection, TextTrack
from src.utils.optical_flow import (
    CoTrackerFlowTracker,
    track_points_farneback,
    track_points_lucas_kanade,
)

logger = logging.getLogger(__name__)


def bbox_iou(a: BBox, b: BBox) -> float:
    """Compute IoU (Intersection over Union) between two bounding boxes."""
    x_overlap = max(0, min(a.x2, b.x2) - max(a.x, b.x))
    y_overlap = max(0, min(a.y2, b.y2) - max(a.y, b.y))
    intersection = x_overlap * y_overlap
    union = a.area() + b.area() - intersection
    return intersection / max(union, 1e-6)


class TextTracker:
    """Groups detections into tracks and fills gaps via optical flow."""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self._cotracker: CoTrackerFlowTracker | None = None

    def group_detections_into_tracks(
        self,
        all_detections: dict[int, list[TextDetection]],
        translate_fn: Callable[[str], str],
        source_lang: str = "en",
        target_lang: str = "es",
    ) -> list[TextTrack]:
        """Group detections across frames into tracks by spatial proximity.

        Uses IoU of bounding boxes between consecutive frames
        with greedy matching.

        Args:
            all_detections: frame_idx -> list of detections.
            translate_fn: Callable that translates source text to target text.
            source_lang: Source language code.
            target_lang: Target language code.
        """
        tracks: list[TextTrack] = []
        next_track_id = 0
        active: dict[int, TextDetection] = {}  # track_id -> last detection

        for frame_idx in sorted(all_detections.keys()):
            dets = all_detections[frame_idx]
            matched_det_idxs: set[int] = set()
            matched_track_ids: set[int] = set()

            # Match detections to existing tracks
            for det_i, det in enumerate(dets):
                best_iou = 0.3  # Minimum IoU threshold
                best_track_id = None
                for track_id, last_det in active.items():
                    if track_id in matched_track_ids:
                        continue
                    iou = bbox_iou(det.bbox, last_det.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_track_id = track_id

                if best_track_id is not None:
                    for track in tracks:
                        if track.track_id == best_track_id:
                            track.detections[frame_idx] = det
                            break
                    active[best_track_id] = det
                    matched_det_idxs.add(det_i)
                    matched_track_ids.add(best_track_id)

            # Create new tracks for unmatched detections
            for det_i, det in enumerate(dets):
                if det_i in matched_det_idxs:
                    continue
                if target_lang:
                    try:
                        translated = translate_fn(det.text)
                    except Exception:
                        logger.warning(
                            "Translation failed for text '%s', using source text as target",
                            det.text,
                        )
                        translated = det.text
                else:
                    translated = det.text
                track = TextTrack(
                    track_id=next_track_id,
                    source_text=det.text,
                    target_text=translated,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    detections={frame_idx: det},
                )
                tracks.append(track)
                active[next_track_id] = det
                next_track_id += 1

        return tracks

    def fill_gaps(
        self,
        tracks: list[TextTrack],
        frames: dict[int, np.ndarray],
    ) -> list[TextTrack]:
        """Fill missing frames in each track using optical flow propagation.

        Behaviour depends on config.flow_fill_strategy:
        - "gaps_only": only create synthetic detections for frames that have
          no OCR detection (original behaviour).
        - "full_propagation": propagate the reference quad to every frame,
          overwriting all existing OCR quads with optical-flow-tracked quads.
        """
        all_frame_idxs = sorted(frames.keys())
        full = self.config.flow_fill_strategy == "full_propagation"

        for track in tracks:
            if track.reference_frame_idx < 0 or not track.detections:
                continue

            ref_idx = track.reference_frame_idx
            tracked_quads = self._track_quad_across_frames(
                track, frames, all_frame_idxs, ref_idx,
                ref_only=full,
            )

            for frame_idx, quad in tracked_quads.items():
                if frame_idx == ref_idx:
                    continue  # never overwrite the reference frame itself
                if full or frame_idx not in track.detections:
                    track.detections[frame_idx] = TextDetection(
                        frame_idx=frame_idx,
                        quad=quad,
                        bbox=quad.to_bbox(),
                        text=track.source_text,
                        ocr_confidence=0.0,
                    )

        return tracks

    def _track_quad_across_frames(
        self,
        track: TextTrack,
        frames: dict[int, np.ndarray],
        all_frame_idxs: list[int],
        ref_idx: int,
        ref_only: bool = False,
    ) -> dict[int, Quad]:
        """Track the reference quad to all frames using optical flow.

        Args:
            ref_only: If True, seed propagation from only the reference quad
                (ignoring other detected quads). This produces a smooth,
                purely flow-based trajectory from a single anchor.
        """
        # CoTracker: batch-track all points from the reference frame at once
        if self.config.optical_flow_method == "cotracker":
            ref_det = track.detections.get(ref_idx)
            if ref_det is None:
                return {}
            if self._cotracker is None:
                self._cotracker = CoTrackerFlowTracker(self.config)
            tracked_points = self._cotracker.track_points_batch(
                frames, all_frame_idxs, ref_idx, ref_det.quad.points,
            )
            return {
                idx: Quad(points=pts) for idx, pts in tracked_points.items()
            }

        # Pairwise methods (farneback / lucas_kanade)
        tracked_quads: dict[int, Quad] = {}
        if ref_only:
            ref_det = track.detections.get(ref_idx)
            if ref_det is not None:
                tracked_quads[ref_idx] = ref_det.quad
        else:
            for idx, det in track.detections.items():
                tracked_quads[idx] = det.quad

        # Forward pass: ref_idx -> end
        forward_idxs = [i for i in all_frame_idxs if i >= ref_idx]
        self._propagate_quads(forward_idxs, frames, tracked_quads)

        # Backward pass: ref_idx -> start
        backward_idxs = [i for i in reversed(all_frame_idxs) if i <= ref_idx]
        self._propagate_quads(backward_idxs, frames, tracked_quads)

        return tracked_quads

    def _propagate_quads(
        self,
        ordered_idxs: list[int],
        frames: dict[int, np.ndarray],
        tracked_quads: dict[int, Quad],
    ) -> None:
        """Fill missing quads by propagating optical flow from known quads."""
        for i in range(1, len(ordered_idxs)):
            curr_idx = ordered_idxs[i]
            prev_idx = ordered_idxs[i - 1]

            if curr_idx in tracked_quads:
                continue
            if prev_idx not in tracked_quads:
                continue

            prev_gray = cv2.cvtColor(frames[prev_idx], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[curr_idx], cv2.COLOR_BGR2GRAY)
            prev_points = tracked_quads[prev_idx].points

            if self.config.optical_flow_method == "farneback":
                new_points = track_points_farneback(
                    prev_gray, curr_gray, prev_points, self.config
                )
            else:
                new_points = track_points_lucas_kanade(
                    prev_gray, curr_gray, prev_points, self.config
                )

            if new_points is not None:
                tracked_quads[curr_idx] = Quad(points=new_points)
