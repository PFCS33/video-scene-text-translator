"""
Track text detections across frames and fill gaps via optical flow.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import cv2
import numpy as np

from src.config import DetectionConfig
from src.data_types import BBox, Quad, TextDetection, TextTrack

# Kalman
from src.utils.kalman import QuadKalmanFilter
from src.utils.optical_flow import (
    CoTrackerFlowTracker,
    track_points_farneback,
    track_points_lucas_kanade,
)

logger = logging.getLogger(__name__)


def bbox_iou(a: BBox, b: BBox) -> float:
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

        # Kalman filters per track
        self._kalman_filters: dict[int, QuadKalmanFilter] = {}

        # EMA state per track
        self._ema_states: dict[int, np.ndarray] = {}

        # EMA smoothing factor (tunable via config)
        self._ema_alpha = getattr(config, "ema_alpha", 0.6)

    # -------------------------
    # Kalman helpers
    # -------------------------
    def _get_kalman(self, track_id: int) -> QuadKalmanFilter:
        if track_id not in self._kalman_filters:
            self._kalman_filters[track_id] = QuadKalmanFilter()
        return self._kalman_filters[track_id]

    # -------------------------
    # EMA helper
    # -------------------------
    def _apply_ema(self, track_id: int, points: np.ndarray) -> np.ndarray:
        prev = self._ema_states.get(track_id)

        if prev is None:
            smoothed = points
        else:
            smoothed = self._ema_alpha * prev + (1.0 - self._ema_alpha) * points

        self._ema_states[track_id] = smoothed
        return smoothed

    # -------------------------
    # Text normalization & similarity
    # -------------------------
    def _normalize_text(self, text: str) -> str:
        return text.lower().replace(" ", "")

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        t1 = self._normalize_text(text1)
        t2 = self._normalize_text(text2)

        if not t1 or not t2:
            return 0.0

        matches = sum(c1 == c2 for c1, c2 in zip(t1, t2))
        max_len = max(len(t1), len(t2))
        return matches / max_len

    # -------------------------
    # Smoothing utilities (light)
    # -------------------------
    def _to_np(self, pts):
        return np.asarray(pts, dtype=np.float32)

    def _smooth_quad(self, prev: Quad, new: Quad, alpha: float = 0.3) -> Quad:
        prev_pts = self._to_np(prev.points)
        new_pts = self._to_np(new.points)

        smoothed = alpha * prev_pts + (1.0 - alpha) * new_pts
        return Quad(points=smoothed)

    def _smooth_or_update_detection(
        self,
        track: TextTrack,
        frame_idx: int,
        new_det: TextDetection,
    ) -> TextDetection:
        if frame_idx in track.detections:
            existing = track.detections[frame_idx]
            smoothed_quad = self._smooth_quad(existing.quad, new_det.quad)

            return TextDetection(
                frame_idx=frame_idx,
                quad=smoothed_quad,
                bbox=smoothed_quad.to_bbox(),
                text=new_det.text,
                ocr_confidence=new_det.ocr_confidence,
                sharpness_score=new_det.sharpness_score,
                contrast_score=new_det.contrast_score,
                frontality_score=new_det.frontality_score,
                composite_score=new_det.composite_score,
            )

        return new_det

    # -------------------------
    # Tracking
    # -------------------------
    def group_detections_into_tracks(
        self,
        all_detections: dict[int, list[TextDetection]],
        translate_fn: Callable[[str], str],
        source_lang: str = "en",
        target_lang: str = "es",
    ) -> list[TextTrack]:

        tracks: list[TextTrack] = []
        next_track_id = 0
        active: dict[int, TextDetection] = {}

        IOU_THRESHOLD = 0.2

        for frame_idx in sorted(all_detections.keys()):
            dets = all_detections[frame_idx]

            matched_det_idxs: set[int] = set()
            matched_track_ids: set[int] = set()

            for det_i, det in enumerate(dets):

                best_score = 0.0
                best_track_id = None

                for track_id, last_det in active.items():

                    if track_id in matched_track_ids:
                        continue

                    if abs(last_det.frame_idx - frame_idx) > self.config.track_break_threshold:
                        continue

                    iou = bbox_iou(det.bbox, last_det.bbox)
                    if iou < IOU_THRESHOLD:
                        continue

                    text_similarity = self._compute_text_similarity(det.text, last_det.text)
                    temporal_score = 1.0 / (1.0 + abs(last_det.frame_idx - frame_idx))

                    center_distance = np.linalg.norm(
                        np.array([
                            (det.bbox.x + det.bbox.x2) / 2.0,
                            (det.bbox.y + det.bbox.y2) / 2.0
                        ]) - np.array([
                            (last_det.bbox.x + last_det.bbox.x2) / 2.0,
                            (last_det.bbox.y + last_det.bbox.y2) / 2.0
                        ])
                    )

                    distance_penalty = 1.0 / (1.0 + center_distance)

                    score = (
                        0.50 * iou +
                        0.20 * text_similarity +
                        0.20 * temporal_score +
                        0.10 * distance_penalty
                    )

                    if score > best_score:
                        best_score = score
                        best_track_id = track_id

                if best_track_id is not None:
                    for track in tracks:
                        if track.track_id == best_track_id:
                            smoothed_det = self._smooth_or_update_detection(
                                track,
                                frame_idx,
                                det,
                            )
                            track.detections[frame_idx] = smoothed_det
                            break

                    active[best_track_id] = det
                    matched_det_idxs.add(det_i)
                    matched_track_ids.add(best_track_id)

            # Create new tracks
            for det_i, det in enumerate(dets):
                if det_i in matched_det_idxs:
                    continue

                if target_lang:
                    try:
                        translated = translate_fn(det.text)
                    except Exception:
                        logger.warning(
                            "Translation failed for text '%s', using source text",
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

                logger.info(
                    f"Created track {track.track_id} for text '{track.source_text}' at frame {frame_idx}"
                )

                tracks.append(track)
                active[next_track_id] = det
                next_track_id += 1

        return tracks

    # -------------------------
    # Optical flow gap filling
    # -------------------------
    def fill_gaps(
        self,
        tracks: list[TextTrack],
        frames: dict[int, np.ndarray],
    ) -> list[TextTrack]:

        all_frame_idxs = sorted(frames.keys())
        full = self.config.flow_fill_strategy == "full_propagation"

        for track in tracks:
            if track.reference_frame_idx < 0 or not track.detections:
                continue

            ref_idx = track.reference_frame_idx

            tracked_quads = self._track_quad_across_frames(
                track,
                frames,
                all_frame_idxs,
                ref_idx,
                ref_only=full,
            )

            for frame_idx, quad in tracked_quads.items():
                if frame_idx == ref_idx:
                    continue

                existing = track.detections.get(frame_idx)

                if existing is not None:
                    quad = self._smooth_quad(existing.quad, quad, alpha=0.3)

                quad_points = self._to_np(quad.points)
                filtered_points = quad_points

                if self.config.use_kalman_smoothing:
                    kalman = self._get_kalman(track.track_id)
                    filtered_points = kalman.update(filtered_points)

                if self.config.use_ema_smoothing:
                    filtered_points = self._apply_ema(track.track_id, filtered_points)

                quad = Quad(points=filtered_points)

                track.detections[frame_idx] = TextDetection(
                    frame_idx=frame_idx,
                    quad=quad,
                    bbox=quad.to_bbox(),
                    text=track.source_text,
                    ocr_confidence=0.0,
                )

        return tracks

    # -------------------------
    # Optical flow tracking
    # -------------------------
    def _track_quad_across_frames(
        self,
        track: TextTrack,
        frames: dict[int, np.ndarray],
        all_frame_idxs: list[int],
        ref_idx: int,
        ref_only: bool = False,
    ) -> dict[int, Quad]:

        track_frame_idx_start = min(track.detections.keys())
        track_frame_idx_end = max(track.detections.keys())

        if self.config.optical_flow_method == "cotracker":
            ref_det = track.detections.get(ref_idx)
            if ref_det is None:
                return {}

            if self._cotracker is None:
                self._cotracker = CoTrackerFlowTracker(self.config)

            target_frame_idxs = [
                i for i in all_frame_idxs
                if track_frame_idx_start <= i <= track_frame_idx_end
            ]

            ref_points = np.asarray(ref_det.quad.points, dtype=np.float32)

            tracked_points = self._cotracker.track_points_batch(
                frames,
                target_frame_idxs,
                ref_idx,
                ref_points,
            )

            return {
                idx: Quad(points=pts.tolist() if isinstance(pts, np.ndarray) else pts)
                for idx, pts in tracked_points.items()
            }

        tracked_quads: dict[int, Quad] = {}

        if ref_only:
            ref_det = track.detections.get(ref_idx)
            if ref_det is not None:
                tracked_quads[ref_idx] = ref_det.quad
        else:
            for idx, det in track.detections.items():
                tracked_quads[idx] = det.quad

        forward_idxs = [
            i for i in all_frame_idxs
            if i >= ref_idx and track_frame_idx_start <= i <= track_frame_idx_end
        ]
        self._propagate_quads(forward_idxs, frames, tracked_quads)

        backward_idxs = [
            i for i in reversed(all_frame_idxs)
            if i <= ref_idx and track_frame_idx_start <= i <= track_frame_idx_end
        ]
        self._propagate_quads(backward_idxs, frames, tracked_quads)

        return tracked_quads

    def _propagate_quads(
        self,
        ordered_idxs: list[int],
        frames: dict[int, np.ndarray],
        tracked_quads: dict[int, Quad],
    ) -> None:

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
