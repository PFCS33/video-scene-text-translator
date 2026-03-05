"""Stage 2: Frontalization.

Tracks text quads across frames via optical flow and computes
homography from each frame to the reference frame.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.config import PipelineConfig
from src.data_types import (
    BBox,
    FrameHomography,
    Quad,
    TextDetection,
    TextTrack,
)
from src.utils.geometry import compute_homography
from src.utils.optical_flow import (
    track_points_farneback,
    track_points_lucas_kanade,
)

logger = logging.getLogger(__name__)


class FrontalizationStage:
    def __init__(self, config: PipelineConfig):
        self.config = config.frontalization

    def track_quad_across_frames(
        self,
        track: TextTrack,
        frames: dict[int, np.ndarray],
    ) -> dict[int, Quad]:
        """Track the reference quad to all frames using optical flow.

        For frames with existing detections, uses detected quads directly.
        For gaps, propagates via optical flow bidirectionally from the
        reference frame.
        """
        ref_idx = track.reference_frame_idx
        all_frame_idxs = sorted(frames.keys())

        # Start with known detections
        tracked_quads: dict[int, Quad] = {}
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

    def compute_homographies(
        self,
        track: TextTrack,
        tracked_quads: dict[int, Quad],
    ) -> dict[int, FrameHomography]:
        """Compute homography from each frame's quad to the reference quad."""
        ref_points = track.reference_quad.points
        homographies: dict[int, FrameHomography] = {}

        for frame_idx, quad in tracked_quads.items():
            if frame_idx == track.reference_frame_idx:
                homographies[frame_idx] = FrameHomography(
                    frame_idx=frame_idx,
                    H_to_ref=np.eye(3, dtype=np.float64),
                    H_from_ref=np.eye(3, dtype=np.float64),
                    is_valid=True,
                )
                continue

            H_to_ref, H_from_ref, is_valid = compute_homography(
                src_points=quad.points,
                dst_points=ref_points,
                method=self.config.homography_method,
                ransac_threshold=self.config.ransac_reproj_threshold,
            )
            homographies[frame_idx] = FrameHomography(
                frame_idx=frame_idx,
                H_to_ref=H_to_ref,
                H_from_ref=H_from_ref,
                is_valid=is_valid,
            )

        return homographies

    def frontalize_roi(
        self,
        frame: np.ndarray,
        homography: FrameHomography,
        ref_bbox: BBox,
    ) -> np.ndarray | None:
        """Warp a frame's ROI to match the reference frame's frontal pose."""
        if not homography.is_valid or homography.H_to_ref is None:
            return None
        return cv2.warpPerspective(
            frame,
            homography.H_to_ref,
            (ref_bbox.width, ref_bbox.height),
        )

    def run(
        self,
        tracks: list[TextTrack],
        frames: dict[int, np.ndarray],
    ) -> dict[int, dict[int, FrameHomography]]:
        """Full S2: track quads and compute homographies.

        Returns:
            track_id -> (frame_idx -> FrameHomography)
        """
        logger.info("S2: Computing frontalization for %d tracks", len(tracks))
        all_homographies: dict[int, dict[int, FrameHomography]] = {}

        for track in tracks:
            tracked_quads = self.track_quad_across_frames(track, frames)
            homographies = self.compute_homographies(track, tracked_quads)
            all_homographies[track.track_id] = homographies

            # Store tracked quads for frames that weren't OCR'd
            for frame_idx, quad in tracked_quads.items():
                if frame_idx not in track.detections:
                    track.detections[frame_idx] = TextDetection(
                        frame_idx=frame_idx,
                        quad=quad,
                        bbox=quad.to_bbox(),
                        text=track.source_text,
                        ocr_confidence=0.0,
                    )

        return all_homographies
