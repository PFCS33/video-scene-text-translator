"""Stage 5: Revert (De-Frontalization + ROI Compositing).

Applies inverse homography to warp translated ROIs back to each
frame's perspective, then alpha-blends them into the original frames.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.config import PipelineConfig
from src.data_types import BBox, PropagatedROI, TextTrack

logger = logging.getLogger(__name__)


class RevertStage:
    def __init__(self, config: PipelineConfig):
        self.config = config.revert

    def warp_roi_to_frame(
        self,
        propagated_roi: PropagatedROI,
        H_from_frontal: np.ndarray | None,
        frame_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, BBox] | None:
        """Warp a propagated ROI back to the original frame's perspective.

        Uses the inverse homography (frontal -> frame) to undo frontalization,
        warping only to the target quad's bounding box region instead of the
        full frame.

        Args:
            propagated_roi: The color-adapted ROI.
            H_from_frontal: 3x3 homography matrix (frontal -> frame), or None.
            frame_shape: (height, width) of the target frame.

        Returns:
            (warped_roi, warped_alpha, target_bbox) or None if homography
            is None or the target bbox has zero area after clamping.
        """
        if H_from_frontal is None:
            return None

        frame_h, frame_w = frame_shape

        # Compute target bbox from the detection's quad
        target_bbox = propagated_roi.target_quad.to_bbox()

        # Clamp bbox to frame bounds
        x1 = max(target_bbox.x, 0)
        y1 = max(target_bbox.y, 0)
        x2 = min(target_bbox.x2, frame_w)
        y2 = min(target_bbox.y2, frame_h)
        clamped_w = x2 - x1
        clamped_h = y2 - y1

        if clamped_w <= 0 or clamped_h <= 0:
            return None

        target_bbox = BBox(x=x1, y=y1, width=clamped_w, height=clamped_h)

        # Translation matrix to offset coordinates into bbox-local space
        T = np.array([
            [1, 0, -target_bbox.x],
            [0, 1, -target_bbox.y],
            [0, 0, 1],
        ], dtype=np.float64)

        # Coordinate chain: canonical space -> frame space (H_from_frontal)
        # -> bbox-local coords (T shifts origin to bbox top-left)
        H_adjusted = T @ H_from_frontal

        warped_roi = cv2.warpPerspective(
            propagated_roi.roi_image,
            H_adjusted,
            (target_bbox.width, target_bbox.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        warped_alpha = cv2.warpPerspective(
            propagated_roi.alpha_mask,
            H_adjusted,
            (target_bbox.width, target_bbox.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        return warped_roi, warped_alpha, target_bbox

    def composite_roi_into_frame(
        self,
        frame: np.ndarray,
        warped_roi: np.ndarray,
        warped_alpha: np.ndarray,
        target_bbox: BBox,
    ) -> np.ndarray:
        """Alpha-blend the warped ROI into the frame at the target bbox region.

        output[bbox_region] = frame * (1 - alpha) + warped_roi * alpha
        """
        roi_slice = target_bbox.to_slice()
        region = frame[roi_slice].copy()
        alpha_3ch = warped_alpha[:, :, np.newaxis]
        blended = (
            region.astype(np.float32) * (1 - alpha_3ch)
            + warped_roi.astype(np.float32) * alpha_3ch
        ).astype(np.uint8)
        frame[roi_slice] = blended
        return frame

    def run(
        self,
        frames: dict[int, np.ndarray],
        propagated_rois: dict[int, list[PropagatedROI]],
        tracks: list[TextTrack],
    ) -> list[np.ndarray]:
        """Apply inverse homography and composite for all frames.

        Reads H_from_frontal from TextDetection on each track, instead of
        a separate all_homographies dict.

        Returns:
            Output frames in frame_idx order with text replaced.
        """
        logger.info("S5: Reverting and compositing across %d frames", len(frames))
        sorted_idxs = sorted(frames.keys())

        # Build lookup for tracks by track_id
        tracks_by_id = {t.track_id: t for t in tracks}

        output_frames = []

        for frame_idx in sorted_idxs:
            frame = frames[frame_idx].copy()

            for prop_roi in propagated_rois.get(frame_idx, []):
                track = tracks_by_id.get(prop_roi.track_id)
                if track is None:
                    continue

                det = track.detections.get(frame_idx)
                if det is None or not det.homography_valid or det.H_from_frontal is None:
                    continue

                result = self.warp_roi_to_frame(
                    prop_roi, det.H_from_frontal, frame.shape[:2]
                )
                if result is None:
                    continue

                warped_roi, warped_alpha, target_bbox = result
                frame = self.composite_roi_into_frame(
                    frame, warped_roi, warped_alpha, target_bbox
                )

            output_frames.append(frame)

        return output_frames
