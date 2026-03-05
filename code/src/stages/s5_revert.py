"""Stage 5: Revert (De-Frontalization + ROI Compositing).

Applies inverse homography to warp translated ROIs back to each
frame's perspective, then alpha-blends them into the original frames.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.config import PipelineConfig
from src.data_types import FrameHomography, PropagatedROI, TextTrack

logger = logging.getLogger(__name__)


class RevertStage:
    def __init__(self, config: PipelineConfig):
        self.config = config.revert

    def warp_roi_to_frame(
        self,
        propagated_roi: PropagatedROI,
        homography: FrameHomography,
        frame_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Warp a propagated ROI back to the original frame's perspective.

        Uses the inverse homography (ref -> frame) to undo frontalization.

        Args:
            propagated_roi: The color-adapted ROI.
            homography: FrameHomography for this frame.
            frame_shape: (height, width) of the target frame.

        Returns:
            (warped_roi, warped_alpha) or None if homography is invalid.
        """
        if not homography.is_valid or homography.H_from_ref is None:
            return None

        h, w = frame_shape
        warped_roi = cv2.warpPerspective(
            propagated_roi.roi_image,
            homography.H_from_ref,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        warped_alpha = cv2.warpPerspective(
            propagated_roi.alpha_mask,
            homography.H_from_ref,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        return warped_roi, warped_alpha

    def composite_roi_into_frame(
        self,
        frame: np.ndarray,
        warped_roi: np.ndarray,
        warped_alpha: np.ndarray,
    ) -> np.ndarray:
        """Alpha-blend the warped ROI into the frame.

        output = frame * (1 - alpha) + warped_roi * alpha
        """
        alpha_3ch = warped_alpha[:, :, np.newaxis]
        result = (
            frame.astype(np.float32) * (1 - alpha_3ch)
            + warped_roi.astype(np.float32) * alpha_3ch
        ).astype(np.uint8)
        return result

    def run(
        self,
        frames: dict[int, np.ndarray],
        propagated_rois: dict[int, list[PropagatedROI]],
        all_homographies: dict[int, dict[int, FrameHomography]],
        tracks: list[TextTrack],
    ) -> list[np.ndarray]:
        """Apply inverse homography and composite for all frames.

        Returns:
            Output frames in frame_idx order with text replaced.
        """
        logger.info("S5: Reverting and compositing across %d frames", len(frames))
        sorted_idxs = sorted(frames.keys())
        output_frames = []

        for frame_idx in sorted_idxs:
            frame = frames[frame_idx].copy()

            for prop_roi in propagated_rois.get(frame_idx, []):
                track_id = prop_roi.track_id
                homographies = all_homographies.get(track_id, {})
                homography = homographies.get(frame_idx)
                if homography is None:
                    continue

                result = self.warp_roi_to_frame(
                    prop_roi, homography, frame.shape[:2]
                )
                if result is None:
                    continue

                warped_roi, warped_alpha = result
                frame = self.composite_roi_into_frame(
                    frame, warped_roi, warped_alpha
                )

            output_frames.append(frame)

        return output_frames
