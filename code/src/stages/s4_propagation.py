"""Stage 4: Propagation.

Adapts the translated reference ROI to match each frame's lighting
via histogram matching, and generates feathered alpha masks.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.config import PipelineConfig
from src.data_types import PropagatedROI, TextTrack
from src.utils.image_processing import match_histogram_luminance

logger = logging.getLogger(__name__)


class PropagationStage:
    def __init__(self, config: PipelineConfig):
        self.config = config.propagation

    def propagate_to_frame(
        self,
        edited_roi: np.ndarray,
        target_frame_roi: np.ndarray,
    ) -> np.ndarray:
        """Adapt edited reference ROI to match a target frame's appearance.

        Matches luminance histogram so the edited ROI looks natural
        in the target frame.
        """
        h, w = edited_roi.shape[:2]
        target_resized = cv2.resize(target_frame_roi, (w, h))

        return match_histogram_luminance(
            source=edited_roi,
            reference=target_resized,
            color_space=self.config.color_space,
        )

    def _create_alpha_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Create a feathered alpha mask for smooth blending.

        Center is 1.0, edges linearly feather to 0.0.
        """
        h, w = shape
        mask = np.ones((h, w), dtype=np.float32)
        border = max(1, min(h, w) // 10)

        for i in range(border):
            alpha = (i + 1) / border
            mask[i, :] = np.minimum(mask[i, :], alpha)
            mask[h - 1 - i, :] = np.minimum(mask[h - 1 - i, :], alpha)
            mask[:, i] = np.minimum(mask[:, i], alpha)
            mask[:, w - 1 - i] = np.minimum(mask[:, w - 1 - i], alpha)

        return mask

    def run(
        self,
        tracks: list[TextTrack],
        frames: dict[int, np.ndarray],
    ) -> dict[int, list[PropagatedROI]]:
        """Propagate edited ROIs to all frames.

        Returns:
            frame_idx -> list of PropagatedROI for that frame.
        """
        logger.info("S4: Propagating edited ROIs across frames")
        propagated: dict[int, list[PropagatedROI]] = {}

        for track in tracks:
            if track.edited_roi is None:
                logger.warning(
                    "S4: Track %d has no edited ROI, skipping", track.track_id
                )
                continue

            for frame_idx, det in track.detections.items():
                frame = frames.get(frame_idx)
                if frame is None:
                    continue

                original_roi = frame[det.bbox.to_slice()]
                if original_roi.size == 0:
                    continue

                adapted_roi = self.propagate_to_frame(
                    track.edited_roi, original_roi
                )
                alpha = self._create_alpha_mask(adapted_roi.shape[:2])

                prop_roi = PropagatedROI(
                    frame_idx=frame_idx,
                    track_id=track.track_id,
                    roi_image=adapted_roi,
                    alpha_mask=alpha,
                    target_quad=det.quad,
                )

                if frame_idx not in propagated:
                    propagated[frame_idx] = []
                propagated[frame_idx].append(prop_roi)

        return propagated
