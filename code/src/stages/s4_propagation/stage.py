"""Stage 4: Propagation.

Adapts the translated reference ROI to match each frame's lighting and
generates feathered alpha masks.

Two lighting-correction paths are supported:

1. **LCM** (Lighting Correction Module from STRIVE TPM): used when each
   detection has an `inpainted_background` populated. Computes a per-pixel
   multiplicative ratio map between the reference and target inpainted
   backgrounds and applies it to the edited reference ROI. This is the
   first half of STRIVE's Text Propagation Module; the BPN (blur prediction)
   half will follow.
2. **Histogram matching** (legacy): YCrCb luminance histogram matching
   between the edited ROI and each frame's raw ROI crop. Used as a
   fallback when inpainted backgrounds are unavailable.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.config import PipelineConfig
from src.data_types import PropagatedROI, TextTrack
from src.utils.image_processing import match_histogram_luminance

from .lighting_correction_module import LCMConfig, LightingCorrectionModule

logger = logging.getLogger(__name__)


class PropagationStage:
    def __init__(self, config: PipelineConfig):
        self.config = config.propagation
        self.lcm = LightingCorrectionModule(
            LCMConfig(
                eps=self.config.lcm_eps,
                ratio_clip_min=self.config.lcm_ratio_clip_min,
                ratio_clip_max=self.config.lcm_ratio_clip_max,
                ratio_blur_ksize=self.config.lcm_ratio_blur_ksize,
                use_log_domain=self.config.lcm_use_log_domain,
                temporal_alpha=self.config.lcm_temporal_alpha,
                neighbor_self_weight=self.config.lcm_neighbor_self_weight,
            )
        )

    def propagate_to_frame(
        self,
        edited_roi: np.ndarray,
        target_frame_roi: np.ndarray,
    ) -> np.ndarray:
        """Adapt edited reference ROI to match a target frame's appearance.

        Both inputs should be in canonical frontal space (same size),
        but resizes as a safety fallback if dimensions differ.
        """
        h, w = edited_roi.shape[:2]
        if target_frame_roi.shape[:2] != (h, w):
            target_frame_roi = cv2.resize(target_frame_roi, (w, h))

        return match_histogram_luminance(
            source=edited_roi,
            reference=target_frame_roi,
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

            ref_det = track.detections.get(track.reference_frame_idx)
            ref_background = ref_det.inpainted_background if ref_det else None
            self.lcm.reset()  # clear EMA buffer between tracks

            for frame_idx, det in track.detections.items():
                frame = frames.get(frame_idx)
                if frame is None:
                    continue

                # Warp to canonical frontal if homography available
                if (det.H_to_frontal is not None and det.homography_valid
                        and track.canonical_size is not None):
                    w, h = track.canonical_size
                    target_roi = cv2.warpPerspective(
                        frame, det.H_to_frontal, (w, h)
                    )
                else:
                    # Fallback: bbox crop
                    target_roi = frame[det.bbox.to_slice()]

                if target_roi.size == 0:
                    continue

                # Choose LCM if available, else fall back to histogram matching
                if (self.config.use_lcm
                        and ref_background is not None
                        and det.inpainted_background is not None):
                    adapted_roi = self.lcm.correct(
                        edited_roi=track.edited_roi,
                        ref_background=ref_background,
                        target_background=det.inpainted_background,
                    )
                else:
                    adapted_roi = self.propagate_to_frame(
                        track.edited_roi, target_roi
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
