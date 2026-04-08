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

from .base_inpainter import BaseBackgroundInpainter
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
        # Inpainter is loaded lazily on first use to avoid paying the model
        # load cost when LCM is disabled.
        self._inpainter: BaseBackgroundInpainter | None = None

    def _get_inpainter(self) -> BaseBackgroundInpainter | None:
        """Lazy-load the configured background inpainter, or None if disabled."""
        if self._inpainter is not None:
            return self._inpainter
        backend = self.config.inpainter_backend
        if backend in (None, "", "none"):
            return None
        if backend == "srnet":
            from .srnet_inpainter import SRNetInpainter
            ckpt = self.config.inpainter_checkpoint_path
            if not ckpt:
                logger.warning(
                    "S4: inpainter_backend=srnet but no checkpoint_path "
                    "configured; LCM will be skipped"
                )
                return None
            logger.info("S4: loading SRNet inpainter from %s", ckpt)
            self._inpainter = SRNetInpainter(
                checkpoint_path=ckpt,
                device=self.config.inpainter_device,
            )
            return self._inpainter
        raise ValueError(f"Unknown inpainter_backend: {backend!r}")

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

    def _warp_to_canonical(
        self, track: TextTrack, det, frame: np.ndarray
    ) -> np.ndarray | None:
        """Warp a frame's detection into canonical frontal ROI space.

        Falls back to a bbox crop when no homography is available.
        Returns None if the result would be empty.
        """
        if (det.H_to_frontal is not None and det.homography_valid
                and track.canonical_size is not None):
            w, h = track.canonical_size
            return cv2.warpPerspective(frame, det.H_to_frontal, (w, h))
        return frame[det.bbox.to_slice()]

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

        # Lazy-load inpainter once for the whole run
        inpainter = self._get_inpainter() if self.config.use_lcm else None

        for track in tracks:
            if track.edited_roi is None:
                logger.warning(
                    "S4: Track %d has no edited ROI, skipping", track.track_id
                )
                continue

            self.lcm.reset()  # clear EMA buffer between tracks

            # Compute reference background once per track if LCM is enabled.
            # We need the reference detection's frontalized ROI to feed the
            # inpainter; that requires the reference frame to be available.
            ref_det = track.detections.get(track.reference_frame_idx)
            ref_background: np.ndarray | None = None
            if (inpainter is not None and ref_det is not None
                    and ref_det.inpainted_background is None):
                ref_frame = frames.get(track.reference_frame_idx)
                if ref_frame is not None:
                    ref_canonical = self._warp_to_canonical(track, ref_det, ref_frame)
                    if ref_canonical is not None and ref_canonical.size > 0:
                        ref_det.inpainted_background = inpainter.inpaint(ref_canonical)
            if ref_det is not None:
                ref_background = ref_det.inpainted_background

            for frame_idx, det in track.detections.items():
                frame = frames.get(frame_idx)
                if frame is None:
                    continue

                target_roi = self._warp_to_canonical(track, det, frame)
                if target_roi is None or target_roi.size == 0:
                    continue

                # Inpaint this detection's background on demand if LCM is on
                if (inpainter is not None
                        and det.inpainted_background is None):
                    det.inpainted_background = inpainter.inpaint(target_roi)

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
