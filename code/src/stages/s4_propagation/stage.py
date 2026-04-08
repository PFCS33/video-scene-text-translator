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
        # Inpainter and BPN are loaded lazily on first use to avoid paying
        # the model load cost when LCM/BPN are disabled.
        self._inpainter: BaseBackgroundInpainter | None = None
        self._bpn = None  # type: ignore[assignment]

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

    def _get_bpn(self):
        """Lazy-load the BPN predictor, or None if disabled."""
        if self._bpn is not None:
            return self._bpn
        if not self.config.use_bpn:
            return None
        ckpt = self.config.bpn_checkpoint_path
        if not ckpt:
            logger.warning(
                "S4: use_bpn=True but no bpn_checkpoint_path configured; "
                "BPN will be skipped"
            )
            return None
        from .bpn_predictor import BPNPredictor
        logger.info("S4: loading BPN from %s", ckpt)
        self._bpn = BPNPredictor(
            checkpoint_path=ckpt,
            n_neighbors=self.config.bpn_n_neighbors,
            image_size=tuple(self.config.bpn_image_size),
            kernel_size=self.config.bpn_kernel_size,
            device=self.config.bpn_device,
        )
        return self._bpn

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

        # Lazy-load inpainter and BPN once for the whole run
        inpainter = self._get_inpainter() if self.config.use_lcm else None
        bpn = self._get_bpn() if self.config.use_bpn else None

        for track in tracks:
            if track.edited_roi is None:
                logger.warning(
                    "S4: Track %d has no edited ROI, skipping", track.track_id
                )
                continue

            self.lcm.reset()  # clear EMA buffer between tracks

            # ----- Reference background (LCM only) -----
            ref_det = track.detections.get(track.reference_frame_idx)
            ref_background: np.ndarray | None = None
            ref_canonical: np.ndarray | None = None
            if ref_det is not None:
                ref_frame = frames.get(track.reference_frame_idx)
                if ref_frame is not None:
                    ref_canonical = self._warp_to_canonical(track, ref_det, ref_frame)
                if (inpainter is not None and ref_canonical is not None
                        and ref_canonical.size > 0
                        and ref_det.inpainted_background is None):
                    ref_det.inpainted_background = inpainter.inpaint(ref_canonical)
                ref_background = ref_det.inpainted_background

            # ----- First pass: build per-detection lit ROIs -----
            # Each entry: (frame_idx, det, target_canonical, lit_edited_roi)
            per_det_outputs: list[tuple[int, object, np.ndarray, np.ndarray]] = []

            for frame_idx, det in track.detections.items():
                frame = frames.get(frame_idx)
                if frame is None:
                    continue

                target_canonical = self._warp_to_canonical(track, det, frame)
                if target_canonical is None or target_canonical.size == 0:
                    continue

                # Inpaint this detection's background on demand if LCM is on
                if (inpainter is not None
                        and det.inpainted_background is None):
                    det.inpainted_background = inpainter.inpaint(target_canonical)

                # Choose LCM if available, else fall back to histogram matching
                if (self.config.use_lcm
                        and ref_background is not None
                        and det.inpainted_background is not None):
                    lit_edited = self.lcm.correct(
                        edited_roi=track.edited_roi,
                        ref_background=ref_background,
                        target_background=det.inpainted_background,
                    )
                else:
                    lit_edited = self.propagate_to_frame(
                        track.edited_roi, target_canonical
                    )

                per_det_outputs.append((frame_idx, det, target_canonical, lit_edited))

            if not per_det_outputs:
                continue

            # ----- Second pass: optional BPN differential blur -----
            # BPN reads the original ref + target canonicals to predict
            # how blurry each target is relative to the reference, then
            # we apply that blur to each target's lit_edited ROI so the
            # final composite matches the target's sharpness.
            if bpn is not None and ref_canonical is not None:
                target_canonicals = [t for (_, _, t, _) in per_det_outputs]
                try:
                    params = bpn.predict_params(
                        ref_canonical=ref_canonical,
                        target_canonicals=target_canonicals,
                    )
                    new_outputs = []
                    for i, (frame_idx, det, target_canonical, lit_edited) in enumerate(per_det_outputs):
                        blurred = bpn.apply_blur(
                            lit_edited,
                            params["sigma_x"][i],
                            params["sigma_y"][i],
                            params["rho"][i],
                            params["w"][i],
                        )
                        new_outputs.append((frame_idx, det, target_canonical, blurred))
                    per_det_outputs = new_outputs
                except Exception as e:  # noqa: BLE001
                    logger.warning("S4: BPN inference failed (%s); skipping", e)

            # ----- Emit PropagatedROIs -----
            for frame_idx, det, _, adapted_roi in per_det_outputs:
                alpha = self._create_alpha_mask(adapted_roi.shape[:2])
                prop_roi = PropagatedROI(
                    frame_idx=frame_idx,
                    track_id=track.track_id,
                    roi_image=adapted_roi,
                    alpha_mask=alpha,
                    target_quad=det.quad,
                )
                propagated.setdefault(frame_idx, []).append(prop_roi)

        return propagated
