"""Stage 5: Revert (De-Frontalization + ROI Compositing).

Applies inverse homography to warp translated ROIs back to each
frame's perspective, then alpha-blends them into the original frames.
Optionally runs the ROI alignment refiner on each (reference, target)
canonical ROI pair to correct residual CoTracker tracking error before
compositing.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.config import PipelineConfig
from src.data_types import BBox, PropagatedROI, TextTrack

from .refiner import RefinerInference

logger = logging.getLogger(__name__)


# DIAGNOSTIC FLAG (temporary — flip to True to produce a refiner-sanity
# visualization, then flip back to False before committing).
#
# When True and revert.use_refiner is on, the S5 run() loop paints pure
# blue over every pixel where the **unrefined** warped alpha would have
# placed content, then composites the **refined** ROI on top using
# plain alpha blending (not seamlessClone — Poisson would smear the
# blue into halos and muddy the signal). Therefore:
#
#   - If ΔH ≈ I  → the refined alpha covers every blue pixel → no blue
#     visible in the output.
#   - If ΔH ≠ I  → the refined composite lands at a shifted position,
#     leaving a crescent of exposed blue on the side the shape shifted
#     away from. The crescent width equals the per-pixel shift magnitude.
#
# Leave this False for normal runs. Intended only for one-off debug
# video renders comparing the unrefined vs. refined geometry.
_REFINER_DIAGNOSTIC_BLUE = False


class RevertStage:
    def __init__(self, config: PipelineConfig):
        self.config = config.revert
        self._refiner: RefinerInference | None = None
        if self.config.use_refiner:
            self._refiner = RefinerInference(
                checkpoint_path=self.config.refiner_checkpoint_path,
                device=self.config.refiner_device,
                image_size=tuple(self.config.refiner_image_size),
                max_corner_offset_px=self.config.refiner_max_corner_offset_px,
                use_gate=self.config.use_refiner_gate,
                score_margin=self.config.refiner_score_margin,
            )

    def warp_roi_to_frame(
        self,
        propagated_roi: PropagatedROI,
        H_from_frontal: np.ndarray | None,
        frame_shape: tuple[int, int],
        delta_H: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, BBox] | None:
        """Warp a propagated ROI back to the original frame's perspective.

        Uses the inverse homography (frontal -> frame) to undo frontalization,
        warping only to the target quad's bounding box region instead of the
        full frame.

        Args:
            propagated_roi: The color-adapted ROI.
            H_from_frontal: 3x3 homography matrix (frontal -> frame), or None.
            frame_shape: (height, width) of the target frame.
            delta_H: Optional 3x3 forward homography in canonical pixel
                coords, predicted by the S5 alignment refiner. When
                present, it's composed into the warp chain as
                ``T @ H_from_frontal @ delta_H``. Semantically, the
                edited ROI lives in the reference-canonical alignment,
                ``delta_H`` maps it into the target-canonical alignment,
                and ``H_from_frontal`` then maps it into target frame
                space. None skips refinement entirely (no-op).

        Returns:
            (warped_roi, warped_alpha, target_bbox) or None if homography
            is None or the target bbox has zero area after clamping.
        """
        if H_from_frontal is None:
            return None

        frame_h, frame_w = frame_shape

        # Compute target bbox from the detection's quad
        target_bbox = propagated_roi.target_quad.to_bbox()

        # expand the box by a small margin (5% of each dimension, 2 px minimum)
        expansion_w = max(int(target_bbox.width * 0.05), 2)
        expansion_h = max(int(target_bbox.height * 0.05), 2)
        target_bbox = BBox(
            x=target_bbox.x - expansion_w,
            y=target_bbox.y - expansion_h,
            width=target_bbox.width + 2 * expansion_w,
            height=target_bbox.height + 2 * expansion_h,
        )

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

        # Coordinate chain: canonical (ref alignment) -> canonical (tgt
        # alignment) via delta_H -> frame space (H_from_frontal) ->
        # bbox-local coords (T shifts origin to bbox top-left). When
        # delta_H is None this collapses to the original path.
        if delta_H is not None:
            H_adjusted = T @ H_from_frontal @ delta_H
        else:
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

    def composite_roi_into_frame_seamless(
        self,
        frame: np.ndarray,
        warped_roi: np.ndarray,
        warped_alpha: np.ndarray,
        target_bbox: BBox,
        flags: int = cv2.NORMAL_CLONE,
    ) -> np.ndarray:
        """Composite the warped ROI into the frame using cv2.seamlessClone.

        Poisson blending alternative to alpha compositing — matches local
        gradients/lighting at the boundary instead of feathering RGB values.
        The feathered alpha mask is binarized (>0) to form the clone mask.
        Relies on the bbox expansion in `warp_roi_to_frame` to guarantee a
        zero-alpha border so the mask stays strictly interior (a hard
        requirement of cv2.seamlessClone).
        """
        # Binarize the feathered alpha into a clone mask.
        mask = (warped_alpha > 0).astype(np.uint8) * 255
        if mask.sum() == 0:
            return frame

        src = warped_roi

        # Center of the bbox in destination (frame) coordinates.
        center = (
            target_bbox.x + target_bbox.width // 2,
            target_bbox.y + target_bbox.height // 2,
        )

        # seamlessClone requires the source (centered at `center`) to lie
        # entirely within the destination. Bail out to alpha blending if not.
        sh, sw = src.shape[:2]
        fh, fw = frame.shape[:2]
        half_w, half_h = sw // 2, sh // 2
        if (
            center[0] - half_w < 0
            or center[1] - half_h < 0
            or center[0] + (sw - half_w) > fw
            or center[1] + (sh - half_h) > fh
        ):
            return self.composite_roi_into_frame(
                frame, warped_roi, warped_alpha, target_bbox
            )

        return cv2.seamlessClone(src, frame, mask, center, flags)

    def _build_ref_roi_by_track(
        self,
        tracks: list[TextTrack],
        frames: dict[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        """Pre-compute each track's reference canonical ROI once.

        The refiner aligns the (reference, target) canonical ROI pair,
        so the reference side is the same for every detection in a
        given track — warping it once upfront saves N-1 ``warpPerspective``
        calls per track on the hot path.

        Returns a dict keyed by track_id. Tracks whose reference frame
        is missing, whose reference detection has no valid homography,
        or whose canonical_size is unset are silently omitted — S5's
        inner loop treats a missing entry as "no refinement available"
        and falls through to the non-refiner path.
        """
        ref_roi_by_track: dict[int, np.ndarray] = {}
        for track in tracks:
            ref_frame = frames.get(track.reference_frame_idx)
            if ref_frame is None:
                continue
            ref_det = track.detections.get(track.reference_frame_idx)
            if (ref_det is None
                    or not ref_det.homography_valid
                    or ref_det.H_to_frontal is None
                    or track.canonical_size is None):
                continue
            w_can, h_can = track.canonical_size
            try:
                ref_canonical = cv2.warpPerspective(
                    ref_frame, ref_det.H_to_frontal, (w_can, h_can),
                )
            except cv2.error:
                continue
            if ref_canonical.size == 0:
                continue
            ref_roi_by_track[track.track_id] = ref_canonical
        return ref_roi_by_track

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

        # Refinement: precompute per-track reference canonical ROIs once.
        # Empty dict when the refiner is disabled.
        use_refiner = self._refiner is not None
        ref_roi_by_track = (
            self._build_ref_roi_by_track(tracks, frames) if use_refiner else {}
        )
        refine_total = 0
        refine_rejected = 0

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

                # --- Alignment refinement (optional) ---
                delta_H: np.ndarray | None = None
                if (use_refiner
                        and prop_roi.target_roi_canonical is not None):
                    ref_canonical = ref_roi_by_track.get(prop_roi.track_id)
                    if ref_canonical is not None:
                        refine_total += 1
                        try:
                            delta_H = self._refiner.predict_delta_H(  # type: ignore[union-attr]
                                ref_canonical,
                                prop_roi.target_roi_canonical,
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.debug(
                                "S5 refiner: predict_delta_H raised %s; "
                                "falling back to identity",
                                exc,
                            )
                            delta_H = None
                        if delta_H is None:
                            refine_rejected += 1

                # Diagnostic: paint pure blue at the unrefined warped-alpha
                # region before compositing. If ΔH != I, the refined
                # composite will leave a crescent of exposed blue on the
                # side the shape shifted away from. See
                # _REFINER_DIAGNOSTIC_BLUE at module top for details.
                if _REFINER_DIAGNOSTIC_BLUE and delta_H is not None:
                    prop_roi.alpha_mask = np.ones_like(prop_roi.alpha_mask, dtype=np.float32)
                    unref = self.warp_roi_to_frame(
                        prop_roi, det.H_from_frontal, frame.shape[:2],
                        delta_H=None,
                    )
                    if unref is not None:
                        _, unref_alpha, unref_bbox = unref
                        sl = unref_bbox.to_slice()
                        mask = unref_alpha > 0.0
                        region = frame[sl]
                        region[mask] = (255, 0, 0)  # pure blue (BGR)
                        frame[sl] = region

                result = self.warp_roi_to_frame(
                    prop_roi, det.H_from_frontal, frame.shape[:2],
                    delta_H=delta_H,
                )
                if result is None:
                    continue

                warped_roi, warped_alpha, target_bbox = result
                if _REFINER_DIAGNOSTIC_BLUE:
                    # Force plain alpha compositing — seamlessClone would
                    # smear the blue into halos via Poisson blending and
                    # defeat the diagnostic.
                    frame = self.composite_roi_into_frame(
                        frame, warped_roi, warped_alpha, target_bbox
                    )
                else:
                    frame = self.composite_roi_into_frame_seamless(
                        frame, warped_roi, warped_alpha, target_bbox
                    )

            output_frames.append(frame)

        if use_refiner and refine_total > 0:
            rejection_rate = refine_rejected / refine_total
            msg = (
                "S5 refiner: %d / %d predictions rejected (%.1f%%)"
            )
            args = (refine_rejected, refine_total, rejection_rate * 100.0)
            if rejection_rate >= self.config.refiner_rejection_warn_threshold:
                logger.info(msg, *args)
            else:
                logger.debug(msg, *args)

        return output_frames
