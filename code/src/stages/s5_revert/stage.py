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
from tqdm import tqdm

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
        # Pre-composite inpainter — lazy loaded on first use.
        self._pre_inpainter = None

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
        src_center: tuple[int, int] | None = None,
        flags: int = cv2.NORMAL_CLONE,
    ) -> np.ndarray:
        """Composite the warped ROI into the frame using cv2.seamlessClone.

        Poisson blending alternative to alpha compositing — matches local
        gradients/lighting at the boundary instead of feathering RGB values.
        The feathered alpha mask is binarized (>0) to form the clone mask.
        Relies on the bbox expansion in `warp_roi_to_frame` to guarantee a
        zero-alpha border so the mask stays strictly interior (a hard
        requirement of cv2.seamlessClone).

        ``src_center``: paste location in frame-space (integer pixel). When
        supplied, it overrides the default ``bbox.x + bbox.width//2``
        fallback. Callers should prefer a center derived from the
        float-precision effective frame corners rounded once: computing it
        from ``target_bbox`` applies two rounding steps (once for the
        bbox origin, once for the half-width) that can disagree by 1 px
        across frames and produce visible seamlessClone jitter even when
        the underlying quad is still.
        """
        # Binarize the feathered alpha into a clone mask.
        mask = (warped_alpha > 0).astype(np.uint8) * 255
        if mask.sum() == 0:
            return frame

        src = warped_roi

        if src_center is None:
            center = (
                target_bbox.x + target_bbox.width // 2,
                target_bbox.y + target_bbox.height // 2,
            )
        else:
            center = src_center

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

    # ------------------------------------------------------------------
    # Temporal corner smoothing
    # ------------------------------------------------------------------

    @staticmethod
    def _seamless_center_from_corners(
        corners: np.ndarray,
    ) -> tuple[int, int]:
        """Round-once bbox-midpoint center from float frame-space corners.

        The default seamlessClone center in ``composite_roi_into_frame_seamless``
        is ``bbox.x + bbox.width // 2``. Since ``bbox.x`` and ``bbox.width``
        are each independently ``int(round(...))`` of the floating-point
        quad, the two roundings can disagree by ±1 px across frames even
        when the underlying float midpoint is essentially still — that
        manifests as seamlessClone seed-pixel jitter, which in turn shifts
        the Poisson-blended composite by a pixel frame to frame. Rounding
        the float midpoint once removes that amplification.
        """
        xs = corners[:, 0]
        ys = corners[:, 1]
        cx = (float(xs.min()) + float(xs.max())) / 2.0
        cy = (float(ys.min()) + float(ys.max())) / 2.0
        return (int(round(cx)), int(round(cy)))

    @staticmethod
    def _project_canonical_to_frame(
        H_from_frontal: np.ndarray,
        canonical_corners: np.ndarray,
        delta_H: np.ndarray | None = None,
    ) -> np.ndarray:
        """Project canonical rectangle corners to frame-space positions.

        Args:
            H_from_frontal: 3x3 (canonical -> frame).
            canonical_corners: (4, 2).
            delta_H: optional 3x3 (canonical ref -> canonical tgt).

        Returns:
            (4, 2) frame-space corner positions.
        """
        H_eff = H_from_frontal @ delta_H if delta_H is not None else H_from_frontal
        pts = np.column_stack([canonical_corners, np.ones(4)])  # (4, 3)
        proj = (H_eff @ pts.T).T  # (4, 3)
        w = proj[:, 2:3]
        w = np.where(np.abs(w) < 1e-12, 1e-12, w)
        return (proj[:, :2] / w).astype(np.float64)

    def _get_pre_inpainter(self):
        """Lazy-load the configured pre-composite background inpainter.

        Dispatches on ``config.pre_inpaint_backend``:

        - ``"srnet"``: lksshw/SRNet wrapper (same class S4 uses).
        - ``"hisam"``: Hi-SAM stroke segmentation + cv2.inpaint.

        Raises ``ValueError`` on an unknown backend — unlike S3, S5's
        pre-inpaint is an explicit pipeline stage opted into via
        ``pre_inpaint=True``; misconfiguration should fail loudly rather
        than silently skip.
        """
        if self._pre_inpainter is not None:
            return self._pre_inpainter
        backend = self.config.pre_inpaint_backend
        if backend == "srnet":
            from src.stages.s4_propagation.srnet_inpainter import SRNetInpainter
            logger.info("S5: loading SRNet pre-composite inpainter from %s",
                         self.config.pre_inpaint_checkpoint)
            self._pre_inpainter = SRNetInpainter(
                checkpoint_path=self.config.pre_inpaint_checkpoint,
                device=self.config.pre_inpaint_device,
            )
            return self._pre_inpainter
        if backend == "hisam":
            from src.stages.s4_propagation.segmentation_inpainter import (
                SegmentationBasedInpainter,
            )
            logger.info("S5: loading Hi-SAM pre-composite inpainter from %s",
                         self.config.pre_inpaint_checkpoint)
            self._pre_inpainter = SegmentationBasedInpainter(
                checkpoint_path=self.config.pre_inpaint_checkpoint,
                device=self.config.pre_inpaint_device,
                model_type=self.config.pre_inpaint_hisam_model_type,
                mask_dilation_px=self.config.pre_inpaint_hisam_mask_dilation_px,
                inpaint_method=self.config.pre_inpaint_hisam_inpaint_method,
                use_patch_mode=self.config.pre_inpaint_hisam_use_patch_mode,
            )
            return self._pre_inpainter
        raise ValueError(
            f"Unknown pre_inpaint_backend: {backend!r}. "
            f"Expected one of: 'srnet', 'hisam'."
        )

    @staticmethod
    def _expand_quad_from_centroid(
        corners: np.ndarray, ratio: float,
    ) -> np.ndarray:
        """Expand a (4, 2) quad outward from its centroid by ``ratio``.

        Each corner moves ``ratio * (corner - centroid)`` further from
        the center. Returns (4, 2) float32.
        """
        centroid = corners.mean(axis=0)
        expanded = centroid + (1.0 + ratio) * (corners - centroid)
        return expanded.astype(np.float32)

    @staticmethod
    def _shrink_quad_to_centroid(
        corners: np.ndarray, shrink_px: float,
    ) -> np.ndarray:
        """Move each corner of a (4, 2) quad ``shrink_px`` toward the centroid.

        Unlike ``_expand_quad_from_centroid`` which scales by a ratio,
        this shrink is expressed in pixels — it replaces the pre-fix
        5×5 ``cv2.erode`` step, which buffered the boundary by a
        constant pixel margin regardless of quad size.
        """
        centroid = corners.mean(axis=0)
        vecs = centroid - corners  # (4, 2) from each corner toward center
        dists = np.linalg.norm(vecs, axis=1, keepdims=True)
        unit = vecs / np.maximum(dists, 1e-6)
        return (corners + shrink_px * unit).astype(np.float32)

    @staticmethod
    def _build_antialiased_mask(
        corners: np.ndarray,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Rasterise a (4, 2) quad to a grayscale mask with sub-pixel AA edges.

        ``shape``: ``(H, W)`` of the returned mask. Returns ``(H, W)``
        uint8 with values in ``[0, 255]``.

        Uses ``shift=4`` (1/16 px precision) and
        ``lineType=cv2.LINE_AA`` so sub-pixel quad motion produces a
        proportional change in the boundary mask values, instead of a
        1-px jump when int-cast corners cross a pixel grid line. The
        grayscale output doubles as a soft alpha for blending.
        """
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        shift = 4
        corners_fixed = np.round(corners * (1 << shift)).astype(np.int32)
        cv2.fillConvexPoly(
            mask, corners_fixed, 255,
            lineType=cv2.LINE_AA,
            shift=shift,
        )
        return mask

    def _pre_inpaint_region(
        self,
        frame: np.ndarray,
        quad_corners: np.ndarray,
    ) -> np.ndarray:
        """Inpaint the text region in-place on ``frame`` before compositing.

        1. Expand the quad outward from its centroid.
        2. Warp the expanded region to a rectangle.
        3. Run the configured pre-composite inpainter (SRNet or Hi-SAM)
           to erase any text.
        4. Warp the inpainted result back into the frame.

        Returns the modified frame (may be a new array or in-place).
        """
        expansion = self.config.pre_inpaint_expansion
        inpainter = self._get_pre_inpainter()

        # Expand the quad
        expanded = self._expand_quad_from_centroid(
            quad_corners.astype(np.float32), expansion,
        )

        # Determine the output rectangle size preserving the quad's
        # approximate aspect ratio. Use the average edge lengths.
        top_w = float(np.linalg.norm(expanded[1] - expanded[0]))
        bot_w = float(np.linalg.norm(expanded[2] - expanded[3]))
        left_h = float(np.linalg.norm(expanded[3] - expanded[0]))
        right_h = float(np.linalg.norm(expanded[2] - expanded[1]))
        rect_w = max(4, int(round((top_w + bot_w) / 2)))
        rect_h = max(4, int(round((left_h + right_h) / 2)))

        dst_rect = np.array(
            [[0, 0], [rect_w, 0], [rect_w, rect_h], [0, rect_h]],
            dtype=np.float32,
        )

        # Warp frame region to rectangle
        H_to_rect = cv2.getPerspectiveTransform(expanded, dst_rect)
        rect_crop = cv2.warpPerspective(
            frame, H_to_rect, (rect_w, rect_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Inpaint
        inpainted = inpainter.inpaint(rect_crop)

        # Warp back into the frame
        H_from_rect = cv2.getPerspectiveTransform(dst_rect, expanded)
        frame_h, frame_w = frame.shape[:2]

        # Only overwrite the region covered by the expanded quad, not
        # the whole frame. Use a mask to avoid touching pixels outside.
        # BORDER_REPLICATE so edge pixels of the inpainted image extend
        # outward instead of blending with black — prevents the dark
        # fringe that BORDER_CONSTANT produces at bilinear boundaries.
        warped_back = cv2.warpPerspective(
            inpainted, H_from_rect, (frame_w, frame_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        # Build a sub-pixel antialiased mask and use it as a soft alpha
        # for blending inpainted pixels into the frame. Replaces the old
        # ``fillConvexPoly(int32) + 5×5 erode + boolean paste`` path,
        # which had two jitter sources:
        #   1. ``.astype(np.int32)`` quantises each quad corner to the
        #      pixel grid, so a sub-pixel quad drift that straddles a
        #      half-integer boundary flipped the rasterised edge by a
        #      full pixel between frames.
        #   2. The hard (boolean) paste meant that 1-px edge flip
        #      exposed any tonal mismatch between the inpainted and
        #      original backgrounds as a flickering 1-px ring.
        # Shrinking the expanded quad 2 px inward replaces the 5×5
        # erode's buffer against BORDER_REPLICATE interpolation leakage
        # at the warp edge, without destroying the AA gradient.
        mask_quad = self._shrink_quad_to_centroid(expanded, shrink_px=2.0)
        mask = self._build_antialiased_mask(mask_quad, (frame_h, frame_w))
        alpha = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
        frame[:] = (
            frame.astype(np.float32) * (1.0 - alpha)
            + warped_back.astype(np.float32) * alpha
        ).astype(np.uint8)

        return frame

    @staticmethod
    def _smooth_corner_trajectories(
        trajectories: dict[int, np.ndarray],
        frame_indices: list[int],
        window: int,
        sigma: float,
    ) -> dict[int, np.ndarray]:
        """Gaussian-weighted temporal smoothing of per-frame corner positions.

        Args:
            trajectories: frame_idx -> (4, 2) projected corners.
            frame_indices: sorted list of frame indices where this track
                has valid data.
            window: total window width (must be odd, >= 3). Clamped to
                min(window, len(frame_indices)).
            sigma: Gaussian sigma in frames.

        Returns:
            frame_idx -> (4, 2) smoothed corners.
        """
        n = len(frame_indices)
        if n < 3 or window < 3:
            return trajectories

        window = min(window, n)
        if window % 2 == 0:
            window -= 1

        # Stack into (N, 4, 2) array for vectorised smoothing.
        stacked = np.stack([trajectories[fi] for fi in frame_indices])  # (N, 4, 2)

        # Build 1D Gaussian kernel.
        half = window // 2
        x = np.arange(-half, half + 1, dtype=np.float64)
        kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
        kernel /= kernel.sum()

        # Pad with edge replication so boundary frames get smoothed too.
        padded = np.pad(stacked, ((half, half), (0, 0), (0, 0)), mode="edge")

        smoothed = np.empty_like(stacked)
        for i in range(n):
            # Weighted sum over the window centered on frame i.
            patch = padded[i:i + window]  # (window, 4, 2)
            smoothed[i] = np.einsum("w,wcd->cd", kernel, patch)

        return {fi: smoothed[i] for i, fi in enumerate(frame_indices)}

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

        smooth_window = self.config.temporal_smooth_window
        use_smoothing = smooth_window >= 3

        # ----- Pass 1: collect per-detection delta_H + projected corners -----
        # Keyed by (frame_idx, track_id) for compositing lookup.
        # Also collect per-track corner trajectories for smoothing.
        delta_H_map: dict[tuple[int, int], np.ndarray | None] = {}
        # Per-track projected corner trajectories: track_id -> {frame_idx -> (4, 2)}
        track_corners: dict[int, dict[int, np.ndarray]] = {}
        # Per-track sorted frame indices (for smoothing order)
        track_frame_order: dict[int, list[int]] = {}

        for frame_idx in sorted_idxs:
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

                key = (frame_idx, prop_roi.track_id)
                delta_H_map[key] = delta_H

                if use_smoothing and track.canonical_size is not None:
                    w_can, h_can = track.canonical_size
                    can_corners = np.array(
                        [[0, 0], [w_can, 0], [w_can, h_can], [0, h_can]],
                        dtype=np.float64,
                    )
                    proj = self._project_canonical_to_frame(
                        det.H_from_frontal, can_corners, delta_H,
                    )
                    track_corners.setdefault(prop_roi.track_id, {})[frame_idx] = proj
                    track_frame_order.setdefault(prop_roi.track_id, []).append(frame_idx)

        # ----- Smooth per-track corner trajectories -----
        smoothed_H_map: dict[tuple[int, int], np.ndarray] = {}
        if use_smoothing:
            sigma = self.config.temporal_smooth_sigma
            for track_id, corners_by_frame in track_corners.items():
                track = tracks_by_id.get(track_id)
                if track is None or track.canonical_size is None:
                    continue
                fi_sorted = sorted(track_frame_order[track_id])
                smoothed = self._smooth_corner_trajectories(
                    corners_by_frame, fi_sorted, smooth_window, sigma,
                )
                w_can, h_can = track.canonical_size
                can_corners = np.array(
                    [[0, 0], [w_can, 0], [w_can, h_can], [0, h_can]],
                    dtype=np.float32,
                )
                for fi, sm_corners in smoothed.items():
                    # Recover an effective H from canonical -> smoothed frame
                    # corners via DLT (cv2.getPerspectiveTransform).
                    try:
                        H_smooth = cv2.getPerspectiveTransform(
                            can_corners,
                            sm_corners.astype(np.float32),
                        )
                        smoothed_H_map[(fi, track_id)] = H_smooth
                    except cv2.error:
                        pass
            logger.debug(
                "S5: temporal smoothing applied to %d tracks (window=%d, σ=%.1f)",
                len(track_corners), smooth_window, sigma,
            )

        # ----- Pass 2: composite -----
        # Progress bar sized to total propagated ROIs (one inpaint call
        # per ROI when pre_inpaint is on). Disabled otherwise, since the
        # rest of the composite path is fast enough not to need a bar.
        total_rois = sum(len(rois) for rois in propagated_rois.values())
        pbar = tqdm(
            total=total_rois,
            desc="S5 pre-inpaint",
            unit="roi",
            leave=False,
            disable=not self.config.pre_inpaint,
        )

        output_frames = []
        for frame_idx in sorted_idxs:
            frame = frames[frame_idx].copy()

            for prop_roi in propagated_rois.get(frame_idx, []):
                pbar.update(1)
                track = tracks_by_id.get(prop_roi.track_id)
                if track is None:
                    continue
                det = track.detections.get(frame_idx)
                if det is None or not det.homography_valid or det.H_from_frontal is None:
                    continue

                key = (frame_idx, prop_roi.track_id)
                delta_H = delta_H_map.get(key)

                # Use the smoothed effective homography when available.
                # It replaces H_from_frontal @ delta_H entirely — the
                # smoothed H already maps canonical -> frame with both
                # the original tracking and the refiner correction baked
                # in, just temporally filtered.
                smoothed_H = smoothed_H_map.get(key) if use_smoothing else None

                if smoothed_H is not None:
                    result = self.warp_roi_to_frame(
                        prop_roi, smoothed_H, frame.shape[:2],
                        delta_H=None,  # already baked in
                    )
                else:
                    # Diagnostic: paint pure blue at the unrefined
                    # warped-alpha region before compositing.
                    if _REFINER_DIAGNOSTIC_BLUE and delta_H is not None:
                        prop_roi.alpha_mask = np.ones_like(prop_roi.alpha_mask, dtype=np.float32)
                        unref = self.warp_roi_to_frame(
                            prop_roi, det.H_from_frontal, frame.shape[:2],
                            delta_H=None,
                        )
                        if unref is not None:
                            _, unref_alpha, unref_bbox = unref
                            sl = unref_bbox.to_slice()
                            mask_diag = unref_alpha > 0.0
                            region = frame[sl]
                            region[mask_diag] = (255, 0, 0)
                            frame[sl] = region

                    result = self.warp_roi_to_frame(
                        prop_roi, det.H_from_frontal, frame.shape[:2],
                        delta_H=delta_H,
                    )

                if result is None:
                    continue

                warped_roi, warped_alpha, target_bbox = result

                # Pre-composite inpainting: erase original text from the
                # frame region under the ROI before compositing the edited
                # ROI on top. Prevents Poisson blending artifacts from
                # remnant original text leaking past the quad boundary.
                if self.config.pre_inpaint and track.canonical_size is not None:
                    H_eff = smoothed_H if smoothed_H is not None else (
                        det.H_from_frontal @ delta_H
                        if delta_H is not None else det.H_from_frontal
                    )
                    w_can, h_can = track.canonical_size
                    can_corners = np.array(
                        [[0, 0], [w_can, 0], [w_can, h_can], [0, h_can]],
                        dtype=np.float64,
                    )
                    frame_quad = self._project_canonical_to_frame(
                        H_eff, can_corners,
                    )
                    try:
                        frame = self._pre_inpaint_region(frame, frame_quad)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "S5: pre-inpaint failed for track %d frame %d: %s",
                            prop_roi.track_id, frame_idx, exc,
                        )

                if _REFINER_DIAGNOSTIC_BLUE:
                    frame = self.composite_roi_into_frame(
                        frame, warped_roi, warped_alpha, target_bbox
                    )
                else:
                    # Derive the seamlessClone center from the effective
                    # frame-space corners (post-refiner / post-smoothing)
                    # rounded once, instead of letting cv2.seamlessClone
                    # fall back to the int-bbox-based midpoint which
                    # jitters by 1 px across frames as two independent
                    # roundings cross half-integer boundaries.
                    src_center: tuple[int, int] | None = None
                    if track.canonical_size is not None:
                        w_can, h_can = track.canonical_size
                        can_corners = np.array(
                            [[0, 0], [w_can, 0], [w_can, h_can], [0, h_can]],
                            dtype=np.float64,
                        )
                        if smoothed_H is not None:
                            eff_corners = self._project_canonical_to_frame(
                                smoothed_H, can_corners,
                            )
                        else:
                            eff_corners = self._project_canonical_to_frame(
                                det.H_from_frontal, can_corners, delta_H,
                            )
                        src_center = self._seamless_center_from_corners(
                            eff_corners,
                        )
                    frame = self.composite_roi_into_frame_seamless(
                        frame, warped_roi, warped_alpha, target_bbox,
                        src_center=src_center,
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

        pbar.close()
        return output_frames
