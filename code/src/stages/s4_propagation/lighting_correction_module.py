"""Lighting Correction Module (LCM) — first half of the STRIVE TPM.

From STRIVE Section 3.2 (arXiv:2109.02762):

    The LCM computes a spatially-varying, per-channel multiplicative
    correction map between a reference frame ROI and each target frame
    ROI, then applies that correction to the *edited* reference ROI so
    its illumination matches the target frame's lighting.

    The correction is computed on the inpainted (text-removed) backgrounds
    rather than the raw ROIs, because the text content differs between
    the two frames and would otherwise pollute the ratio at text borders.

    The paper formulates the per-pixel ratio as

        r(x, y) = (I_i^p(x, y) + eps) / (I_R^p(x, y) + eps)

    where I_R^p and I_i^p are the inpainted backgrounds of the reference
    and target frames respectively, and eps avoids division-by-zero in
    near-black regions. A weighted average of N neighboring inpainted
    backgrounds is taken before the ratio is computed, which stabilizes
    the estimate against single-frame inpainting artifacts.

    The lit edited ROI is then I_R^edit_lit = r * I_R^edit, applied
    multiplicatively in the canonical frontal space.

This implementation differs from the paper in a few practical respects:
- Adds optional log-domain ratio computation for numerical stability
  (small reference values blow up the linear ratio).
- Adds a Gaussian smoothing pass on the ratio map to suppress
  single-pixel artifacts left over from the inpainting step.
- Adds an optional EMA temporal smoothing across consecutive frames.

The LCM operates entirely in canonical frontal ROI space — its caller
(s4_propagation/stage.py) is responsible for warping into and out of
that space.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LCMConfig:
    """Knobs for the LCM. All defaults match the values used in
    classical_propagation.py before refactoring."""

    eps: float = 1e-3
    """Added to numerator and denominator before computing the ratio,
    avoids division by zero in dark regions."""

    ratio_clip_min: float = 0.5
    ratio_clip_max: float = 2.0
    """Hard clip on the ratio map to suppress extreme outliers from
    inpainting failures or near-black pixels."""

    ratio_blur_ksize: int = 9
    """Gaussian smoothing kernel size for the ratio map (per channel)."""

    use_log_domain: bool = True
    """If True, compute log(I_i + eps) - log(I_R + eps) and exponentiate.
    Numerically more stable than direct division when references are dark."""

    temporal_alpha: float = 1.0
    """EMA weight for temporal smoothing of the ratio map. 1.0 disables
    smoothing (use only the current frame's ratio). Lower values blend
    in the previous frame's ratio for stability."""

    neighbor_self_weight: float = 2.0
    """Weight given to the current frame's own background when averaging
    over neighbors. Higher = trust the current frame more."""


class LightingCorrectionModule:
    """Stateless-ish LCM. Holds an EMA buffer for temporal smoothing
    that should be reset between tracks via :meth:`reset`."""

    def __init__(self, config: LCMConfig | None = None):
        self.cfg = config or LCMConfig()
        self._prev_ratio_map: np.ndarray | None = None

    def reset(self) -> None:
        """Clear temporal smoothing state. Call between tracks."""
        self._prev_ratio_map = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_ratio_map(
        self,
        ref_background: np.ndarray,
        target_background: np.ndarray,
        neighbor_backgrounds: list[np.ndarray] | None = None,
        neighbor_distances: list[int] | None = None,
    ) -> np.ndarray:
        """Compute the per-pixel multiplicative correction map.

        Args:
            ref_background: Inpainted background of the reference ROI in
                canonical frontal space, BGR uint8 (H, W, 3).
            target_background: Inpainted background of the target frame's
                ROI in the same canonical space and shape.
            neighbor_backgrounds: Optional list of inpainted backgrounds
                from frames neighboring the target. If provided, the
                effective target background is a distance-weighted average
                of these plus the target itself.
            neighbor_distances: Frame-distance of each neighbor from the
                target (positive ints). Required if neighbor_backgrounds
                is provided. Used to weight closer neighbors more.

        Returns:
            ratio: float32 (H, W, 3) multiplicative map. Multiply this
                element-wise into the [0, 1] normalized edited ROI to
                obtain the lighting-corrected edited ROI.
        """
        if neighbor_backgrounds:
            target_background = self._weighted_average_backgrounds(
                target_background, neighbor_backgrounds, neighbor_distances
            )

        ratio = self._stable_ratio(ref_background, target_background)
        ratio = self._smooth_ratio(ratio)
        ratio = self._temporal_smooth(ratio)
        return ratio

    def apply(self, edited_roi: np.ndarray, ratio_map: np.ndarray) -> np.ndarray:
        """Apply the ratio map to an edited ROI.

        Args:
            edited_roi: BGR uint8 (H, W, 3) — typically the Stage A
                output for the reference frame.
            ratio_map: Float32 (H, W, 3) returned by :meth:`compute_ratio_map`.

        Returns:
            BGR uint8 (H, W, 3) — the lighting-corrected edited ROI.
        """
        if edited_roi.shape[:2] != ratio_map.shape[:2]:
            raise ValueError(
                f"Shape mismatch: edited_roi {edited_roi.shape[:2]} vs "
                f"ratio_map {ratio_map.shape[:2]}"
            )
        img = edited_roi.astype(np.float32) / 255.0
        out = np.clip(img * ratio_map, 0.0, 1.0)
        return (out * 255.0).astype(np.uint8)

    def correct(
        self,
        edited_roi: np.ndarray,
        ref_background: np.ndarray,
        target_background: np.ndarray,
        neighbor_backgrounds: list[np.ndarray] | None = None,
        neighbor_distances: list[int] | None = None,
    ) -> np.ndarray:
        """Convenience: compute the ratio map and apply it in one call."""
        ratio = self.compute_ratio_map(
            ref_background, target_background,
            neighbor_backgrounds, neighbor_distances,
        )
        return self.apply(edited_roi, ratio)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _stable_ratio(
        self, ref_bg: np.ndarray, target_bg: np.ndarray
    ) -> np.ndarray:
        ref_f = ref_bg.astype(np.float32) / 255.0
        cur_f = target_bg.astype(np.float32) / 255.0
        eps = self.cfg.eps

        if self.cfg.use_log_domain:
            ratio = np.exp(np.log(cur_f + eps) - np.log(ref_f + eps))
        else:
            ratio = (cur_f + eps) / (ref_f + eps)

        ratio = np.clip(ratio, self.cfg.ratio_clip_min, self.cfg.ratio_clip_max)
        return ratio.astype(np.float32)

    def _smooth_ratio(self, ratio_map: np.ndarray) -> np.ndarray:
        """Per-channel Gaussian blur to suppress inpainting artifacts."""
        k = self.cfg.ratio_blur_ksize
        if k <= 1:
            return ratio_map
        out = np.empty_like(ratio_map)
        for c in range(ratio_map.shape[2]):
            out[..., c] = cv2.GaussianBlur(ratio_map[..., c], (k, k), sigmaX=0)
        return out

    def _temporal_smooth(self, ratio_map: np.ndarray) -> np.ndarray:
        if self.cfg.temporal_alpha >= 1.0 or self._prev_ratio_map is None:
            self._prev_ratio_map = ratio_map
            return ratio_map
        a = self.cfg.temporal_alpha
        smoothed = (a * ratio_map + (1.0 - a) * self._prev_ratio_map).astype(np.float32)
        self._prev_ratio_map = smoothed
        return smoothed

    def _weighted_average_backgrounds(
        self,
        target_bg: np.ndarray,
        neighbor_bgs: list[np.ndarray],
        neighbor_distances: list[int] | None,
    ) -> np.ndarray:
        """Weighted average of inpainted backgrounds.

        The target's own background is weighted highest (configurable);
        each neighbor is weighted by 1 / max(1, |distance|) so closer
        frames contribute more.
        """
        if neighbor_distances is None:
            neighbor_distances = [1] * len(neighbor_bgs)
        if len(neighbor_distances) != len(neighbor_bgs):
            raise ValueError("neighbor_distances length mismatch")

        imgs = [target_bg.astype(np.float32)]
        weights = [self.cfg.neighbor_self_weight]

        for bg, dt in zip(neighbor_bgs, neighbor_distances):
            if bg.shape != target_bg.shape:
                continue
            imgs.append(bg.astype(np.float32))
            weights.append(1.0 / max(1, abs(int(dt))))

        weights_arr = np.asarray(weights, dtype=np.float32)
        weights_arr /= weights_arr.sum()

        acc = np.zeros_like(imgs[0])
        for img, w in zip(imgs, weights_arr):
            acc += img * w
        return np.clip(acc, 0, 255).astype(np.uint8)
