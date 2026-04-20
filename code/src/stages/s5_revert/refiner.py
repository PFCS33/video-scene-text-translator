"""S5 alignment refiner inference wrapper.

Wraps ``src.models.refiner.ROIRefiner`` for inference inside the S5 stage.
Handles:

- Lazy model load (no torch imports at module level, matching BPN/AnyText2).
- Scale handling between the network's fixed ``(H_net, W_net)`` input
  size and each track's actual canonical ROI size.
- ΔH composition direction: the model predicts a forward homography in
  network pixel coords; we unscale to canonical pixels, build the 3x3
  ``ΔH`` there, and return it for composition into the existing S5 warp
  chain.
- Runtime sanity checks that fall back to ``None`` on bad predictions
  (see plan.md §2.5). The S5 stage then uses its existing refiner-disabled
  path — the refiner **must never crash the pipeline**.

Direction convention
--------------------
The trained model is contracted to produce ``Δcorners`` such that
``warp_image(ref_canonical, ΔH) ≈ target_canonical`` (see
``src/models/refiner/README.md``). Equivalently, ``ΔH`` is a forward
homography in canonical pixel space mapping a feature's position in
ref_canonical coordinates to its position in target_canonical coordinates.

S5 composes this into the existing warp chain as::

    H_adjusted = T @ H_from_frontal[tgt] @ ΔH

(no inverse — because ``cv2.warpPerspective`` without ``WARP_INVERSE_MAP``
treats its matrix as a **forward** src→dst homography and internally
computes the inverse for sampling). The direction is pinned by a unit
test in test_s5_revert.py.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RefinerInference:
    """Lazy-loaded inference wrapper around a trained ``ROIRefiner``."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        image_size: tuple[int, int] = (64, 128),
        max_corner_offset_px: float = 16.0,
        use_gate: bool = True,
        score_margin: float = 0.01,
    ):
        """
        Args:
            checkpoint_path: Path to a ``.pt`` saved by
                ``src.models.refiner.train``. The checkpoint's embedded
                config is used to reconstruct the model architecture, so
                this wrapper is robust to architecture changes between
                training and inference.
            device: ``"cuda"`` or ``"cpu"``. Falls back to CPU if CUDA
                is unavailable at load time.
            image_size: ``(H, W)`` network input size. Used as a fallback
                if the checkpoint doesn't embed it; the checkpoint's
                embedded size takes precedence when present.
            max_corner_offset_px: Predictions with any corner offset
                larger than this (in canonical pixels) are rejected.
                Catches clearly-insane predictions from the very rare
                degenerate input.
            use_gate: When True (default), enable the do-no-harm gate:
                after sanity checks pass, score the alignment under
                identity vs ΔH using masked NCC on luminance, and
                only return ΔH if it strictly improves the score by
                ``score_margin``. Catches "good cases get worse" failure
                mode where the network over-corrects on already-aligned
                pairs.
            score_margin: Minimum NCC improvement required to apply ΔH
                when the gate is enabled. Default 0.01 (matches the
                production config default in RevertConfig). Set to 0.0
                for a more permissive "any improvement is good enough"
                gate, or raise further to be more conservative.
        """
        self.checkpoint_path = checkpoint_path
        self.device_str = device
        self.image_size = image_size  # (H, W), network input
        self.max_corner_offset_px = float(max_corner_offset_px)
        self.use_gate = bool(use_gate)
        self.score_margin = float(score_margin)

        # Lazy — not loaded until first predict_delta_H call.
        self._model: Any = None
        self._device: Any = None
        self._src_corners_net: np.ndarray | None = None
        # Cached center-weight masks keyed by (H, W) — same canonical size
        # repeats across many detections of one track, so caching is cheap.
        self._center_weight_cache: dict[tuple[int, int], np.ndarray] = {}

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch  # local import so module-level loads don't need torch

        from src.models.refiner.model import ROIRefiner

        logger.info("S5 refiner: loading %s", self.checkpoint_path)
        device_str = self.device_str
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "S5 refiner: CUDA requested but unavailable; falling back to CPU"
            )
            device_str = "cpu"
        device = torch.device(device_str)

        # Wrap torch.load so a slow/stuck checkpoint read is visible. This
        # is a one-shot first-call blocker — e.g. a cold disk read on a
        # large .pt file can silently consume tens of seconds with no
        # other signal. Re-raise so the normal load-failure path fires.
        t_load_start = time.monotonic()
        try:
            ckpt = torch.load(
                self.checkpoint_path, map_location=device, weights_only=False,
            )
        except Exception:
            elapsed = time.monotonic() - t_load_start
            logger.exception(
                "S5 refiner: torch.load(%s) failed after %.1fs",
                self.checkpoint_path, elapsed,
            )
            raise
        logger.info(
            "S5 refiner: loaded %s in %.1fs",
            self.checkpoint_path, time.monotonic() - t_load_start,
        )
        cfg = ckpt.get("config", {})
        mc = cfg.get("model", {})
        dc = cfg.get("data", {})
        image_size = tuple(dc.get("image_size", list(self.image_size)))
        model = ROIRefiner(
            base_channels=mc.get("base_channels", 32),
            dropout=mc.get("dropout", 0.2),
            image_size=image_size,
            head_init_scale=mc.get("head_init_scale", 1e-3),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        self._model = model
        self._device = device
        self.image_size = image_size

        H_net, W_net = image_size
        self._src_corners_net = np.array(
            [[0, 0], [W_net, 0], [W_net, H_net], [0, H_net]],
            dtype=np.float32,
        )
        logger.info(
            "S5 refiner: ready (device=%s, image_size=%s, %d params)",
            device, image_size, model.num_parameters(),
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_delta_H(
        self,
        ref_canonical: np.ndarray,
        target_canonical: np.ndarray,
    ) -> np.ndarray | None:
        """Predict the residual homography ΔH in canonical pixel space.

        Args:
            ref_canonical: ``(H_can, W_can, 3)`` uint8 BGR — reference
                frame warped to canonical frontal.
            target_canonical: ``(H_can, W_can, 3)`` uint8 BGR — target
                frame warped to canonical frontal. Same spatial size as
                ``ref_canonical``.

        Returns:
            ``(3, 3)`` float64 forward homography in canonical pixel
            coords such that feature at ref pixel ``p`` maps to
            ``ΔH @ p`` in target canonical coords. Returns ``None`` if
            any sanity check fails — callers must fall back to the
            refiner-disabled warp chain.
        """
        if ref_canonical is None or target_canonical is None:
            return None
        if ref_canonical.ndim != 3 or ref_canonical.shape[2] != 3:
            logger.debug("S5 refiner: bad ref shape %s", ref_canonical.shape)
            return None
        if target_canonical.ndim != 3 or target_canonical.shape[2] != 3:
            logger.debug("S5 refiner: bad target shape %s", target_canonical.shape)
            return None
        if ref_canonical.shape[:2] != target_canonical.shape[:2]:
            logger.debug(
                "S5 refiner: ref/target size mismatch %s vs %s",
                ref_canonical.shape[:2], target_canonical.shape[:2],
            )
            return None

        H_can, W_can = ref_canonical.shape[:2]
        if H_can == 0 or W_can == 0:
            return None

        self._ensure_loaded()
        import torch

        H_net, W_net = self.image_size

        # Resize to network input. cv2 uses BGR uint8 which is fine —
        # the model was trained on RGB in [0, 1] but luminance / NCC /
        # convolution features are close enough under a color swap that
        # the pretrained model generalizes. To be safe, convert to RGB.
        ref_net = cv2.resize(ref_canonical, (W_net, H_net), interpolation=cv2.INTER_LINEAR)
        tgt_net = cv2.resize(target_canonical, (W_net, H_net), interpolation=cv2.INTER_LINEAR)
        ref_rgb = cv2.cvtColor(ref_net, cv2.COLOR_BGR2RGB)
        tgt_rgb = cv2.cvtColor(tgt_net, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            src_t = (
                torch.from_numpy(ref_rgb).permute(2, 0, 1).float().div(255.0)
                .unsqueeze(0).to(self._device)
            )
            tgt_t = (
                torch.from_numpy(tgt_rgb).permute(2, 0, 1).float().div(255.0)
                .unsqueeze(0).to(self._device)
            )
            delta_net = self._model(src_t, tgt_t)  # (1, 4, 2) network pixels
            delta_net_np = delta_net[0].cpu().numpy().astype(np.float64)

        if not np.all(np.isfinite(delta_net_np)):
            logger.debug("S5 refiner: non-finite network output, rejecting")
            return None

        # Unscale: network predicts in (W_net, H_net) pixels; convert to
        # canonical pixel units. Note order: delta columns are (dx, dy).
        scale = np.array([W_can / W_net, H_can / H_net], dtype=np.float64)
        delta_canonical = delta_net_np * scale

        # Max offset sanity check (plan §2.5 check 1).
        max_off = float(np.abs(delta_canonical).max())
        if max_off > self.max_corner_offset_px:
            logger.debug(
                "S5 refiner: reject — max corner offset %.2f > %.2f px",
                max_off, self.max_corner_offset_px,
            )
            return None

        # Build ΔH in canonical pixel space via cv2 (numerically matches
        # our own corners_to_H by construction).
        src_corners_can = np.array(
            [[0, 0], [W_can, 0], [W_can, H_can], [0, H_can]],
            dtype=np.float32,
        )
        dst_corners_can = (src_corners_can + delta_canonical).astype(np.float32)
        try:
            delta_H = cv2.getPerspectiveTransform(src_corners_can, dst_corners_can)
        except cv2.error as e:
            logger.debug("S5 refiner: getPerspectiveTransform failed: %s", e)
            return None

        # Finiteness check (plan §2.5 check 4).
        if not np.all(np.isfinite(delta_H)):
            logger.debug("S5 refiner: non-finite ΔH, rejecting")
            return None

        # Determinant sanity check (plan §2.5 check 2). For a valid
        # near-identity homography this should be ~1.
        det = float(np.linalg.det(delta_H))
        if not (0.5 <= det <= 2.0):
            logger.debug("S5 refiner: reject — det(ΔH) = %.4f", det)
            return None

        # Condition number sanity check (plan §2.5 check 3).
        try:
            cond = float(np.linalg.cond(delta_H))
        except np.linalg.LinAlgError:
            logger.debug("S5 refiner: cond(ΔH) raised LinAlgError, rejecting")
            return None
        if not np.isfinite(cond) or cond > 1e4:
            logger.debug("S5 refiner: reject — cond(ΔH) = %.2e", cond)
            return None

        # ----- Do-no-harm gate (Tier 1 improvement) -----
        # Score the alignment under identity vs. ΔH using masked NCC on
        # luminance, weighted by a fixed center-feather mask. Only apply
        # ΔH if it strictly improves the score by `score_margin`.
        # Catches the failure mode where the network over-corrects on
        # already-aligned pairs and adds visible jitter.
        #
        # Wrapped in try/except so a numerical hiccup leaves the rest of
        # the path intact — gate failure means "fall back to applying
        # ΔH without gating", not "drop the prediction".
        if self.use_gate:
            try:
                weight = self._get_center_weight(H_can, W_can)
                refined_ref = cv2.warpPerspective(
                    ref_canonical, delta_H, (W_can, H_can),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
                score_identity = self._masked_ncc_luminance(
                    ref_canonical, target_canonical, weight,
                )
                score_refined = self._masked_ncc_luminance(
                    refined_ref, target_canonical, weight,
                )
                gain = score_refined - score_identity
                if gain < self.score_margin:
                    logger.debug(
                        "S5 refiner: gate rejected ΔH (refined NCC %.5f - "
                        "identity NCC %.5f = %.5f < margin %.5f)",
                        score_refined, score_identity, gain, self.score_margin,
                    )
                    return None
                else:
                    logger.debug(
                        "S5 refiner: gate accepted ΔH (gain %+.5f, "
                        "refined %.5f vs identity %.5f)",
                        gain, score_refined, score_identity,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "S5 refiner: gate computation raised %s; failing open",
                    exc,
                )

        return delta_H

    # ------------------------------------------------------------------
    # Do-no-harm gate helpers
    # ------------------------------------------------------------------

    def _get_center_weight(self, h: int, w: int) -> np.ndarray:
        """Cached soft rectangular feather mask, ``(H, W)`` float32.

        1.0 in the interior, fading linearly to 0.1 over the outer 10%
        of each axis. Mirrors ``losses.center_weight_map`` so the gate's
        scoring weight matches the training-time weighting.
        """
        key = (int(h), int(w))
        cached = self._center_weight_cache.get(key)
        if cached is not None:
            return cached
        border_frac = 0.1
        edge_value = 0.1
        y = np.arange(h, dtype=np.float32)
        x = np.arange(w, dtype=np.float32)
        d_y = np.minimum(y, (h - 1) - y)
        d_x = np.minimum(x, (w - 1) - x)
        band_y = max(1, int(h * border_frac))
        band_x = max(1, int(w * border_frac))
        w_y = np.clip(d_y / band_y, 0.0, 1.0) * (1.0 - edge_value) + edge_value
        w_x = np.clip(d_x / band_x, 0.0, 1.0) * (1.0 - edge_value) + edge_value
        mask = (w_y[:, None] * w_x[None, :]).astype(np.float32)
        self._center_weight_cache[key] = mask
        return mask

    @staticmethod
    def _masked_ncc_luminance(
        a: np.ndarray, b: np.ndarray, weight: np.ndarray,
    ) -> float:
        """Masked normalized cross-correlation of two BGR uint8 images
        on the BT.601 luminance channel.

        Returns a scalar in ``[-1, 1]``, higher is better aligned.
        Uses float64 internally so the gate isn't sensitive to
        accumulated float32 noise on small ROIs.
        """
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
        # BGR -> luminance via BT.601 weights, matching losses.luminance.
        # cv2.cvtColor uses the same coefficients for COLOR_BGR2GRAY.
        a_lum = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        b_lum = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        w = weight.astype(np.float64)
        w_sum = float(w.sum()) + 1e-12
        a_mean = float((w * a_lum).sum()) / w_sum
        b_mean = float((w * b_lum).sum()) / w_sum
        a_c = a_lum - a_mean
        b_c = b_lum - b_mean
        cov = float((w * a_c * b_c).sum()) / w_sum
        var_a = float((w * a_c * a_c).sum()) / w_sum
        var_b = float((w * b_c * b_c).sum()) / w_sum
        denom = float(np.sqrt(var_a * var_b)) + 1e-12
        ncc = cov / denom
        return float(np.clip(ncc, -1.0, 1.0))
