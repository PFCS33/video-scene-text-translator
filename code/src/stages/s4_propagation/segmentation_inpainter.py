"""Segmentation-based background inpainter (Hi-SAM + ``cv2.inpaint``).

Pipeline: BGR ROI -> Hi-SAM stroke mask -> optional dilation -> ``cv2.inpaint``
(Navier-Stokes or Telea) -> BGR ROI. Same BGR-uint8 in / BGR-uint8 out contract
as :class:`SRNetInpainter`, so the downstream LCM ratio-map code is unaffected.
Selected via ``propagation.inpainter_backend: "hisam"``.

The heavy lifting lives in :class:`HiSAMSegmenter` — this wrapper is the thin
layer that turns a binary stroke mask into a clean background via classical
inpainting.

Compatibility notes:
- ``cv2`` is imported as a module (not ``from cv2 import inpaint``) so tests
  can patch ``src.stages.s4_propagation.segmentation_inpainter.cv2.inpaint``.
- The underlying :class:`HiSAMSegmenter` is lazily constructed on the first
  call to :meth:`inpaint` — the expensive Hi-SAM checkpoint load is deferred
  until actually needed. Callers that want to inject a test double can pass
  ``segmenter=...`` to the constructor; in that case no :class:`HiSAMSegmenter`
  is ever constructed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .base_inpainter import BaseBackgroundInpainter
from .hisam_segmenter import HiSAMSegmenter

# Radius (in pixels) used as the ``inpaintRadius`` argument of ``cv2.inpaint``.
# 3 px is OpenCV's demo default and works well for thin text strokes.
_INPAINT_RADIUS_PX = 3

# 3x3 rectangular kernel used for mask dilation. Cached at module load so we
# don't rebuild it on every ``inpaint()`` call.
_DILATION_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Mapping from user-facing method name to the corresponding OpenCV flag.
_INPAINT_METHOD_FLAGS = {
    "ns": cv2.INPAINT_NS,
    "telea": cv2.INPAINT_TELEA,
}


class SegmentationBasedInpainter(BaseBackgroundInpainter):
    """Hi-SAM segmentation + ``cv2.inpaint`` classical background fill.

    See module docstring for the full pipeline.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cuda",
        model_type: str = "vit_l",
        mask_dilation_px: int = 3,
        inpaint_method: str = "ns",
        use_patch_mode: bool = False,
        segmenter: HiSAMSegmenter | None = None,
    ) -> None:
        """
        Args:
            checkpoint_path: Path to the Hi-SAM head checkpoint
                (e.g. ``sam_tss_l_textseg.pth``). Forwarded to
                :class:`HiSAMSegmenter` on first :meth:`inpaint` call.
            device: torch device string (``"cuda"`` or ``"cpu"``).
            model_type: Hi-SAM backbone type — one of
                ``"vit_b"``, ``"vit_l"``, ``"vit_h"``.
            mask_dilation_px: Number of dilation iterations applied to the
                stroke mask before ``cv2.inpaint``. ``0`` disables dilation.
                Defaults to ``3`` to absorb anti-aliased stroke halos.
            inpaint_method: ``"ns"`` (Navier-Stokes — default) or ``"telea"``
                (Fast Marching). Validated at construction time.
            use_patch_mode: Forwarded to :class:`HiSAMSegmenter`. If True,
                Hi-SAM runs tile-by-tile sliding-window inference.
            segmenter: Test-injection hook. If provided, the inpainter uses
                this instance as-is and will NOT construct a
                :class:`HiSAMSegmenter` internally. Production callers should
                leave this as ``None``.

        Raises:
            ValueError: ``inpaint_method`` is not ``"ns"`` or ``"telea"``, or
                ``mask_dilation_px`` is negative.
        """
        if inpaint_method not in _INPAINT_METHOD_FLAGS:
            raise ValueError(
                f"inpaint_method must be one of {sorted(_INPAINT_METHOD_FLAGS)}, "
                f"got {inpaint_method!r}"
            )
        if mask_dilation_px < 0:
            raise ValueError(
                f"mask_dilation_px must be >= 0, got {mask_dilation_px}"
            )

        self._checkpoint_path = checkpoint_path
        self._device = device
        self._model_type = model_type
        self._mask_dilation_px = int(mask_dilation_px)
        self._inpaint_flag = _INPAINT_METHOD_FLAGS[inpaint_method]
        self._use_patch_mode = use_patch_mode
        self._segmenter: HiSAMSegmenter | None = segmenter

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------

    def inpaint(self, canonical_roi: np.ndarray) -> np.ndarray:
        """Erase text from a canonical-frontal ROI via Hi-SAM + cv2.inpaint.

        Args:
            canonical_roi: BGR uint8 image, shape ``(H, W, 3)``.

        Returns:
            BGR uint8 image of the same shape as the input with text-stroke
            pixels replaced by a classical-inpainted background.

        Raises:
            ValueError: Input is not an ``(H, W, 3)`` uint8 array.
        """
        self._validate_input(canonical_roi)
        segmenter = self._ensure_segmenter()
        mask = segmenter.segment(canonical_roi)
        mask = self._dilate(mask)
        # Positional call: ``cv2.inpaint(src, mask, inpaintRadius, flags)``.
        # Tests patch this symbol with a ``side_effect`` that takes positional
        # args, so we must not use kwargs here.
        return cv2.inpaint(
            canonical_roi, mask, _INPAINT_RADIUS_PX, self._inpaint_flag
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_segmenter(self) -> HiSAMSegmenter:
        """Lazily construct the underlying :class:`HiSAMSegmenter`.

        Production callers hit this on the first :meth:`inpaint` call; a
        test-injected segmenter (``segmenter=...`` at construction time)
        short-circuits construction entirely.

        :class:`HiSAMSegmenter` eagerly calls ``load_model()`` when given a
        checkpoint path, so the explicit ``.load_model()`` here is
        idempotent — it also covers the edge case of an inpainter built with
        ``checkpoint_path=None`` that gets a checkpoint-aware segmenter
        assigned later.
        """
        if self._segmenter is None:
            self._segmenter = HiSAMSegmenter(
                checkpoint_path=self._checkpoint_path,
                device=self._device,
                model_type=self._model_type,
                use_patch_mode=self._use_patch_mode,
            )
            self._segmenter.load_model()
        return self._segmenter

    @staticmethod
    def _validate_input(roi: np.ndarray) -> None:
        """Raise ``ValueError`` on shape or dtype mismatch."""
        if roi.ndim != 3 or roi.shape[2] != 3:
            raise ValueError(
                f"Expected (H, W, 3) BGR image, got shape {roi.shape}"
            )
        if roi.dtype != np.uint8:
            raise ValueError(f"Expected uint8 dtype, got {roi.dtype}")

    def _dilate(self, mask: np.ndarray) -> np.ndarray:
        """Apply ``cv2.dilate`` if ``mask_dilation_px > 0``, else passthrough."""
        if self._mask_dilation_px <= 0:
            return mask
        return cv2.dilate(
            mask, _DILATION_KERNEL, iterations=self._mask_dilation_px
        )
