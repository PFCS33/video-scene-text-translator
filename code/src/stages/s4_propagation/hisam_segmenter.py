"""Hi-SAM text stroke segmenter (vendored at ``third_party/Hi-SAM/``).

Thin wrapper around Hi-SAM's `SamPredictor` that produces a binary
text-stroke mask for a BGR canonical ROI. The mask is consumed by
`SegmentationBasedInpainter` (Step 4) as input to ``cv2.inpaint``.

Compatibility notes:
- Only ``torch``, ``torchvision``, ``einops``, ``shapely``, ``pyclipper``,
  ``scikit-image``, ``scipy``, ``pillow`` are real runtime dependencies
  (all already installed in the project venv — confirmed by the earlier
  GPU smoke test).
- Hi-SAM's ``hi_sam/modeling/build.py`` hardcodes a **relative** path to
  SAM's ViT encoder checkpoint (e.g. ``pretrained_checkpoint/sam_vit_l_*.pth``).
  To make it resolvable regardless of caller cwd we ``contextlib.chdir``
  into the Hi-SAM repo root during model construction only — the chdir is
  restored immediately after. Inference (`segment`) runs from any cwd.
- ``sys.path`` is lazily extended with the Hi-SAM repo root so
  ``from hi_sam.modeling.build import model_registry`` resolves.
- Color order: Hi-SAM's ``SamPredictor.set_image`` natively accepts
  ``image_format="BGR"`` and converts internally — no BGR/RGB juggling
  required on our side.
- ``segment(bgr_roi)`` returns a ``uint8`` ``H x W`` mask with values in
  ``{0, 255}`` (same spatial shape as the input).
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np

# Path to the vendored Hi-SAM repo. Resolved relative to the repo root so the
# wrapper works regardless of CWD.
#   __file__ = .../code/src/stages/s4_propagation/hisam_segmenter.py
#   parents[0] = .../s4_propagation
#   parents[1] = .../stages
#   parents[2] = .../src
#   parents[3] = .../code
#   parents[4] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[4]
_HISAM_DIR = _REPO_ROOT / "third_party" / "Hi-SAM"


# ---------------------------------------------------------------------------
# Sliding-window patch helpers (copied verbatim from Hi-SAM's ``demo_hisam.py``
# so we don't depend on that script as an importable module). Used only when
# ``use_patch_mode=True``.
# ---------------------------------------------------------------------------


def _patchify_sliding(
    image: np.ndarray, patch_size: int = 512, stride: int = 384
) -> tuple[list[np.ndarray], list[slice], list[slice]]:
    """Split an image into overlapping patches for multi-pass inference.

    Mirrors ``demo_hisam.patchify_sliding``.
    """
    h, w = image.shape[:2]
    patch_list: list[np.ndarray] = []
    h_slice_list: list[slice] = []
    w_slice_list: list[slice] = []
    for j in range(0, h, stride):
        start_h, end_h = j, j + patch_size
        if end_h > h:
            start_h = max(h - patch_size, 0)
            end_h = h
        for i in range(0, w, stride):
            start_w, end_w = i, i + patch_size
            if end_w > w:
                start_w = max(w - patch_size, 0)
                end_w = w
            h_slice = slice(start_h, end_h)
            w_slice = slice(start_w, end_w)
            h_slice_list.append(h_slice)
            w_slice_list.append(w_slice)
            patch_list.append(image[h_slice, w_slice])
    return patch_list, h_slice_list, w_slice_list


def _unpatchify_sliding(
    patch_list: list[np.ndarray],
    h_slice_list: list[slice],
    w_slice_list: list[slice],
    ori_size: tuple[int, int],
) -> np.ndarray:
    """Accumulate overlapping patches by summing into a full logits map.

    Mirrors ``demo_hisam.unpatchify_sliding``. The sum is later thresholded
    — since each pixel may be covered by >1 patch the threshold naturally
    handles overlap.
    """
    if len(ori_size) != 2:
        raise ValueError(f"ori_size must be (H, W), got {ori_size!r}")
    whole_logits = np.zeros(ori_size, dtype=np.float32)
    if not (len(patch_list) == len(h_slice_list) == len(w_slice_list)):
        raise ValueError(
            f"patch_list / h_slice_list / w_slice_list length mismatch: "
            f"{len(patch_list)} / {len(h_slice_list)} / {len(w_slice_list)}"
        )
    for patch, h_slice, w_slice in zip(
        patch_list, h_slice_list, w_slice_list, strict=True
    ):
        whole_logits[h_slice, w_slice] += patch
    return whole_logits


class HiSAMSegmenter:
    """Produces binary text-stroke masks for a canonical ROI via Hi-SAM."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cuda",
        model_type: str = "vit_l",
        use_patch_mode: bool = False,
    ) -> None:
        """
        Args:
            checkpoint_path: Path to the Hi-SAM head checkpoint
                (e.g. ``sam_tss_l_textseg.pth``). If ``None``, the model is
                left uninitialized — call :meth:`load_model` later.
            device: torch device string, e.g. ``"cuda"`` or ``"cpu"``.
            model_type: Hi-SAM backbone type — one of
                ``"vit_b"``, ``"vit_l"``, ``"vit_h"``.
            use_patch_mode: If True, run inference tile-by-tile with
                512x512 sliding windows (stride=384) and re-stitch logits.
                Useful for very large ROIs; off by default.
        """
        # Resolve the checkpoint path to absolute at construction time. The
        # chdir in ``load_model`` would otherwise break any caller-supplied
        # relative path.
        self._checkpoint_path: str | None = (
            str(Path(checkpoint_path).resolve()) if checkpoint_path is not None else None
        )
        self._device = device
        self._model_type = model_type
        self._use_patch_mode = use_patch_mode
        self._predictor = None  # type: ignore[assignment]
        if checkpoint_path is not None:
            self.load_model()

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Construct the Hi-SAM model + ``SamPredictor``.

        Idempotent — a second call is a no-op. Performs two side effects
        scoped to this call:

        - Inserts ``third_party/Hi-SAM`` into ``sys.path`` so
          ``from hi_sam.modeling.build import model_registry`` resolves.
        - Temporarily ``chdir``s into ``third_party/Hi-SAM`` during model
          construction so Hi-SAM's hardcoded relative path to the SAM ViT
          encoder weights resolves. The cwd is restored on exit (even on
          error) via ``contextlib.chdir``.

        Warning: ``contextlib.chdir`` mutates the process-global cwd and is
        not thread-safe. Call this method only from the main thread. If
        parallel model loading is ever needed, guard the chdir with a
        module-level ``threading.Lock`` or patch Hi-SAM's ``build.py`` to
        take an explicit encoder path instead.
        """
        if self._predictor is not None:
            return
        if self._checkpoint_path is None:
            raise RuntimeError(
                "HiSAMSegmenter has no checkpoint_path. Pass checkpoint_path "
                "to __init__ or set it before calling load_model()."
            )

        # Make Hi-SAM importable.
        hisam_path = str(_HISAM_DIR)
        if hisam_path not in sys.path:
            sys.path.insert(0, hisam_path)

        # Chdir is required only while the model constructor is loading the
        # SAM encoder via build.py's hardcoded relative path. Once weights
        # are in the module the cwd is no longer needed.
        with contextlib.chdir(_HISAM_DIR):
            # Imports are lazy to defer torch / Hi-SAM initialization until
            # this method runs, and to give the ``sys.path`` insertion above
            # a chance to take effect.
            from hi_sam.modeling.build import (
                model_registry,  # type: ignore[import-not-found]
            )
            from hi_sam.modeling.predictor import (
                SamPredictor,  # type: ignore[import-not-found]
            )

            args = argparse.Namespace(
                model_type=self._model_type,
                checkpoint=self._checkpoint_path,
                # We only need stroke segmentation; the H-Decoder head is not
                # loaded / used.
                hier_det=False,
                # Self-prompting head hyperparameters — demo_hisam defaults.
                attn_layers=1,
                prompt_len=12,
            )
            hisam = model_registry[self._model_type](args)
            hisam.eval()
            hisam.to(self._device)
            predictor = SamPredictor(hisam)

        self._predictor = predictor

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def segment(self, bgr_roi: np.ndarray) -> np.ndarray:
        """Produce a binary text-stroke mask for a canonical ROI.

        Args:
            bgr_roi: BGR uint8 image, shape ``(H, W, 3)``.

        Returns:
            uint8 ``(H, W)`` mask with values in ``{0, 255}``. 255 marks
            text-stroke pixels.
        """
        if self._predictor is None:
            raise RuntimeError(
                "HiSAMSegmenter has no weights loaded. Pass checkpoint_path "
                "to __init__ or call load_model() first."
            )
        if bgr_roi.ndim != 3 or bgr_roi.shape[2] != 3:
            raise ValueError(
                f"Expected (H, W, 3) BGR image, got shape {bgr_roi.shape}"
            )

        predictor = self._predictor
        if self._use_patch_mode:
            return self._segment_patch_mode(bgr_roi, predictor)
        return self._segment_single_pass(bgr_roi, predictor)

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _segment_single_pass(bgr_roi: np.ndarray, predictor) -> np.ndarray:  # noqa: ANN001
        """Single-forward-pass inference. Matches demo_hisam.py lines 232-244."""
        predictor.set_image(bgr_roi, image_format="BGR")
        _, hr_mask, _, _ = predictor.predict(multimask_output=False)
        # ``hr_mask`` is a (1, H, W) boolean mask already thresholded by the
        # predictor. Convert to uint8 {0, 255}.
        return (hr_mask[0].astype(np.uint8) * 255)

    @staticmethod
    def _segment_patch_mode(bgr_roi: np.ndarray, predictor) -> np.ndarray:  # noqa: ANN001
        """Sliding-window inference. Matches demo_hisam.py lines 217-230."""
        ori_size = bgr_roi.shape[:2]
        patch_list, h_slices, w_slices = _patchify_sliding(
            bgr_roi, patch_size=512, stride=384
        )
        logits_patches: list[np.ndarray] = []
        for patch in patch_list:
            predictor.set_image(patch, image_format="BGR")
            _, hr_logits, _, _ = predictor.predict(
                multimask_output=False, return_logits=True
            )
            if hr_logits.shape[0] != 1:
                raise RuntimeError(
                    f"Hi-SAM returned {hr_logits.shape[0]} logit maps; "
                    "expected 1 (multimask_output=False)"
                )
            logits_patches.append(hr_logits[0])
        full_logits = _unpatchify_sliding(logits_patches, h_slices, w_slices, ori_size)
        binary = full_logits > predictor.model.mask_threshold
        return (binary.astype(np.uint8) * 255)
