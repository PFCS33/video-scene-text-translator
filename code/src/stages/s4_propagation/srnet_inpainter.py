"""SRNet-based background inpainter (lksshw/SRNet PyTorch reimplementation).

Loads only the background-inpainting subnetwork (`Generator._bin`) from a
SRNet checkpoint and exposes a clean BGR-uint8 in / BGR-uint8 out API.
The other SRNet subnetworks (text conversion, fusion, discriminators)
are loaded into `Generator` but never executed.

Compatibility notes (see chat history for full analysis):
- Only `torch` + `torchvision` are real runtime dependencies (already in venv).
- We `sys.path` into third_party/SRNet/ at construction time so the
  upstream `model.py` can `import cfg`. We do NOT touch their requirements.
- `torch.load(weights_only=False)` is required because the checkpoint
  contains optimizer state and arbitrary pickled objects.
- The SRNet model uses a broken `temp_shape=(0,0)` global at construction
  time, which makes its conv layers use `padding=0`. The released weights
  were trained at H=64 with W as a multiple of 8, so we resize inputs to
  match before inference and resize the output back to the caller's shape.
- Color order: SRNet was trained on RGB (skimage). We BGR↔RGB convert at
  the boundary so the rest of our pipeline can stay BGR.
- Normalization: SRNet expects inputs in [-1, 1] (`x/127.5 - 1`) and
  outputs in [-1, 1] (tanh). We map back to uint8 via `(x+1)/2 * 255`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from .base_inpainter import BaseBackgroundInpainter

# Path to the lksshw/SRNet repo. Resolved relative to the repo root so
# the wrapper works regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRNET_DIR = _REPO_ROOT / "third_party" / "SRNet"


class SRNetInpainter(BaseBackgroundInpainter):
    """Background inpainter wrapping SRNet's `_bin` subnetwork."""

    # SRNet expects this fixed input height; width must be a multiple of
    # this stride to play nicely with three stride-2 down/up samplings.
    INPUT_HEIGHT = 64
    WIDTH_STRIDE = 8

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            checkpoint_path: Path to the SRNet `.model` checkpoint
                (contains the full Generator state dict). If None, the
                model is left uninitialized — call :meth:`load_model` later.
            device: torch device string.
        """
        self.device = torch.device(device)
        self._generator: torch.nn.Module | None = None
        if checkpoint_path is not None:
            self.load_model(str(checkpoint_path), device=str(self.device))

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    def load_model(self, model_path: str, device: str = "cpu") -> None:
        """Construct the Generator and load the inpainting weights."""
        self.device = torch.device(device)

        # Add SRNet to sys.path so its `import cfg` works. Insert at the
        # front so we don't accidentally import a stale `model` from elsewhere.
        srnet_path = str(_SRNET_DIR)
        if srnet_path not in sys.path:
            sys.path.insert(0, srnet_path)

        # Import here, after sys.path is set, to avoid polluting the
        # module-level namespace and to defer the (slow) torchvision
        # vgg19 import path until actually needed.
        from model import Generator  # type: ignore[import-not-found]

        gen = Generator(in_channels=3)
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        # The lksshw/SRNet checkpoint has the layout:
        #   {'generator': state_dict, 'discriminator1': ..., 'g_optimizer': ..., ...}
        # We only need the generator's state_dict; everything else is ignored.
        gen.load_state_dict(ckpt["generator"])
        gen.to(self.device)
        gen.eval()
        self._generator = gen

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inpaint(self, canonical_roi: np.ndarray) -> np.ndarray:
        """Erase text from a canonical-frontal ROI.

        Args:
            canonical_roi: BGR uint8 (H, W, 3).

        Returns:
            BGR uint8 (H, W, 3) — same shape as input — with text removed.
        """
        if self._generator is None:
            raise RuntimeError(
                "SRNetInpainter has no weights loaded. Pass checkpoint_path "
                "to __init__ or call load_model()."
            )
        if canonical_roi.ndim != 3 or canonical_roi.shape[2] != 3:
            raise ValueError(
                f"Expected (H, W, 3) BGR image, got shape {canonical_roi.shape}"
            )

        orig_h, orig_w = canonical_roi.shape[:2]

        # Resize to SRNet's expected input shape: H=64, W = round(W*64/H/8)*8.
        scale = self.INPUT_HEIGHT / orig_h
        to_h = self.INPUT_HEIGHT
        to_w = max(self.WIDTH_STRIDE,
                   int(round(orig_w * scale / self.WIDTH_STRIDE)) * self.WIDTH_STRIDE)
        resized = cv2.resize(canonical_roi, (to_w, to_h), interpolation=cv2.INTER_LINEAR)

        # BGR uint8 -> RGB float [-1, 1] -> (1, 3, H, W) tensor
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 127.5) - 1.0
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Run only the inpainting subnetwork (not the full Generator).
        # `_bin` returns (background, fuse_features); we discard fuse.
        o_b, _ = self._generator._bin(x)

        # tanh output [-1, 1] -> uint8 BGR at original size
        o_b = o_b.squeeze(0).clamp(-1.0, 1.0).cpu().numpy()
        o_b = ((o_b + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        o_b = np.transpose(o_b, (1, 2, 0))  # (3, H, W) -> (H, W, 3) RGB
        bgr = cv2.cvtColor(o_b, cv2.COLOR_RGB2BGR)

        if (bgr.shape[0], bgr.shape[1]) != (orig_h, orig_w):
            bgr = cv2.resize(bgr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return bgr
