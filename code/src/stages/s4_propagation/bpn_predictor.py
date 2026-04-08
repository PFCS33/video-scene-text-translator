"""BPN predictor wrapping the trained Background Prediction Network.

Loads a BPN checkpoint and exposes a clean BGR-uint8 in / BGR-uint8 out
API for use inside the propagation stage. The network predicts blur
parameters (sigma_x, sigma_y, rho, w) per neighbor frame; the predictor
class also applies them via the existing DifferentiableBlur module.

Two important details for inference:

1. **Sigma rescaling.** The BPN was trained at a fixed small resolution
   (typically 64x128). Sigmas are in *pixel units* — a sigma of 2.0 at
   training res represents a different physical blur than 2.0 at the
   larger inference resolution. We rescale predicted sigmas by
   ``inference_height / training_height`` so the kernel covers an
   equivalent fraction of the image. rho and w are scale-independent.

2. **Chunked prediction.** The model has a fixed input window of
   ``n_neighbors`` targets. When a track has more frames than that, we
   tile the targets into non-overlapping chunks of size n_neighbors,
   pad the last chunk with the final target if needed, run the model
   once per chunk, and stitch the outputs back together. This mirrors
   how evaluate.py visualizes long tracks.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from src.models.bpn import BPN, DifferentiableBlur


class BPNPredictor:
    """Inference-time wrapper around the trained BPN."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        n_neighbors: int = 3,
        image_size: tuple[int, int] = (64, 128),
        kernel_size: int = 41,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            checkpoint_path: Path to the trained BPN checkpoint (.pt).
            n_neighbors: Must match the value used during training.
            image_size: (H, W) the network was trained at. Used both for
                resizing inputs and for computing the sigma rescaling
                factor at inference.
            kernel_size: Differentiable blur kernel size in pixels at
                inference resolution. Should be large enough to cover
                ~6*sigma at the maximum predicted sigma after rescaling.
            device: torch device.
        """
        self.device = torch.device(device)
        self.n_neighbors = n_neighbors
        self.training_size = image_size  # (H, W)
        self.kernel_size = kernel_size

        self.model = BPN(n_neighbors=n_neighbors, pretrained=False).to(self.device)
        ckpt = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.blur = DifferentiableBlur(kernel_size=kernel_size).to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_params(
        self,
        ref_canonical: np.ndarray,
        target_canonicals: list[np.ndarray],
    ) -> dict[str, torch.Tensor]:
        """Predict per-target blur parameters with one batched forward pass.

        Args:
            ref_canonical: BGR uint8 (H, W, 3) — the *original* reference
                frame ROI in canonical space.
            target_canonicals: List of BGR uint8 (H, W, 3) — the
                original target frame ROIs in canonical space.

        Returns:
            Dict with float tensors of shape (n_targets,) for each of
            ``sigma_x``, ``sigma_y``, ``rho``, ``w``. Sigmas are already
            rescaled to inference (native ROI) pixel units.
        """
        n_targets = len(target_canonicals)
        if n_targets == 0:
            return {
                k: torch.empty(0, device=self.device)
                for k in ("sigma_x", "sigma_y", "rho", "w")
            }

        ref_h, ref_w = ref_canonical.shape[:2]
        for t in target_canonicals:
            if t.shape[:2] != (ref_h, ref_w):
                raise ValueError(
                    f"BPNPredictor: target shape {t.shape[:2]} does not match "
                    f"reference shape {(ref_h, ref_w)}"
                )

        # Resize ref + targets to the network's training resolution.
        train_h, train_w = self.training_size
        ref_small = self._to_tensor(cv2.resize(ref_canonical, (train_w, train_h)))
        targets_small = [
            self._to_tensor(cv2.resize(t, (train_w, train_h)))
            for t in target_canonicals
        ]

        sigma_x_all = torch.empty(n_targets, device=self.device)
        sigma_y_all = torch.empty(n_targets, device=self.device)
        rho_all = torch.empty(n_targets, device=self.device)
        w_all = torch.empty(n_targets, device=self.device)

        # Non-overlapping chunks of size n_neighbors; pad last chunk by
        # repeating the final target.
        for chunk_start in range(0, n_targets, self.n_neighbors):
            chunk_end = min(chunk_start + self.n_neighbors, n_targets)
            chunk_len = chunk_end - chunk_start
            chunk = targets_small[chunk_start:chunk_end]
            if chunk_len < self.n_neighbors:
                chunk = chunk + [chunk[-1]] * (self.n_neighbors - chunk_len)

            stacked = torch.cat([ref_small] + chunk, dim=0).unsqueeze(0)
            out = self.model(stacked)

            sigma_x_all[chunk_start:chunk_end] = out["sigma_x"][0, :chunk_len]
            sigma_y_all[chunk_start:chunk_end] = out["sigma_y"][0, :chunk_len]
            rho_all[chunk_start:chunk_end] = out["rho"][0, :chunk_len]
            w_all[chunk_start:chunk_end] = out["w"][0, :chunk_len]

        # Sigma rescaling: predicted in training-resolution pixels,
        # applied at native resolution.
        scale = ref_h / train_h
        sigma_x_all = sigma_x_all * scale
        sigma_y_all = sigma_y_all * scale

        return {
            "sigma_x": sigma_x_all,
            "sigma_y": sigma_y_all,
            "rho": rho_all,
            "w": w_all,
        }

    @torch.no_grad()
    def apply_blur(
        self,
        image: np.ndarray,
        sigma_x: torch.Tensor,
        sigma_y: torch.Tensor,
        rho: torch.Tensor,
        w: torch.Tensor,
    ) -> np.ndarray:
        """Apply a single set of blur params to one BGR image.

        sigma_x, sigma_y, rho, w should each be 0-d or 1-d tensors with
        a single element (one target).
        """
        x = self._to_tensor(image).unsqueeze(0)  # (1, 3, H, W)
        sx = sigma_x.view(1).to(self.device)
        sy = sigma_y.view(1).to(self.device)
        r = rho.view(1).to(self.device)
        ww = w.view(1).to(self.device)
        blurred = self.blur(x, sx, sy, r, ww).squeeze(0).clamp(0, 1)
        arr = (blurred.cpu().numpy() * 255).astype(np.uint8)
        arr = np.transpose(arr, (1, 2, 0))  # (H, W, 3) RGB
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _to_tensor(self, bgr: np.ndarray) -> torch.Tensor:
        """BGR uint8 (H, W, 3) -> RGB float [0, 1] (3, H, W) on device."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return t.to(self.device)
