"""Loss functions for BPN training.

From STRIVE Section 3.2:
    L_BPN = lambda_psi * L_psi + lambda_R * L_R + lambda_T * L_T

- L_psi: parameter regression loss (Stage 1 only, with synthetic data)
- L_R: reconstruction loss (both stages)
- L_T: temporal consistency loss on predicted parameters (both stages)
"""

import torch
import torch.nn as nn

from .blur import DifferentiableBlur


class BPNLoss(nn.Module):
    """Combined BPN training loss."""

    def __init__(
        self,
        blur_module: DifferentiableBlur,
        lambda_psi: float = 1.0,
        lambda_recon: float = 1.0,
        lambda_temporal: float = 0.5,
        use_psi_loss: bool = True,
    ):
        super().__init__()
        self.blur = blur_module
        self.lambda_psi = lambda_psi
        self.lambda_recon = lambda_recon
        self.lambda_temporal = lambda_temporal
        self.use_psi_loss = use_psi_loss

    def forward(
        self,
        pred_params: dict[str, torch.Tensor],
        ref_image: torch.Tensor,
        neighbor_images: torch.Tensor,
        gt_params: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pred_params: dict with sigma_x, sigma_y, rho, w each (B, N)
            ref_image: (B, 3, H, W) reference ROI
            neighbor_images: (B, N, 3, H, W) target neighbor ROIs
            gt_params: optional ground truth params for Stage 1

        Returns:
            dict with total_loss and individual loss components
        """
        B, N, C, H, W = neighbor_images.shape
        losses = {}

        # -- Reconstruction loss L_R --
        # Apply predicted blur to reference for each neighbor
        recon_loss = torch.tensor(0.0, device=ref_image.device)
        for i in range(N):
            blurred = self.blur(
                ref_image,
                pred_params["sigma_x"][:, i],
                pred_params["sigma_y"][:, i],
                pred_params["rho"][:, i],
                pred_params["w"][:, i],
            )
            recon_loss = recon_loss + nn.functional.mse_loss(
                blurred, neighbor_images[:, i]
            )
        recon_loss = recon_loss / N
        losses["recon"] = recon_loss

        # -- Temporal consistency loss L_T --
        # Encourage smooth parameter changes across neighbors
        temporal_loss = torch.tensor(0.0, device=ref_image.device)
        if N > 1:
            for key in ("sigma_x", "sigma_y", "rho", "w"):
                p = pred_params[key]  # (B, N)
                # Sum of squared diffs between consecutive neighbors
                diffs = (p[:, 1:] - p[:, :-1]) ** 2
                temporal_loss = temporal_loss + diffs.mean()
            temporal_loss = temporal_loss / 4.0
        losses["temporal"] = temporal_loss

        # -- Parameter regression loss L_psi (Stage 1 only) --
        # Each parameter is normalized by its natural scale so all four
        # contribute comparable gradient magnitudes. Without this, w (range
        # ~[-0.8, 0.8]) gets dominated by sigma (range ~[0.5, 4]) and rho
        # (range ~[-pi, pi]), and the network learns to predict w=0.
        param_scales = {
            "sigma_x": 1.8,
            "sigma_y": 1.8,
            "rho": 3.14159,
            "w": 0.4,
        }
        psi_loss = torch.tensor(0.0, device=ref_image.device)
        if self.use_psi_loss and gt_params is not None:
            for key in ("sigma_x", "sigma_y", "rho", "w"):
                scale = param_scales[key]
                diff = (pred_params[key] - gt_params[key]) / scale
                psi_loss = psi_loss + (diff ** 2).mean()
            psi_loss = psi_loss / 4.0
        losses["psi"] = psi_loss

        # -- Total loss --
        total = (self.lambda_recon * recon_loss
                 + self.lambda_temporal * temporal_loss)
        if self.use_psi_loss and gt_params is not None:
            total = total + self.lambda_psi * psi_loss
        losses["total"] = total

        return losses
