"""Loss functions for ROI alignment refiner training.

Layout:

    Primitives (pure functions):
        luminance            : RGB -> BT.601 Y channel
        sobel_magnitude      : per-pixel gradient magnitude
        center_weight_map    : fixed soft rectangular feather mask
        masked_charbonnier   : per-sample masked robust photometric loss
        masked_ncc           : per-sample masked normalized cross-correlation

    Top-level module:
        RefinerLoss          : composes the full training objective with
                               per-sample Type A / Type B routing.

Routing contract (matches plan.md §1.5):

    Type A (synthetic, has_gt_corners=True)
        * L_corner : smooth_l1 on predicted vs. GT corner offsets
        * L_recon  : masked Charbonnier on warped source vs. target (RGB)

    Type B (real, has_gt_corners=False)
        * L_ncc    : 1 - masked NCC on luminance (illumination-invariant)
        * L_grad   : masked Charbonnier on Sobel magnitudes (edge-structure)
        * L_reg    : mean squared corner offset (keeps ΔH near identity)

A batch may mix both types. Per-component values are averaged over the
appropriate subset of the batch and contribute zero when the subset is empty,
so a fully-synthetic or fully-real batch cleanly collapses to Stage 1 or
Stage 2 respectively.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .warp import canonical_corners, corners_to_H, warp_image, warp_validity_mask

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


# BT.601 luminance weights — standard for SDR video.
_LUMA_WEIGHTS = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)

_SOBEL_X = torch.tensor(
    [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
    dtype=torch.float32,
)
_SOBEL_Y = torch.tensor(
    [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
    dtype=torch.float32,
)


def luminance(rgb: torch.Tensor) -> torch.Tensor:
    """Convert ``(B, 3, H, W)`` RGB in [0, 1] to ``(B, 1, H, W)`` luminance."""
    if rgb.dim() != 4 or rgb.shape[1] != 3:
        raise ValueError(f"expected (B, 3, H, W), got {rgb.shape}")
    w = _LUMA_WEIGHTS.to(device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    return (rgb * w).sum(dim=1, keepdim=True)


def sobel_magnitude(img: torch.Tensor) -> torch.Tensor:
    """Per-pixel Sobel magnitude of a ``(B, C, H, W)`` image, returned as-is.

    Callers that want illumination-robust structure should pass a single-
    channel luminance image. Magnitude is used (not signed gradients) because
    it survives sign flips caused by brightness inversion.
    """
    if img.dim() != 4:
        raise ValueError(f"expected (B, C, H, W), got {img.shape}")
    C = img.shape[1]
    gx_kernel = _SOBEL_X.to(device=img.device, dtype=img.dtype)
    gy_kernel = _SOBEL_Y.to(device=img.device, dtype=img.dtype)
    # Replicate kernels per input channel for grouped conv.
    gx_kernel = gx_kernel.expand(C, 1, 3, 3).contiguous()
    gy_kernel = gy_kernel.expand(C, 1, 3, 3).contiguous()
    # Reflect padding so edge pixels don't get a spurious gradient from the
    # zero-padding boundary; the validity mask still handles warp boundaries.
    padded = F.pad(img, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(padded, gx_kernel, groups=C)
    gy = F.conv2d(padded, gy_kernel, groups=C)
    return torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-12)


def center_weight_map(
    height: int,
    width: int,
    border_frac: float = 0.1,
    edge_value: float = 0.1,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Soft rectangular center-weighting mask.

    Returns a ``(1, 1, H, W)`` tensor that is 1.0 in the interior and fades
    linearly to ``edge_value`` at the image boundary over the outer
    ``border_frac`` of each dimension. Used to downweight the bbox-fringe
    region that's more likely to contain scene background than text.

    Args:
        height: map height.
        width: map width.
        border_frac: fraction of each dim used for the falloff band.
        edge_value: weight at the outermost pixels (> 0 so edges still count).
        device, dtype: optional torch allocation knobs.
    """
    if not (0 < border_frac <= 0.5):
        raise ValueError(f"border_frac must be in (0, 0.5], got {border_frac}")
    if not (0 <= edge_value <= 1):
        raise ValueError(f"edge_value must be in [0, 1], got {edge_value}")
    dtype = dtype if dtype is not None else torch.float32
    y = torch.arange(height, dtype=dtype, device=device)
    x = torch.arange(width, dtype=dtype, device=device)
    # Distance to nearest horizontal / vertical edge, in pixels.
    d_y = torch.minimum(y, (height - 1) - y)
    d_x = torch.minimum(x, (width - 1) - x)
    band_y = max(1, int(height * border_frac))
    band_x = max(1, int(width * border_frac))
    w_y = (d_y / band_y).clamp(0, 1) * (1 - edge_value) + edge_value
    w_x = (d_x / band_x).clamp(0, 1) * (1 - edge_value) + edge_value
    return (w_y.view(-1, 1) * w_x.view(1, -1)).view(1, 1, height, width)


def masked_charbonnier(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Per-sample masked Charbonnier loss ``sqrt((x-y)^2 + eps^2)``.

    Args:
        x, y: ``(B, C, H, W)`` tensors.
        weight: ``(B, 1, H, W)`` broadcastable mask >= 0.
        eps: Charbonnier smoothing constant.

    Returns:
        ``(B,)`` per-sample scalar losses, normalized by mask sum per sample.
    """
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: x {x.shape} vs y {y.shape}")
    # Mean across channels first so C=1 and C=3 both produce comparable scale.
    diff = torch.sqrt((x - y).pow(2) + eps * eps).mean(dim=1, keepdim=True)
    numer = (diff * weight).flatten(1).sum(dim=1)
    denom = weight.flatten(1).sum(dim=1).clamp(min=1e-6)
    return numer / denom


def masked_ncc(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Per-sample masked normalized cross-correlation in [-1, 1].

    Args:
        x, y: ``(B, 1, H, W)`` single-channel tensors (typically luminance).
        weight: ``(B, 1, H, W)`` broadcastable mask >= 0.

    Returns:
        ``(B,)`` per-sample NCC in [-1, 1]. Higher is better alignment.
    """
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: x {x.shape} vs y {y.shape}")
    if x.shape[1] != 1:
        raise ValueError(f"expected single channel, got {x.shape[1]}")

    w = weight.flatten(1)
    xf = x.flatten(1)
    yf = y.flatten(1)
    w_sum = w.sum(dim=1).clamp(min=1e-6)

    x_mean = (w * xf).sum(dim=1) / w_sum
    y_mean = (w * yf).sum(dim=1) / w_sum

    xc = xf - x_mean.unsqueeze(1)
    yc = yf - y_mean.unsqueeze(1)

    cov = (w * xc * yc).sum(dim=1) / w_sum
    var_x = (w * xc * xc).sum(dim=1) / w_sum
    var_y = (w * yc * yc).sum(dim=1) / w_sum

    # eps inside the sqrt to avoid gradients -> infinity on degenerate samples.
    ncc = cov / (torch.sqrt(var_x * var_y + 1e-12) + 1e-6)
    return ncc.clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Top-level loss module
# ---------------------------------------------------------------------------


@dataclass
class RefinerLossWeights:
    """Per-component loss weights. See plan.md §1.6."""

    corner: float = 1.0   # Type A supervised corner regression
    recon: float = 0.25   # Type A RGB reconstruction
    ncc: float = 1.0      # Type B illumination-invariant alignment
    grad: float = 1.0     # Type B edge-structure alignment
    reg: float = 0.01     # Type B corner magnitude regularization


class RefinerLoss(nn.Module):
    """Full refiner training objective with per-sample Type A/B routing.

    Internally:
        1. Builds ``pred_H`` from ``pred_corners`` via ``corners_to_H``.
        2. Warps ``source`` and a validity mask through ``pred_H``.
        3. Combines validity with a cached center-weight map.
        4. Computes each component loss per-sample, then routes by type.

    Output is a dict with ``total`` (backprop-able) and the unweighted
    per-component scalar values for logging.
    """

    center_weight: torch.Tensor  # buffer

    def __init__(
        self,
        image_size: tuple[int, int] = (64, 128),
        weights: RefinerLossWeights | None = None,
        border_frac: float = 0.1,
        edge_value: float = 0.1,
        charbonnier_eps: float = 1e-3,
    ):
        super().__init__()
        self.image_size = image_size
        self.weights = weights if weights is not None else RefinerLossWeights()
        self.charbonnier_eps = charbonnier_eps

        cw = center_weight_map(
            image_size[0], image_size[1],
            border_frac=border_frac,
            edge_value=edge_value,
        )
        # Buffer so it moves with .to(device) and isn't trainable.
        self.register_buffer("center_weight", cw, persistent=False)

    def forward(
        self,
        source: torch.Tensor,          # (B, 3, H, W)
        target: torch.Tensor,          # (B, 3, H, W)
        pred_corners: torch.Tensor,    # (B, 4, 2)
        gt_corners: torch.Tensor,      # (B, 4, 2)
        has_gt_corners: torch.Tensor,  # (B,) bool
    ) -> dict[str, torch.Tensor]:
        if source.shape != target.shape:
            raise ValueError(
                f"source/target shape mismatch: {source.shape} vs {target.shape}"
            )
        if source.shape[-2:] != self.image_size:
            raise ValueError(
                f"expected spatial size {self.image_size}, got {source.shape[-2:]}"
            )

        B, _, H, W = source.shape
        device = source.device
        dtype = source.dtype

        # Build predicted homography from corner offsets.
        src_corners = canonical_corners(H, W, device=device, dtype=dtype)
        src_corners = src_corners.unsqueeze(0).expand(B, -1, -1).contiguous()
        pred_H = corners_to_H(src_corners, src_corners + pred_corners)

        # Warp source and its validity mask through pred_H.
        warped = warp_image(source, pred_H, (H, W))
        valid = warp_validity_mask(pred_H, (H, W), (H, W))  # (B, 1, H, W)
        weight = valid * self.center_weight

        # Luminance used by NCC and Sobel.
        warped_lum = luminance(warped)
        target_lum = luminance(target)

        # Per-sample type masks.
        is_syn = has_gt_corners.to(dtype=dtype)
        is_real = 1.0 - is_syn
        n_syn = is_syn.sum().clamp(min=1e-6)
        n_real = is_real.sum().clamp(min=1e-6)

        # --- Type A: corner regression (smooth L1 on offsets) ---
        corner_per_sample = F.smooth_l1_loss(
            pred_corners, gt_corners, reduction="none"
        ).mean(dim=(1, 2))  # (B,)
        corner_loss = (corner_per_sample * is_syn).sum() / n_syn

        # --- Type A: RGB reconstruction (masked Charbonnier) ---
        recon_per_sample = masked_charbonnier(
            warped, target, weight, self.charbonnier_eps
        )
        recon_loss = (recon_per_sample * is_syn).sum() / n_syn

        # --- Type B: luminance NCC ---
        ncc_per_sample = masked_ncc(warped_lum, target_lum, weight)
        ncc_loss = ((1.0 - ncc_per_sample) * is_real).sum() / n_real

        # --- Type B: Sobel-magnitude Charbonnier ---
        warped_sobel = sobel_magnitude(warped_lum)
        target_sobel = sobel_magnitude(target_lum)
        grad_per_sample = masked_charbonnier(
            warped_sobel, target_sobel, weight, self.charbonnier_eps
        )
        grad_loss = (grad_per_sample * is_real).sum() / n_real

        # --- Type B: corner magnitude regularization ---
        reg_per_sample = pred_corners.pow(2).mean(dim=(1, 2))  # (B,)
        reg_loss = (reg_per_sample * is_real).sum() / n_real

        w = self.weights
        total = (
            w.corner * corner_loss
            + w.recon * recon_loss
            + w.ncc * ncc_loss
            + w.grad * grad_loss
            + w.reg * reg_loss
        )

        return {
            "total": total,
            "corner": corner_loss.detach(),
            "recon": recon_loss.detach(),
            "ncc": ncc_loss.detach(),
            "grad": grad_loss.detach(),
            "reg": reg_loss.detach(),
        }
