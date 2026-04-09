"""ROIRefiner network.

First-version architecture: HomographyNet-style concat-and-CNN + FC head.

    Input:  two ``(B, 3, H, W)`` RGB ROIs concatenated along the channel dim
            => ``(B, 6, H, W)``.
    Output: ``(B, 4, 2)`` corner offsets in **network pixel coordinates**
            (i.e. at the model's input resolution, not the real ROI size).

The final ``Linear`` is initialized with small weights so the initial output
is near zero and the initial predicted homography is near identity — same
trick BPN uses to keep its tanh/softplus head in the linear regime at init
(see CHANGELOG.md 2026-04-08). This is important because both the
reconstruction loss and the residual regularizer are best-conditioned when
training starts from "no-op" rather than from a random large warp.

See plan.md §1.4 for the design rationale and the upgrade path (correlation
volume, multi-scale, etc.) if this minimum version plateaus.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    """Conv2d -> BatchNorm2d -> ReLU block with ``bias=False`` on the conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ROIRefiner(nn.Module):
    """4-corner offset regressor for residual ROI alignment.

    Parameters
    ----------
    in_channels:
        Input channels after source/target concat. Default 6 (two RGB inputs).
    base_channels:
        First conv's output channel count. Later blocks widen to
        ``2x``, ``3x``, ``4x`` base.
    dropout:
        Dropout probability applied between the two FC layers.
    image_size:
        ``(H, W)`` of network inputs. Used to size the FC1 layer; the backbone
        downsamples by a factor of 16, so both dims must be divisible by 16.
    head_init_scale:
        Standard deviation of the final Linear's weight init. Small values
        (default ``1e-3``) make the initial output ~0 so the initial
        homography is ~identity.
    """

    def __init__(
        self,
        in_channels: int = 6,
        base_channels: int = 32,
        dropout: float = 0.2,
        image_size: tuple[int, int] = (64, 128),
        head_init_scale: float = 1e-3,
    ):
        super().__init__()
        self.image_size = image_size

        H, W = image_size
        if H % 16 != 0 or W % 16 != 0:
            raise ValueError(
                f"image_size {image_size} must be divisible by 16 "
                f"(backbone downsamples by 4 stride-2 blocks)"
            )

        c = base_channels
        # 4 stride-2 blocks: (H, W) -> (H/16, W/16). For (64, 128): (4, 8).
        self.backbone = nn.Sequential(
            ConvBNReLU(in_channels, c,      stride=2),   # /2
            ConvBNReLU(c,            c * 2, stride=2),   # /4
            ConvBNReLU(c * 2,        c * 3, stride=2),   # /8
            ConvBNReLU(c * 3,        c * 4, stride=2),   # /16
        )

        feat_h, feat_w = H // 16, W // 16
        flat_dim = c * 4 * feat_h * feat_w

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 8),
        )

        # Initial output near zero -> initial ΔH ≈ I.
        final_linear = self.head[-1]
        assert isinstance(final_linear, nn.Linear)
        nn.init.normal_(final_linear.weight, std=head_init_scale)
        nn.init.zeros_(final_linear.bias)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Predict corner offsets aligning ``source`` to ``target``.

        Args:
            source: (B, 3, H, W) pre-aligned source ROI in [0, 1].
            target: (B, 3, H, W) target ROI in [0, 1].

        Returns:
            (B, 4, 2) corner offsets in network pixel coordinates.
            The offsets describe where the canonical corners of ``source``
            should move to produce an image aligned with ``target``; pass
            them to ``corners_to_H`` to build a forward homography.
        """
        if source.shape != target.shape:
            raise ValueError(
                f"source and target must have same shape, got "
                f"{source.shape} vs {target.shape}"
            )
        if source.dim() != 4 or source.shape[1] != 3:
            raise ValueError(f"expected (B, 3, H, W), got {source.shape}")
        expected_hw = self.image_size
        if source.shape[-2:] != expected_hw:
            raise ValueError(
                f"expected spatial size {expected_hw}, got {source.shape[-2:]}"
            )

        x = torch.cat([source, target], dim=1)
        feat = self.backbone(x)
        delta = self.head(feat)
        return delta.view(-1, 4, 2)

    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
