"""Differentiable oriented 2D Gaussian blur for BPN.

Implements the blur model from STRIVE Section 3.2:
    I_out = (1 + w) * I_ref - w * (I_ref * G_{sigma, rho})

where G is an oriented 2D Gaussian parameterized by (sigma_x, sigma_y, rho)
and w controls blur/sharpen intensity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DifferentiableBlur(nn.Module):
    """Apply oriented Gaussian blur/sharpen given predicted parameters."""

    def __init__(self, kernel_size: int = 21):
        super().__init__()
        self.kernel_size = kernel_size
        # Pre-compute coordinate grids (centered at 0)
        half = kernel_size // 2
        ax = torch.arange(-half, half + 1, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ax, ax, indexing="ij")
        self.register_buffer("grid_x", grid_x)
        self.register_buffer("grid_y", grid_y)

    def build_kernel(self, sigma_x: torch.Tensor, sigma_y: torch.Tensor,
                     rho: torch.Tensor) -> torch.Tensor:
        """Build oriented 2D Gaussian kernel from parameters.

        Args:
            sigma_x: (B,) blur spread along rotated x-axis
            sigma_y: (B,) blur spread along rotated y-axis
            rho: (B,) rotation angle in radians

        Returns:
            kernel: (B, 1, K, K) normalized Gaussian kernel
        """
        B = sigma_x.shape[0]
        # Clamp sigmas to avoid division by zero or degenerate kernels
        sigma_x = sigma_x.clamp(min=0.3)
        sigma_y = sigma_y.clamp(min=0.3)

        cos_r = torch.cos(rho)  # (B,)
        sin_r = torch.sin(rho)

        # Rotate coordinates: x' = cos*x + sin*y, y' = -sin*x + cos*y
        gx = self.grid_x.unsqueeze(0)  # (1, K, K)
        gy = self.grid_y.unsqueeze(0)

        # (B, K, K)
        x_rot = cos_r[:, None, None] * gx + sin_r[:, None, None] * gy
        y_rot = -sin_r[:, None, None] * gx + cos_r[:, None, None] * gy

        # Gaussian: exp(-(x'^2/sx^2 + y'^2/sy^2) / 2)
        sx = sigma_x[:, None, None]
        sy = sigma_y[:, None, None]
        exponent = -0.5 * (x_rot ** 2 / sx ** 2 + y_rot ** 2 / sy ** 2)
        kernel = torch.exp(exponent)

        # Normalize
        kernel = kernel / kernel.sum(dim=(-2, -1), keepdim=True)
        return kernel.unsqueeze(1)  # (B, 1, K, K)

    def forward(self, image: torch.Tensor, sigma_x: torch.Tensor,
                sigma_y: torch.Tensor, rho: torch.Tensor,
                w: torch.Tensor) -> torch.Tensor:
        """Apply differential blur/sharpen to image.

        Args:
            image: (B, C, H, W) input image (reference ROI)
            sigma_x, sigma_y, rho, w: (B,) blur parameters

        Returns:
            output: (B, C, H, W) blurred/sharpened image
        """
        B, C, H, W = image.shape
        kernel = self.build_kernel(sigma_x, sigma_y, rho)  # (B, 1, K, K)

        # Apply blur per-sample using grouped convolution
        # Reshape to (1, B*C, H, W) so each channel gets its own group
        pad = self.kernel_size // 2
        img_flat = image.reshape(1, B * C, H, W)
        # Kernel: (B*C, 1, K, K) — one kernel per group
        kern_flat = kernel.repeat(1, C, 1, 1).reshape(B * C, 1,
                                                        self.kernel_size,
                                                        self.kernel_size)
        blurred_flat = F.conv2d(img_flat, kern_flat, padding=pad,
                                groups=B * C)
        blurred = blurred_flat.reshape(B, C, H, W)

        # Differential blur model: I_out = (1+w)*I - w*(I*G)
        w_expanded = w[:, None, None, None]
        output = (1.0 + w_expanded) * image - w_expanded * blurred
        return output.clamp(0, 1)
