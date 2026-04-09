"""Differentiable homography warp utilities.

All homography direction conventions live in exactly one place so downstream
code (model head, losses, inference wrapper) cannot get them wrong.

Direction convention
--------------------
Every ``H`` produced or consumed here is a **forward** homography that maps
*source* image-space pixel coordinates to *destination* image-space pixel
coordinates. Equivalently, given a source image ``S``, ``warp_image(S, H)``
produces an image in which the source's canonical corners
``[[0,0], [W,0], [W,H], [0,H]]`` have been moved to ``H @ canonical_corners``.

This matches ``cv2.getPerspectiveTransform(src_pts, dst_pts)`` exactly.

Image coordinate convention
---------------------------
We use pixel-edge coordinates: the image rectangle of an ``(H_img, W_img)``
tensor spans ``[0, W_img] x [0, H_img]`` and the top-left pixel's center is at
``(0.5, 0.5)``. This matches ``grid_sample(..., align_corners=False)``.

Canonical corners are therefore ``[[0,0], [W,0], [W,H], [0,H]]`` as stated in
the refiner plan, not the ``W-1``/``H-1`` form you would use with
``align_corners=True``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def canonical_corners(
    height: int,
    width: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return the canonical image-rectangle corners.

    Order: top-left, top-right, bottom-right, bottom-left — matches the
    ``Quad`` convention used elsewhere in the pipeline.

    Args:
        height: image height in pixels.
        width: image width in pixels.
        device: optional torch device.
        dtype: optional torch dtype (defaults to float32).

    Returns:
        (4, 2) tensor ``[[0,0], [W,0], [W,H], [0,H]]``.
    """
    return torch.tensor(
        [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]],
        device=device,
        dtype=dtype if dtype is not None else torch.float32,
    )


def corners_to_H(src_corners: torch.Tensor, dst_corners: torch.Tensor) -> torch.Tensor:
    """Solve the 4-point DLT for a batch of corner correspondences.

    Given 4 source points and 4 destination points per batch item, returns the
    3x3 homography ``H`` such that (for each correspondence ``i``):

        (dst_i.x, dst_i.y, 1) ~ H @ (src_i.x, src_i.y, 1)

    where ``~`` is equality up to scale. The returned ``H`` is normalized so
    ``H[2, 2] == 1``.

    The 8-unknown linear system matches the derivation used by
    ``cv2.getPerspectiveTransform``.

    Args:
        src_corners: (B, 4, 2) source point coordinates.
        dst_corners: (B, 4, 2) destination point coordinates.

    Returns:
        (B, 3, 3) forward homographies. Fully differentiable.
    """
    if src_corners.shape != dst_corners.shape:
        raise ValueError(
            f"src_corners and dst_corners must have same shape, got "
            f"{src_corners.shape} vs {dst_corners.shape}"
        )
    if src_corners.dim() != 3 or src_corners.shape[-2:] != (4, 2):
        raise ValueError(f"expected (B, 4, 2), got {src_corners.shape}")

    B = src_corners.shape[0]
    dtype = src_corners.dtype
    device = src_corners.device

    x = src_corners[..., 0]  # (B, 4)
    y = src_corners[..., 1]
    xp = dst_corners[..., 0]
    yp = dst_corners[..., 1]

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    # Each correspondence contributes two rows of the 8x8 system A @ h = b.
    # Row 1: [x, y, 1, 0, 0, 0, -x*xp, -y*xp] -> xp
    # Row 2: [0, 0, 0, x, y, 1, -x*yp, -y*yp] -> yp
    row1 = torch.stack([x, y, ones, zeros, zeros, zeros, -x * xp, -y * xp], dim=-1)  # (B, 4, 8)
    row2 = torch.stack([zeros, zeros, zeros, x, y, ones, -x * yp, -y * yp], dim=-1)  # (B, 4, 8)
    A = torch.stack([row1, row2], dim=2).reshape(B, 8, 8)
    b = torch.stack([xp, yp], dim=-1).reshape(B, 8)

    h = torch.linalg.solve(A, b)  # (B, 8)
    H = torch.cat([h, torch.ones(B, 1, device=device, dtype=dtype)], dim=-1)
    return H.reshape(B, 3, 3)


def compose_H(H_left: torch.Tensor, H_right: torch.Tensor) -> torch.Tensor:
    """Compose two homographies: ``result = H_left @ H_right``.

    Result is normalized so ``result[..., 2, 2] == 1`` for numerical stability
    across long composition chains.
    """
    H = H_left @ H_right
    denom = H[..., 2:3, 2:3]
    return H / (denom + torch.sign(denom) * 1e-12 + 1e-12)


def warp_image(
    image: torch.Tensor,
    H: torch.Tensor,
    out_shape: tuple[int, int],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """Apply a forward homography ``H`` to an image.

    For each output pixel ``(i, j)`` at image-space center ``(i+0.5, j+0.5)``,
    samples the source image at ``H^{-1} @ (i+0.5, j+0.5, 1)``. Equivalent to
    ``cv2.warpPerspective(image, H, (W_out, H_out))`` up to sampling kernel
    differences.

    Args:
        image: (B, C, H_src, W_src) source tensor.
        H: (B, 3, 3) forward homography (source -> destination).
        out_shape: ``(H_out, W_out)``.
        mode: interpolation mode passed to ``grid_sample``.
        padding_mode: padding mode passed to ``grid_sample``.

    Returns:
        (B, C, H_out, W_out) warped tensor.
    """
    if image.dim() != 4:
        raise ValueError(f"expected (B, C, H, W), got {image.shape}")
    if H.dim() != 3 or H.shape[-2:] != (3, 3):
        raise ValueError(f"expected (B, 3, 3), got {H.shape}")
    if image.shape[0] != H.shape[0]:
        raise ValueError(
            f"batch size mismatch: image {image.shape[0]} vs H {H.shape[0]}"
        )

    B, _, H_src, W_src = image.shape
    H_out, W_out = out_shape
    device = image.device
    dtype = image.dtype

    # Build destination pixel-center coordinate grid in image-space.
    ys, xs = torch.meshgrid(
        torch.arange(H_out, device=device, dtype=dtype) + 0.5,
        torch.arange(W_out, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    # (H_out * W_out, 3) homogeneous destination coords
    dst_h = torch.stack([xs, ys, ones], dim=-1).reshape(-1, 3)

    # Map destination coords back to source coords via H^{-1}.
    H_inv = torch.linalg.inv(H.to(torch.float64)).to(dtype)  # (B, 3, 3)
    # (B, 3, 3) @ (3, N) -> (B, 3, N)
    src_h = H_inv @ dst_h.T  # (B, 3, H_out*W_out)
    w = src_h[:, 2:3, :]
    # Avoid division by zero without breaking autograd on degenerate inputs
    w_safe = w + torch.where(w.abs() < 1e-12, torch.full_like(w, 1e-12), torch.zeros_like(w))
    src_x = src_h[:, 0:1, :] / w_safe  # (B, 1, N)
    src_y = src_h[:, 1:2, :] / w_safe

    # Normalize to grid_sample's [-1, 1] range under align_corners=False:
    #     grid_x = 2 * src_x / W_src - 1
    #     grid_y = 2 * src_y / H_src - 1
    grid_x = 2.0 * src_x / W_src - 1.0
    grid_y = 2.0 * src_y / H_src - 1.0
    grid = torch.stack([grid_x.squeeze(1), grid_y.squeeze(1)], dim=-1)  # (B, N, 2)
    grid = grid.reshape(B, H_out, W_out, 2)

    return F.grid_sample(
        image, grid, mode=mode, padding_mode=padding_mode, align_corners=False
    )


def warp_validity_mask(
    H: torch.Tensor,
    in_shape: tuple[int, int],
    out_shape: tuple[int, int],
) -> torch.Tensor:
    """Compute a soft validity mask for ``warp_image``.

    Warps an all-ones tensor by ``H``; the result is ~1 where the destination
    pixel sampled inside the source image and drops smoothly to 0 at the
    boundary (bilinear falloff across ~1 pixel). Use this as a loss weighting
    map to exclude invalid warped-source pixels.

    Args:
        H: (B, 3, 3) forward homography used for the warp.
        in_shape: ``(H_src, W_src)`` of the (virtual) source tensor.
        out_shape: ``(H_out, W_out)`` of the warped result.

    Returns:
        (B, 1, H_out, W_out) soft mask in [0, 1].
    """
    B = H.shape[0]
    device = H.device
    dtype = H.dtype
    ones = torch.ones(B, 1, in_shape[0], in_shape[1], device=device, dtype=dtype)
    return warp_image(ones, H, out_shape, mode="bilinear", padding_mode="zeros")
