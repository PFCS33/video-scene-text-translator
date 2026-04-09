"""Unit tests for ``src.models.refiner.warp``.

These tests pin the direction convention: every ``H`` is forward (source ->
destination), ``corners_to_H`` matches ``cv2.getPerspectiveTransform``, and
``warp_image`` matches ``cv2.warpPerspective`` up to interpolation noise.
Any future refactor that flips a sign will fail these tests loudly.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from src.models.refiner.warp import (
    canonical_corners,
    compose_H,
    corners_to_H,
    warp_image,
    warp_validity_mask,
)

DEVICE = torch.device("cpu")  # fast enough for unit tests; keeps CI GPU-free


# ---------------------------------------------------------------------------
# canonical_corners
# ---------------------------------------------------------------------------

def test_canonical_corners_values():
    c = canonical_corners(64, 128)
    expected = torch.tensor([[0, 0], [128, 0], [128, 64], [0, 64]], dtype=torch.float32)
    assert torch.allclose(c, expected)


def test_canonical_corners_dtype_device():
    c = canonical_corners(8, 16, device=DEVICE, dtype=torch.float64)
    assert c.device == DEVICE
    assert c.dtype == torch.float64


# ---------------------------------------------------------------------------
# corners_to_H
# ---------------------------------------------------------------------------

def test_corners_to_H_identity_is_identity():
    """Mapping canonical corners to themselves should produce the identity."""
    src = canonical_corners(64, 128).unsqueeze(0)
    H = corners_to_H(src, src)
    assert torch.allclose(H[0], torch.eye(3), atol=1e-5)


def test_corners_to_H_matches_cv2():
    """DLT output must agree with cv2.getPerspectiveTransform numerically."""
    rng = np.random.default_rng(0)
    src_np = np.array([[0, 0], [128, 0], [128, 64], [0, 64]], dtype=np.float32)
    # Try a handful of random small perturbations
    for _ in range(10):
        delta = rng.uniform(-8, 8, size=(4, 2)).astype(np.float32)
        dst_np = src_np + delta
        H_cv = cv2.getPerspectiveTransform(src_np, dst_np)
        H_ours = corners_to_H(
            torch.from_numpy(src_np).unsqueeze(0),
            torch.from_numpy(dst_np).unsqueeze(0),
        )[0].numpy()
        # Both are normalized so H[2,2] == 1; compare elementwise.
        assert np.allclose(H_ours, H_cv, atol=1e-4), (
            f"\nours=\n{H_ours}\ncv2=\n{H_cv}"
        )


def test_corners_to_H_batched():
    """Batched input returns one homography per sample, equal to per-sample."""
    src = canonical_corners(64, 128)
    # Make a batch of 3 distinct target quads
    dst_batch = torch.stack([
        src,                                       # identity
        src + torch.tensor([2.0, 0.0]),            # pure translation
        src + torch.tensor([[1, 2], [-1, 2], [-1, -2], [1, -2]], dtype=torch.float32),
    ])
    src_batch = src.unsqueeze(0).expand(3, -1, -1).contiguous()
    H = corners_to_H(src_batch, dst_batch)
    # Per-sample result
    for i in range(3):
        H_i = corners_to_H(src_batch[i:i + 1], dst_batch[i:i + 1])
        assert torch.allclose(H[i:i + 1], H_i, atol=1e-5)


def test_corners_to_H_differentiable():
    """Gradients must flow through DLT solve."""
    src = canonical_corners(64, 128).unsqueeze(0)
    dst = src + torch.tensor([[1.0, 0.5], [-0.5, 1.0], [0.2, -0.8], [-1.0, 0.3]]).unsqueeze(0)
    dst = dst.requires_grad_(True)
    H = corners_to_H(src, dst)
    loss = H.sum()
    loss.backward()
    assert dst.grad is not None
    assert torch.isfinite(dst.grad).all()
    assert dst.grad.abs().sum() > 0


def test_corners_to_H_shape_validation():
    with pytest.raises(ValueError, match="same shape"):
        corners_to_H(torch.zeros(1, 4, 2), torch.zeros(1, 3, 2))
    with pytest.raises(ValueError, match=r"\(B, 4, 2\)"):
        corners_to_H(torch.zeros(4, 2), torch.zeros(4, 2))


# ---------------------------------------------------------------------------
# compose_H
# ---------------------------------------------------------------------------

def test_compose_H_identity_left():
    H = torch.tensor([[[1.0, 0.1, 3], [0.05, 1.0, -2], [0, 0, 1]]])
    eye = torch.eye(3).unsqueeze(0)
    assert torch.allclose(compose_H(eye, H), H, atol=1e-6)


def test_compose_H_identity_right():
    H = torch.tensor([[[1.0, 0.1, 3], [0.05, 1.0, -2], [0, 0, 1]]])
    eye = torch.eye(3).unsqueeze(0)
    assert torch.allclose(compose_H(H, eye), H, atol=1e-6)


def test_compose_H_normalizes_h22():
    # Build a homography scaled by 5 - compose_H should normalize away the scale.
    H = 5.0 * torch.eye(3).unsqueeze(0)
    composed = compose_H(H, torch.eye(3).unsqueeze(0))
    assert pytest.approx(composed[0, 2, 2].item(), abs=1e-6) == 1.0


def test_compose_H_matches_inverse_roundtrip():
    """compose_H(H, H^{-1}) should be the identity."""
    rng = np.random.default_rng(1)
    src = canonical_corners(64, 128).unsqueeze(0)
    dst = src + torch.from_numpy(rng.uniform(-6, 6, (4, 2)).astype(np.float32)).unsqueeze(0)
    H = corners_to_H(src, dst)
    H_inv = torch.linalg.inv(H)
    composed = compose_H(H, H_inv)
    assert torch.allclose(composed[0], torch.eye(3), atol=1e-4)


# ---------------------------------------------------------------------------
# warp_image
# ---------------------------------------------------------------------------

def _checker(h: int, w: int) -> torch.Tensor:
    """Simple test image: checker of 4 quadrants, distinct intensities."""
    img = torch.zeros(1, 1, h, w)
    img[:, :, : h // 2, : w // 2] = 0.25
    img[:, :, : h // 2, w // 2:] = 0.5
    img[:, :, h // 2:, : w // 2] = 0.75
    img[:, :, h // 2:, w // 2:] = 1.0
    return img


def test_warp_image_identity_is_identity():
    img = _checker(64, 128)
    H = torch.eye(3).unsqueeze(0)
    out = warp_image(img, H, (64, 128))
    # Bilinear sampling on an integer-coord identity should reproduce exactly.
    assert torch.allclose(out, img, atol=1e-5)


def test_warp_image_roundtrip_recovers_original():
    """warp(warp(I, H), H^{-1}) matches I inside the valid region."""
    img = _checker(64, 128)
    src = canonical_corners(64, 128).unsqueeze(0)
    dst = src + torch.tensor([[2.0, 1.0], [-1.0, 2.0], [-2.0, -1.0], [1.0, -2.0]]).unsqueeze(0)
    H = corners_to_H(src, dst)
    H_inv = torch.linalg.inv(H)

    warped = warp_image(img, H, (64, 128))
    recovered = warp_image(warped, H_inv, (64, 128))

    # Validity mask of the round-trip (ignore boundary regions where content
    # leaked out of bounds).
    valid = warp_validity_mask(H_inv, (64, 128), (64, 128)) > 0.95

    # Where valid, recovered matches original within a loose tolerance
    # (two bilinear samples accumulate some blur).
    diff = (recovered - img).abs()
    masked_mean = (diff * valid).sum() / (valid.sum() + 1e-6)
    assert masked_mean.item() < 0.02


def test_warp_image_matches_cv2_warpPerspective():
    """Forward direction sanity: output content agrees with cv2.warpPerspective."""
    rng = np.random.default_rng(2)
    H_img, W_img = 64, 128
    # Generate a deterministic noisy image so interpolation has real signal.
    img_np = (rng.uniform(0, 1, (H_img, W_img)).astype(np.float32))
    img_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    src_np = np.array([[0, 0], [W_img, 0], [W_img, H_img], [0, H_img]], dtype=np.float32)
    dst_np = src_np + rng.uniform(-5, 5, (4, 2)).astype(np.float32)
    H_np = cv2.getPerspectiveTransform(src_np, dst_np)

    cv2_out = cv2.warpPerspective(img_np, H_np, (W_img, H_img), flags=cv2.INTER_LINEAR)

    H_t = torch.from_numpy(H_np).unsqueeze(0).to(torch.float32)
    ours_out = warp_image(img_t, H_t, (H_img, W_img))[0, 0].numpy()

    # Both methods are bilinear but differ in sub-pixel grid conventions at
    # ~0.5 px magnitude. Compare on the interior where both agree.
    interior = (5, H_img - 5, 5, W_img - 5)
    r0, r1, c0, c1 = interior
    diff = np.abs(ours_out[r0:r1, c0:c1] - cv2_out[r0:r1, c0:c1])
    assert diff.mean() < 0.02, f"mean diff {diff.mean()}"


def test_warp_image_translation_moves_content():
    """Concrete direction check: positive x translation shifts content right."""
    img = _checker(64, 128)
    src = canonical_corners(64, 128).unsqueeze(0)
    dst = src + torch.tensor([10.0, 0.0])  # move all corners +10 in x
    H = corners_to_H(src, dst)
    out = warp_image(img, H, (64, 128))

    # In the output, the content that was at x=20 in the source should now
    # appear at x=30. Compare column slices well inside each quadrant.
    col_src = img[0, 0, 32, 20].item()      # interior of a quadrant in src
    col_dst = out[0, 0, 32, 30].item()
    assert pytest.approx(col_dst, abs=1e-3) == col_src


def test_warp_image_batched_independent():
    """Each batch item is warped by its own H."""
    img = _checker(64, 128).repeat(2, 1, 1, 1)
    eye = torch.eye(3)
    # Item 0: identity. Item 1: +10 px x translation.
    src = canonical_corners(64, 128).unsqueeze(0)
    dst = src + torch.tensor([10.0, 0.0])
    H_translate = corners_to_H(src, dst)[0]
    H_batch = torch.stack([eye, H_translate])  # (2, 3, 3)

    out = warp_image(img, H_batch, (64, 128))
    # Item 0 unchanged
    assert torch.allclose(out[0], img[0], atol=1e-5)
    # Item 1 shifted
    col_src = img[1, 0, 32, 20].item()
    col_dst = out[1, 0, 32, 30].item()
    assert pytest.approx(col_dst, abs=1e-3) == col_src


def test_warp_image_shape_validation():
    with pytest.raises(ValueError, match=r"\(B, C, H, W\)"):
        warp_image(torch.zeros(1, 64, 128), torch.eye(3).unsqueeze(0), (64, 128))
    with pytest.raises(ValueError, match=r"\(B, 3, 3\)"):
        warp_image(torch.zeros(1, 1, 64, 128), torch.eye(3), (64, 128))
    with pytest.raises(ValueError, match="batch size mismatch"):
        warp_image(
            torch.zeros(2, 1, 64, 128),
            torch.eye(3).unsqueeze(0),  # batch 1
            (64, 128),
        )


# ---------------------------------------------------------------------------
# warp_validity_mask
# ---------------------------------------------------------------------------

def test_warp_validity_mask_identity_is_all_ones():
    H = torch.eye(3).unsqueeze(0)
    mask = warp_validity_mask(H, (64, 128), (64, 128))
    # Bilinear-sampled ones should be ~1 everywhere except a ~1 px border.
    interior = mask[0, 0, 2:-2, 2:-2]
    assert (interior > 0.99).all()


def test_warp_validity_mask_translation_zeros_at_uncovered_edge():
    """Translating by +10 px in x leaves the left 10 columns uncovered."""
    src = canonical_corners(64, 128).unsqueeze(0)
    dst = src + torch.tensor([10.0, 0.0])
    H = corners_to_H(src, dst)
    mask = warp_validity_mask(H, (64, 128), (64, 128))
    # Far-left columns of the output sample from x < 0 in the source: invalid.
    assert mask[0, 0, :, 0:8].max() < 0.05
    # Far-right columns should be fully valid.
    assert mask[0, 0, :, 20:-2].min() > 0.95
