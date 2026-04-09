"""Unit tests for ``src.models.refiner.losses``.

Two layers of tests:
    1. Primitives (luminance, sobel_magnitude, center_weight_map,
       masked_charbonnier, masked_ncc) as pure functions.
    2. ``RefinerLoss`` top-level module: shape contract, per-sample Type A
       vs Type B routing, full-syn / full-real batch collapse, and gradient
       flow through the whole warp + loss chain.
"""

from __future__ import annotations

import pytest
import torch

from src.models.refiner.losses import (
    RefinerLoss,
    RefinerLossWeights,
    center_weight_map,
    luminance,
    masked_charbonnier,
    masked_ncc,
    sobel_magnitude,
)

# ---------------------------------------------------------------------------
# luminance
# ---------------------------------------------------------------------------

def test_luminance_shape_and_range():
    rgb = torch.rand(2, 3, 16, 32)
    lum = luminance(rgb)
    assert lum.shape == (2, 1, 16, 32)
    assert (lum >= 0).all() and (lum <= 1).all()


def test_luminance_weights_sum_to_one():
    """A uniform gray image should map to the same gray luminance value."""
    rgb = torch.full((1, 3, 4, 4), 0.5)
    lum = luminance(rgb)
    assert torch.allclose(lum, torch.full_like(lum, 0.5), atol=1e-6)


def test_luminance_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"\(B, 3, H, W\)"):
        luminance(torch.rand(2, 1, 16, 32))


# ---------------------------------------------------------------------------
# sobel_magnitude
# ---------------------------------------------------------------------------

def test_sobel_magnitude_uniform_is_zero():
    """Uniform image has no gradients -> magnitude near zero everywhere."""
    img = torch.full((1, 1, 16, 32), 0.5)
    mag = sobel_magnitude(img)
    # eps=1e-12 inside the sqrt gives a floor of ~1e-6
    assert mag.max().item() < 1e-3


def test_sobel_magnitude_vertical_edge_lights_up():
    """A vertical step edge produces large gradient magnitude at the boundary."""
    img = torch.zeros(1, 1, 16, 32)
    img[:, :, :, 16:] = 1.0
    mag = sobel_magnitude(img)
    # The edge at column 16 should have strong response
    edge_response = mag[0, 0, :, 15:17].max().item()
    interior_response = mag[0, 0, :, 0:10].max().item()
    assert edge_response > 0.5
    assert interior_response < 1e-3


def test_sobel_magnitude_shape_preserved():
    img = torch.rand(3, 2, 16, 32)
    mag = sobel_magnitude(img)
    assert mag.shape == img.shape


# ---------------------------------------------------------------------------
# center_weight_map
# ---------------------------------------------------------------------------

def test_center_weight_map_shape_and_range():
    w = center_weight_map(64, 128, edge_value=0.1)
    assert w.shape == (1, 1, 64, 128)
    # Minimum is edge_value^2 at the corners (product of two axis falloffs).
    assert (w >= 0.01 - 1e-6).all()
    assert (w <= 1.0 + 1e-6).all()


def test_center_weight_map_center_is_one_corner_is_edge_value_sq():
    w = center_weight_map(64, 128, border_frac=0.1, edge_value=0.1)
    assert pytest.approx(w[0, 0, 32, 64].item(), abs=1e-4) == 1.0
    # Corner is edge_value on each axis -> product = edge_value^2 = 0.01
    assert pytest.approx(w[0, 0, 0, 0].item(), abs=1e-4) == 0.01


def test_center_weight_map_validates_parameters():
    with pytest.raises(ValueError, match="border_frac"):
        center_weight_map(64, 128, border_frac=0.0)
    with pytest.raises(ValueError, match="border_frac"):
        center_weight_map(64, 128, border_frac=0.6)
    with pytest.raises(ValueError, match="edge_value"):
        center_weight_map(64, 128, edge_value=-0.1)


# ---------------------------------------------------------------------------
# masked_charbonnier
# ---------------------------------------------------------------------------

def test_masked_charbonnier_identical_inputs_give_eps():
    """Charbonnier of (x, x) with eps=1e-3 is ~eps."""
    x = torch.rand(2, 3, 16, 32)
    w = torch.ones(2, 1, 16, 32)
    loss = masked_charbonnier(x, x, w, eps=1e-3)
    assert loss.shape == (2,)
    assert (loss - 1e-3).abs().max().item() < 1e-6


def test_masked_charbonnier_zero_weight_is_zero():
    """With a zero mask the denominator clamps; result is 0/clamp = 0."""
    x = torch.rand(1, 3, 16, 32)
    y = torch.rand(1, 3, 16, 32)
    w = torch.zeros(1, 1, 16, 32)
    loss = masked_charbonnier(x, y, w)
    assert loss.shape == (1,)
    # numerator is also zero so the ratio is zero.
    assert loss.abs().max().item() < 1e-6


def test_masked_charbonnier_matches_unmasked_when_weight_is_ones():
    """With weight=1 everywhere, the result equals the unweighted mean."""
    torch.manual_seed(0)
    x = torch.rand(2, 3, 8, 8)
    y = torch.rand(2, 3, 8, 8)
    w = torch.ones(2, 1, 8, 8)
    loss = masked_charbonnier(x, y, w, eps=1e-3)
    # Reference: same formula but unmasked
    diff = torch.sqrt((x - y).pow(2) + 1e-6).mean(dim=1, keepdim=True)
    ref = diff.flatten(1).mean(dim=1)
    assert torch.allclose(loss, ref, atol=1e-6)


def test_masked_charbonnier_shape_validation():
    with pytest.raises(ValueError, match="shape mismatch"):
        masked_charbonnier(
            torch.rand(1, 3, 16, 32),
            torch.rand(1, 3, 16, 16),
            torch.rand(1, 1, 16, 32),
        )


# ---------------------------------------------------------------------------
# masked_ncc
# ---------------------------------------------------------------------------

def test_masked_ncc_identical_is_one():
    torch.manual_seed(0)
    x = torch.rand(2, 1, 16, 32)
    w = torch.ones(2, 1, 16, 32)
    ncc = masked_ncc(x, x, w)
    assert ncc.shape == (2,)
    assert torch.allclose(ncc, torch.ones(2), atol=1e-4)


def test_masked_ncc_invariant_to_brightness():
    """NCC should be ~1 for (x, a*x + b) with a > 0 (scale/shift invariance)."""
    torch.manual_seed(0)
    x = torch.rand(1, 1, 16, 32)
    y = 0.3 * x + 0.4
    w = torch.ones(1, 1, 16, 32)
    ncc = masked_ncc(x, y, w)
    assert ncc[0].item() > 0.999


def test_masked_ncc_anti_correlated_is_minus_one():
    torch.manual_seed(0)
    x = torch.rand(1, 1, 16, 32)
    y = -x
    w = torch.ones(1, 1, 16, 32)
    ncc = masked_ncc(x, y, w)
    assert ncc[0].item() < -0.999


def test_masked_ncc_shape_validation():
    with pytest.raises(ValueError, match="single channel"):
        masked_ncc(
            torch.rand(1, 3, 16, 32),
            torch.rand(1, 3, 16, 32),
            torch.rand(1, 1, 16, 32),
        )


# ---------------------------------------------------------------------------
# RefinerLoss
# ---------------------------------------------------------------------------

def _make_batch(B: int = 4, H: int = 64, W: int = 128):
    torch.manual_seed(0)
    source = torch.rand(B, 3, H, W)
    target = torch.rand(B, 3, H, W)
    pred_corners = torch.randn(B, 4, 2) * 0.5
    gt_corners = torch.randn(B, 4, 2) * 0.5
    return source, target, pred_corners, gt_corners


def test_refiner_loss_return_keys_and_shapes():
    loss_fn = RefinerLoss()
    source, target, pred, gt = _make_batch()
    has_gt = torch.tensor([True, True, False, False])
    out = loss_fn(source, target, pred, gt, has_gt)
    assert set(out.keys()) == {"total", "corner", "recon", "ncc", "grad", "reg"}
    for k, v in out.items():
        assert v.dim() == 0, f"{k} should be scalar, got {tuple(v.shape)}"
        assert torch.isfinite(v), f"{k} not finite: {v}"


def test_refiner_loss_full_synthetic_batch_zeros_type_b_components():
    loss_fn = RefinerLoss()
    source, target, pred, gt = _make_batch()
    has_gt = torch.tensor([True, True, True, True])
    out = loss_fn(source, target, pred, gt, has_gt)
    # Type B components should be 0 when no real samples are present
    assert out["ncc"].item() == 0.0
    assert out["grad"].item() == 0.0
    assert out["reg"].item() == 0.0
    # Type A components should be non-zero
    assert out["corner"].item() > 0.0
    assert out["recon"].item() > 0.0


def test_refiner_loss_full_real_batch_zeros_type_a_components():
    loss_fn = RefinerLoss()
    source, target, pred, gt = _make_batch()
    has_gt = torch.tensor([False, False, False, False])
    out = loss_fn(source, target, pred, gt, has_gt)
    assert out["corner"].item() == 0.0
    assert out["recon"].item() == 0.0
    # Type B components should be non-zero
    assert out["ncc"].item() > 0.0
    assert out["grad"].item() > 0.0
    assert out["reg"].item() > 0.0


def test_refiner_loss_mixed_batch_blends_components():
    loss_fn = RefinerLoss()
    source, target, pred, gt = _make_batch()
    has_gt = torch.tensor([True, True, False, False])
    out = loss_fn(source, target, pred, gt, has_gt)
    # All five components should be nonzero in a mixed batch
    for name in ("corner", "recon", "ncc", "grad", "reg"):
        assert out[name].item() > 0.0, f"{name} should be > 0 in mixed batch"


def test_refiner_loss_corner_matches_hand_computed():
    """With zero pred and known GT, corner loss = smooth_l1(0, gt).mean()."""
    loss_fn = RefinerLoss(
        weights=RefinerLossWeights(corner=1.0, recon=0.0, ncc=0.0, grad=0.0, reg=0.0),
    )
    B = 2
    source = torch.rand(B, 3, 64, 128)
    target = torch.rand(B, 3, 64, 128)
    pred = torch.zeros(B, 4, 2)
    gt = torch.tensor([
        [[1.0, 0.5], [0.0, 1.0], [-1.0, -0.5], [0.5, 0.0]],
        [[0.2, 0.2], [0.3, 0.3], [0.1, 0.1], [0.0, 0.0]],
    ])
    has_gt = torch.tensor([True, True])
    out = loss_fn(source, target, pred, gt, has_gt)
    # Hand reference
    expected = torch.nn.functional.smooth_l1_loss(
        pred, gt, reduction="none"
    ).mean(dim=(1, 2)).mean()
    assert torch.allclose(out["corner"], expected, atol=1e-6)


def test_refiner_loss_reg_matches_pred_norm():
    loss_fn = RefinerLoss(
        weights=RefinerLossWeights(corner=0.0, recon=0.0, ncc=0.0, grad=0.0, reg=1.0),
    )
    B = 2
    source = torch.rand(B, 3, 64, 128)
    target = torch.rand(B, 3, 64, 128)
    pred = torch.tensor([
        [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 2.0]],
    ])
    has_gt = torch.tensor([False, False])
    out = loss_fn(source, target, pred, torch.zeros_like(pred), has_gt)
    # Per sample: (1/8) and (4/8). Mean: 0.3125
    assert pytest.approx(out["reg"].item(), abs=1e-6) == 0.3125


def test_refiner_loss_gradients_flow_to_pred_corners():
    loss_fn = RefinerLoss()
    source, target, pred, gt = _make_batch()
    pred.requires_grad_(True)
    has_gt = torch.tensor([True, True, False, False])
    out = loss_fn(source, target, pred, gt, has_gt)
    out["total"].backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0


def test_refiner_loss_weights_apply_linearly():
    """Halving all weights halves the total loss."""
    source, target, pred, gt = _make_batch()
    has_gt = torch.tensor([True, True, False, False])

    full = RefinerLoss(
        weights=RefinerLossWeights(corner=1.0, recon=1.0, ncc=1.0, grad=1.0, reg=1.0),
    )
    half = RefinerLoss(
        weights=RefinerLossWeights(corner=0.5, recon=0.5, ncc=0.5, grad=0.5, reg=0.5),
    )
    out_full = full(source, target, pred, gt, has_gt)
    out_half = half(source, target, pred, gt, has_gt)
    assert torch.allclose(out_half["total"], 0.5 * out_full["total"], atol=1e-6)


def test_refiner_loss_shape_validation():
    loss_fn = RefinerLoss()
    with pytest.raises(ValueError, match="shape mismatch"):
        loss_fn(
            torch.rand(1, 3, 64, 128),
            torch.rand(1, 3, 32, 64),
            torch.zeros(1, 4, 2),
            torch.zeros(1, 4, 2),
            torch.tensor([True]),
        )
    with pytest.raises(ValueError, match="spatial size"):
        loss_fn(
            torch.rand(1, 3, 32, 64),
            torch.rand(1, 3, 32, 64),
            torch.zeros(1, 4, 2),
            torch.zeros(1, 4, 2),
            torch.tensor([True]),
        )


def test_refiner_loss_buffer_moves_with_device():
    loss_fn = RefinerLoss()
    # CPU only here — just verify the buffer is registered and moves.
    loss_fn = loss_fn.to("cpu")
    assert loss_fn.center_weight.device.type == "cpu"
    assert loss_fn.center_weight.shape == (1, 1, 64, 128)
