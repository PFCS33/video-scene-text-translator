"""Unit tests for ``src.models.refiner.model.ROIRefiner``.

Covers construction, forward shapes, initialization behavior (so the network
starts from near-identity), differentiability, and integration with the
``warp.py`` building blocks downstream.
"""

from __future__ import annotations

import pytest
import torch

from src.models.refiner.model import ROIRefiner
from src.models.refiner.warp import canonical_corners, corners_to_H, warp_image

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_default_construction():
    model = ROIRefiner()
    assert model.image_size == (64, 128)
    # Parameter count is dominated by FC1 (4096 -> 256). Sanity check the
    # overall budget is in the "~1M" ballpark the plan targets.
    n = model.num_parameters()
    assert 0.9e6 < n < 2.0e6, f"unexpected param count {n}"


def test_rejects_non_16_multiple_image_size():
    with pytest.raises(ValueError, match="divisible by 16"):
        ROIRefiner(image_size=(50, 100))


def test_different_image_size_works():
    model = ROIRefiner(image_size=(32, 64))
    x_s = torch.randn(1, 3, 32, 64)
    x_t = torch.randn(1, 3, 32, 64)
    out = model(x_s, x_t)
    assert out.shape == (1, 4, 2)


# ---------------------------------------------------------------------------
# Forward shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_forward_shape(batch_size: int):
    model = ROIRefiner()
    S = torch.randn(batch_size, 3, 64, 128)
    T = torch.randn(batch_size, 3, 64, 128)
    out = model(S, T)
    assert out.shape == (batch_size, 4, 2)
    assert out.dtype == torch.float32


def test_forward_shape_validation_mismatched_shapes():
    model = ROIRefiner()
    S = torch.randn(1, 3, 64, 128)
    T = torch.randn(1, 3, 64, 64)
    with pytest.raises(ValueError, match="same shape"):
        model(S, T)


def test_forward_shape_validation_wrong_channels():
    model = ROIRefiner()
    S = torch.randn(1, 1, 64, 128)
    T = torch.randn(1, 1, 64, 128)
    with pytest.raises(ValueError, match=r"\(B, 3, H, W\)"):
        model(S, T)


def test_forward_shape_validation_wrong_spatial():
    model = ROIRefiner()
    S = torch.randn(1, 3, 32, 64)
    T = torch.randn(1, 3, 32, 64)
    with pytest.raises(ValueError, match="expected spatial size"):
        model(S, T)


# ---------------------------------------------------------------------------
# Initialization behavior: starts from near-identity
# ---------------------------------------------------------------------------

def test_initial_output_near_zero():
    """With head_init_scale=1e-3 the untrained model should emit tiny offsets.

    The plan's whole training setup (residual-around-identity, regularization
    toward zero) assumes we start near zero; this pins that property so a
    future refactor can't silently drop the small init without failing CI.
    """
    torch.manual_seed(0)
    model = ROIRefiner(head_init_scale=1e-3)
    model.eval()  # freeze BN running stats and disable dropout
    S = torch.rand(4, 3, 64, 128)
    T = torch.rand(4, 3, 64, 128)
    with torch.no_grad():
        out = model(S, T)
    # Initial output should be well below our ±8 px perturbation range.
    assert out.abs().max().item() < 0.5, (
        f"initial output too large: max={out.abs().max().item():.4f}"
    )


def test_initial_output_larger_with_bigger_init_scale():
    """Sanity: raising head_init_scale actually makes initial output bigger."""
    torch.manual_seed(0)
    S = torch.rand(4, 3, 64, 128)
    T = torch.rand(4, 3, 64, 128)

    small = ROIRefiner(head_init_scale=1e-3).eval()
    large = ROIRefiner(head_init_scale=1.0).eval()
    with torch.no_grad():
        out_s = small(S, T).abs().mean().item()
        out_l = large(S, T).abs().mean().item()
    assert out_l > out_s * 10


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------

def test_backward_populates_gradients():
    model = ROIRefiner()
    S = torch.rand(2, 3, 64, 128, requires_grad=True)
    T = torch.rand(2, 3, 64, 128, requires_grad=True)
    out = model(S, T)
    loss = (out ** 2).sum()
    loss.backward()

    # Input gradients exist and are finite
    assert S.grad is not None
    assert T.grad is not None
    assert torch.isfinite(S.grad).all()
    assert torch.isfinite(T.grad).all()

    # Every model parameter received a finite gradient
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"params with no gradient: {missing}"
    non_finite = [
        name for name, p in model.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    assert not non_finite, f"params with non-finite gradient: {non_finite}"


# ---------------------------------------------------------------------------
# Train vs eval mode
# ---------------------------------------------------------------------------

def test_eval_mode_is_deterministic():
    """Two forward passes in eval mode produce identical output (dropout off)."""
    torch.manual_seed(0)
    model = ROIRefiner().eval()
    S = torch.rand(2, 3, 64, 128)
    T = torch.rand(2, 3, 64, 128)
    with torch.no_grad():
        a = model(S, T)
        b = model(S, T)
    assert torch.allclose(a, b)


def test_train_mode_dropout_is_stochastic():
    """In train mode, dropout should sometimes produce different outputs."""
    torch.manual_seed(0)
    model = ROIRefiner(dropout=0.5).train()
    # Use the same input; two forward passes through dropout should differ
    # given a high enough dropout rate.
    S = torch.rand(2, 3, 64, 128)
    T = torch.rand(2, 3, 64, 128)
    a = model(S, T)
    b = model(S, T)
    assert not torch.allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------------
# Integration with warp.py
# ---------------------------------------------------------------------------

def test_model_output_feeds_warp_pipeline():
    """End-to-end shape pin: model output -> corners_to_H -> warp_image.

    Catches any breakage where the model output shape drifts away from what
    the warp helpers expect.
    """
    torch.manual_seed(0)
    model = ROIRefiner().eval()
    B, H, W = 3, 64, 128
    S = torch.rand(B, 3, H, W)
    T = torch.rand(B, 3, H, W)

    with torch.no_grad():
        delta = model(S, T)                              # (B, 4, 2)

    src = canonical_corners(H, W).unsqueeze(0).expand(B, -1, -1).contiguous()
    dst = src + delta
    H_pred = corners_to_H(src, dst)                       # (B, 3, 3)
    warped = warp_image(S, H_pred, (H, W))                # (B, 3, H, W)

    assert warped.shape == (B, 3, H, W)
    assert torch.isfinite(warped).all()

    # Because initial ΔH ≈ I, the warped source should be very close to S.
    diff = (warped - S).abs().mean().item()
    assert diff < 0.05, f"initial warp drift too large: {diff}"


def test_model_gradients_flow_through_warp():
    """Gradients from a warp-based loss must reach model parameters.

    This pins the whole training loss path: if something later makes the
    homography composition non-differentiable (e.g., an in-place cast), this
    test will catch it.
    """
    torch.manual_seed(0)
    model = ROIRefiner()
    B, H, W = 2, 64, 128
    S = torch.rand(B, 3, H, W)
    T = torch.rand(B, 3, H, W)

    delta = model(S, T)
    src = canonical_corners(H, W).unsqueeze(0).expand(B, -1, -1).contiguous()
    dst = src + delta
    H_pred = corners_to_H(src, dst)
    warped = warp_image(S, H_pred, (H, W))

    loss = (warped - T).pow(2).mean()
    loss.backward()

    # Backbone + both FC layers should have non-zero gradients.
    any_nonzero = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters()
    )
    assert any_nonzero, "no parameter received a non-zero gradient"
