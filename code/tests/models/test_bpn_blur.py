"""Tests for DifferentiableBlur.

Regression guard for the border-darkening fix: the convolution inside
``DifferentiableBlur`` must be mean-preserving all the way to the ROI
edge. Zero-padding (the F.conv2d default) darkens borders, which is
amplified by S5's Poisson compositing into a visible halo around every
propagated ROI. Reflect-padding keeps the blur energy-conserving at
the boundary. See bpn/blur.py for the fix and CHANGELOG for the
incident this test pins.
"""

from __future__ import annotations

import torch

from src.models.bpn.blur import DifferentiableBlur


def _identity_params(batch: int = 1):
    """Params that make DifferentiableBlur an identity (w=0)."""
    return {
        "sigma_x": torch.full((batch,), 1.0),
        "sigma_y": torch.full((batch,), 1.0),
        "rho": torch.zeros(batch),
        "w": torch.zeros(batch),
    }


class TestBorderPreservation:
    """The convolution should not darken the border of the blurred output.

    These tests fail with F.conv2d(padding=pad) zero-padding and pass
    with the reflect-padding fix in blur.py.
    """

    def test_flat_image_is_preserved_exactly(self):
        """A constant image must emerge unchanged, including the border.

        The blur output feeds into ``(1+w)*I - w*blurred``, so if
        ``blurred`` equals ``I`` everywhere for a flat input, the full
        formula collapses to ``I``. Zero-padding breaks this invariant
        at the border.
        """
        blur = DifferentiableBlur(kernel_size=41)
        img = torch.full((1, 3, 80, 200), 0.5)
        sx = torch.tensor([15.0])
        sy = torch.tensor([8.0])
        rho = torch.tensor([0.0])
        w = torch.tensor([-0.5])

        out = blur(img, sx, sy, rho, w)
        assert out.shape == img.shape
        # The output should equal the input at every pixel, including
        # the corners. 1e-5 absolute tolerance is generous given fp32.
        torch.testing.assert_close(out, img, atol=1e-5, rtol=0.0)

    def test_border_not_darker_than_interior_on_random_image(self):
        """No systematic border-vs-interior gap for random images.

        Zero-padding produces a ~|w|·I darker rim near the border. With
        reflect-padding, the border and interior means should agree
        within sampling noise of the random content.
        """
        torch.manual_seed(0)
        blur = DifferentiableBlur(kernel_size=41)
        img = torch.rand(1, 3, 80, 200)
        sx = torch.tensor([30.0])
        sy = torch.tensor([15.0])
        rho = torch.tensor([0.0])
        w = torch.tensor([-0.5])

        out = blur(img, sx, sy, rho, w)
        border_slices = [
            out[:, :, :5, :],
            out[:, :, -5:, :],
            out[:, :, :, :5],
            out[:, :, :, -5:],
        ]
        border_mean = torch.cat([s.flatten() for s in border_slices]).mean()
        interior_mean = out[:, :, 20:-20, 20:-20].mean()

        # Allow up to 0.02 of sample-mean noise between 5-px border
        # strips and the interior on a random 80×200 image. With the
        # zero-padding bug, this gap grew to ~0.06 at |w|=0.5.
        assert abs(float(interior_mean - border_mean)) < 0.02

    def test_identity_params_is_identity(self):
        """w=0 must return the input exactly (no blur, no shift)."""
        blur = DifferentiableBlur(kernel_size=41)
        torch.manual_seed(1)
        img = torch.rand(1, 3, 64, 128)
        p = _identity_params(batch=1)
        out = blur(img, p["sigma_x"], p["sigma_y"], p["rho"], p["w"])
        torch.testing.assert_close(out, img, atol=1e-6, rtol=0.0)

    def test_small_image_smaller_than_pad(self):
        """Images with H or W < kernel_size//2 must not crash.

        Canonical ROIs for short text tracks can be only 10-15 px tall.
        reflect padding requires pad < input_size, which would fail
        for a 41-px kernel (pad=20) on a 15-px-tall image. replicate
        padding has no such constraint.
        """
        blur = DifferentiableBlur(kernel_size=41)
        torch.manual_seed(2)
        # Heights and widths deliberately below pad=20 on at least one axis.
        for shape in [(1, 3, 15, 77), (1, 3, 10, 52), (1, 3, 25, 19)]:
            img = torch.rand(*shape)
            sx = torch.tensor([15.0])
            sy = torch.tensor([8.0])
            rho = torch.tensor([0.0])
            w = torch.tensor([-0.5])
            out = blur(img, sx, sy, rho, w)
            assert out.shape == img.shape
            assert torch.isfinite(out).all()


class TestShapeContract:
    """Basic shape and range contracts on the blur output."""

    def test_output_shape_matches_input(self):
        blur = DifferentiableBlur(kernel_size=21)
        img = torch.rand(2, 3, 40, 60)
        sx = torch.tensor([1.0, 2.0])
        sy = torch.tensor([1.0, 2.0])
        rho = torch.tensor([0.0, 0.5])
        w = torch.tensor([-0.3, 0.2])
        out = blur(img, sx, sy, rho, w)
        assert out.shape == img.shape

    def test_output_in_unit_range(self):
        blur = DifferentiableBlur(kernel_size=21)
        torch.manual_seed(2)
        img = torch.rand(1, 3, 40, 60)
        sx = torch.tensor([3.0])
        sy = torch.tensor([3.0])
        rho = torch.tensor([0.0])
        w = torch.tensor([0.9])  # aggressive sharpen, tests the clamp
        out = blur(img, sx, sy, rho, w)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0
