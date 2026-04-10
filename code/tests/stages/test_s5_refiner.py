"""Unit tests for ``src.stages.s5_revert.refiner.RefinerInference``.

Tests the inference wrapper in isolation: scale handling, sanity-check
fallbacks, and end-to-end compose math against a hand-built ground-truth
homography.

Strategy: build a tiny real ``ROIRefiner`` with near-zero init, save it
as a checkpoint, then instantiate ``RefinerInference`` on that checkpoint.
Individual tests can monkeypatch ``_model`` to return specific corner
predictions when we need exact arithmetic.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.models.refiner.model import ROIRefiner
from src.stages.s5_revert.refiner import RefinerInference

# ---------------------------------------------------------------------------
# Checkpoint fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> Path:
    """Save a minimal ROIRefiner checkpoint to ``tmp_path``."""
    model = ROIRefiner(
        base_channels=16, dropout=0.0, image_size=(64, 128), head_init_scale=1e-3,
    )
    ckpt = tmp_path / "refiner_v0.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "epoch": 0,
            "best_metric": 0.5,
            "config": {
                "data": {"image_size": [64, 128]},
                "model": {
                    "base_channels": 16,
                    "dropout": 0.0,
                    "head_init_scale": 1e-3,
                },
            },
        },
        ckpt,
    )
    return ckpt


# ---------------------------------------------------------------------------
# Lazy load
# ---------------------------------------------------------------------------


def test_construction_is_lazy(tiny_checkpoint: Path):
    """Constructor should not touch torch or read the checkpoint."""
    nonexistent = tiny_checkpoint.parent / "does_not_exist.pt"
    # Constructor must succeed even though the file doesn't exist
    refiner = RefinerInference(
        checkpoint_path=str(nonexistent), device="cpu",
    )
    assert refiner._model is None


def test_first_predict_triggers_load(tiny_checkpoint: Path):
    refiner = RefinerInference(checkpoint_path=str(tiny_checkpoint), device="cpu")
    assert refiner._model is None
    # Feed a small canonical ROI
    ref = np.full((80, 160, 3), 128, dtype=np.uint8)
    tgt = np.full((80, 160, 3), 128, dtype=np.uint8)
    result = refiner.predict_delta_H(ref, tgt)
    # Result is a 3x3 matrix (or None in case of rejection — both acceptable)
    assert result is None or result.shape == (3, 3)
    assert refiner._model is not None  # loaded


# ---------------------------------------------------------------------------
# Input validation (pre-model-load path)
# ---------------------------------------------------------------------------


def test_none_inputs_return_none(tiny_checkpoint: Path):
    refiner = RefinerInference(checkpoint_path=str(tiny_checkpoint), device="cpu")
    assert refiner.predict_delta_H(None, None) is None  # type: ignore[arg-type]


def test_wrong_shape_inputs_return_none(tiny_checkpoint: Path):
    refiner = RefinerInference(checkpoint_path=str(tiny_checkpoint), device="cpu")
    # 2D (missing channel dim)
    bad = np.zeros((64, 128), dtype=np.uint8)
    good = np.zeros((64, 128, 3), dtype=np.uint8)
    assert refiner.predict_delta_H(bad, good) is None  # type: ignore[arg-type]
    assert refiner.predict_delta_H(good, bad) is None  # type: ignore[arg-type]


def test_size_mismatch_returns_none(tiny_checkpoint: Path):
    refiner = RefinerInference(checkpoint_path=str(tiny_checkpoint), device="cpu")
    a = np.zeros((64, 128, 3), dtype=np.uint8)
    b = np.zeros((80, 160, 3), dtype=np.uint8)
    assert refiner.predict_delta_H(a, b) is None


def test_zero_size_returns_none(tiny_checkpoint: Path):
    refiner = RefinerInference(checkpoint_path=str(tiny_checkpoint), device="cpu")
    # Can't actually make a (0, 0, 3) array that passes earlier checks
    # without tripping numpy; use (0, 128, 3) to trigger the H==0 case.
    empty = np.zeros((0, 128, 3), dtype=np.uint8)
    assert refiner.predict_delta_H(empty, empty) is None


# ---------------------------------------------------------------------------
# Near-zero init -> near-identity prediction
# ---------------------------------------------------------------------------


def test_untrained_model_returns_near_identity(tiny_checkpoint: Path):
    """With head_init_scale=1e-3 the untrained model should produce
    near-zero corner offsets, so ΔH should pass all sanity checks and
    come out close to the identity matrix."""
    refiner = RefinerInference(checkpoint_path=str(tiny_checkpoint), device="cpu")
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 256, (80, 200, 3), dtype=np.uint8)
    tgt = rng.integers(0, 256, (80, 200, 3), dtype=np.uint8)
    delta_H = refiner.predict_delta_H(ref, tgt)
    assert delta_H is not None
    assert delta_H.shape == (3, 3)
    # Should be ~I with a tolerance comparable to the scaled init noise.
    assert np.allclose(delta_H, np.eye(3), atol=0.1)


# ---------------------------------------------------------------------------
# Scale handling: network (64, 128) -> canonical
# ---------------------------------------------------------------------------


def _patch_model_with_fixed_output(
    refiner: RefinerInference, delta_corners_net: np.ndarray,
) -> None:
    """Replace the refiner's model with one that emits a fixed prediction."""
    refiner._ensure_loaded()  # ensure the real model is loaded first

    class _FixedModel:
        def __init__(self, out):
            self.out = torch.from_numpy(out.astype(np.float32)).unsqueeze(0)

        def __call__(self, src, tgt):
            return self.out.to(src.device)

        def eval(self):
            return self

    refiner._model = _FixedModel(delta_corners_net)


def test_scale_unscales_to_canonical(tiny_checkpoint: Path):
    """Network predicts 8 px dx at (64, 128); on a (160, 320) canonical
    the scale should be ``320 / 128 = 2.5x`` in x and ``160 / 64 = 2.5x``
    in y. A pure +8 px x translation in network coords should become a
    pure +20 px x translation in canonical coords.
    """
    refiner = RefinerInference(
        checkpoint_path=str(tiny_checkpoint), device="cpu",
        max_corner_offset_px=100.0,
    )
    # Pure +8 x-translation at every corner at network resolution
    delta_net = np.array(
        [[8.0, 0.0], [8.0, 0.0], [8.0, 0.0], [8.0, 0.0]]
    )
    _patch_model_with_fixed_output(refiner, delta_net)

    ref = np.zeros((160, 320, 3), dtype=np.uint8)
    tgt = np.zeros((160, 320, 3), dtype=np.uint8)
    delta_H = refiner.predict_delta_H(ref, tgt)
    assert delta_H is not None

    # A pure +20 px x-translation homography has [0, 2] == 20 and the
    # other entries of the identity.
    expected = np.array(
        [[1.0, 0.0, 20.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    assert np.allclose(delta_H, expected, atol=1e-3)


def test_scale_anisotropic(tiny_checkpoint: Path):
    """Canonical (64, 256) has x-scale 2.0 and y-scale 1.0 against the
    (64, 128) network. Pure +1 y corner delta should stay +1 y in canonical,
    pure +1 x corner delta should become +2 x."""
    refiner = RefinerInference(
        checkpoint_path=str(tiny_checkpoint), device="cpu",
        max_corner_offset_px=100.0,
    )
    delta_net = np.array(
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
    )
    _patch_model_with_fixed_output(refiner, delta_net)

    ref = np.zeros((64, 256, 3), dtype=np.uint8)
    tgt = np.zeros((64, 256, 3), dtype=np.uint8)
    delta_H = refiner.predict_delta_H(ref, tgt)
    assert delta_H is not None
    # Pure translation by (+2, +1).
    expected = np.array(
        [[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
    )
    assert np.allclose(delta_H, expected, atol=1e-3)


# ---------------------------------------------------------------------------
# Sanity check rejections (plan §2.5)
# ---------------------------------------------------------------------------


def test_rejects_huge_offset(tiny_checkpoint: Path):
    refiner = RefinerInference(
        checkpoint_path=str(tiny_checkpoint), device="cpu",
        max_corner_offset_px=10.0,
    )
    # 30 px offset at network resolution on a 160x320 canonical
    # -> 75 px canonical offset, way over 10 threshold.
    delta_net = np.array(
        [[30.0, 0.0], [30.0, 0.0], [30.0, 0.0], [30.0, 0.0]]
    )
    _patch_model_with_fixed_output(refiner, delta_net)
    ref = np.zeros((160, 320, 3), dtype=np.uint8)
    tgt = np.zeros((160, 320, 3), dtype=np.uint8)
    assert refiner.predict_delta_H(ref, tgt) is None


def test_rejects_nan_model_output(tiny_checkpoint: Path):
    refiner = RefinerInference(
        checkpoint_path=str(tiny_checkpoint), device="cpu",
    )
    delta_net = np.full((4, 2), np.nan)
    _patch_model_with_fixed_output(refiner, delta_net)
    ref = np.zeros((80, 160, 3), dtype=np.uint8)
    tgt = np.zeros((80, 160, 3), dtype=np.uint8)
    assert refiner.predict_delta_H(ref, tgt) is None


def test_rejects_degenerate_determinant(tiny_checkpoint: Path):
    """A cross-corner-swap produces a degenerate homography with
    determinant far from 1 — must be rejected by check 2."""
    refiner = RefinerInference(
        checkpoint_path=str(tiny_checkpoint), device="cpu",
        max_corner_offset_px=10000.0,  # disable the offset check
    )
    # Make the quad collapse to a line — det should blow up or be ~0.
    # At (80, 160) canonical: a near-degenerate quad.
    delta_net = np.array(
        [[0.0, 0.0], [-60.0, 0.0], [0.0, 0.0], [60.0, 0.0]]
    )
    _patch_model_with_fixed_output(refiner, delta_net)
    ref = np.zeros((80, 160, 3), dtype=np.uint8)
    tgt = np.zeros((80, 160, 3), dtype=np.uint8)
    # Result: either None (rejected) or a valid matrix if by chance
    # the degenerate warp still has sane det/cond. Either is acceptable —
    # the test pins that we NEVER return a non-finite matrix and that
    # an obviously-degenerate prediction is rejected in at least one of
    # the sanity checks.
    result = refiner.predict_delta_H(ref, tgt)
    if result is not None:
        # Must satisfy all sanity checks if returned
        assert np.all(np.isfinite(result))
        det = np.linalg.det(result)
        assert 0.5 <= det <= 2.0
    # Crafted near-collapse should typically fail at least one check.
    # Use a stronger collapse so the test is deterministic.
    delta_net2 = np.array(
        [[0.0, 0.0], [-120.0, 0.0], [0.0, 0.0], [120.0, 0.0]]
    )
    _patch_model_with_fixed_output(refiner, delta_net2)
    assert refiner.predict_delta_H(ref, tgt) is None


def test_rejects_non_finite_canonical_H(tiny_checkpoint: Path):
    """If somehow cv2.getPerspectiveTransform produces a non-finite
    matrix, we must reject."""
    refiner = RefinerInference(
        checkpoint_path=str(tiny_checkpoint), device="cpu",
        max_corner_offset_px=1e9,  # don't bail on offset
    )
    # Make the dst quad have zero width AND zero height at one corner by
    # collapsing opposing corners onto each other. This produces a
    # singular 8x8 DLT system that cv2 flags as invalid.
    delta_net = np.array(
        [[160.0, 80.0], [0.0, 0.0], [-160.0, -80.0], [0.0, 0.0]]
    )
    _patch_model_with_fixed_output(refiner, delta_net)
    ref = np.zeros((80, 160, 3), dtype=np.uint8)
    tgt = np.zeros((80, 160, 3), dtype=np.uint8)
    result = refiner.predict_delta_H(ref, tgt)
    # Either None (rejected) or a finite matrix. Never an infected result.
    if result is not None:
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# End-to-end compose math
# ---------------------------------------------------------------------------


def test_predicted_H_approximately_aligns_known_warp(tiny_checkpoint: Path):
    """Hand-construct a warped pair and verify that when we feed a ΔH
    matching the known warp, the refiner's output produces a ΔH that
    (when applied) brings source close to target.

    Uses a patched model so we isolate the scale/compose math from the
    pretrained weights.
    """
    refiner = RefinerInference(
        checkpoint_path=str(tiny_checkpoint), device="cpu",
        max_corner_offset_px=100.0,
    )

    H_can, W_can = 96, 192
    rng = np.random.default_rng(42)
    ref = rng.integers(0, 256, (H_can, W_can, 3), dtype=np.uint8)

    # Build a known small corner offset in CANONICAL pixels
    delta_canonical = np.array(
        [[3.0, 1.0], [-2.0, 0.5], [1.0, -1.5], [0.0, 2.0]]
    )
    src_corners = np.array(
        [[0, 0], [W_can, 0], [W_can, H_can], [0, H_can]],
        dtype=np.float32,
    )
    H_true = cv2.getPerspectiveTransform(
        src_corners, (src_corners + delta_canonical).astype(np.float32),
    )
    tgt = cv2.warpPerspective(ref, H_true, (W_can, H_can))

    # Patch the model to predict the same delta at NETWORK resolution by
    # downscaling the canonical delta. Network is (64, 128), canonical is
    # (96, 192) -> scale x = 192/128 = 1.5, y = 96/64 = 1.5.
    H_net, W_net = 64, 128
    scale = np.array([W_can / W_net, H_can / H_net])
    delta_net = (delta_canonical / scale).astype(np.float32)
    _patch_model_with_fixed_output(refiner, delta_net)

    delta_H = refiner.predict_delta_H(ref, tgt)
    assert delta_H is not None

    # Applying delta_H to ref should approximate tgt closely.
    recovered = cv2.warpPerspective(ref, delta_H, (W_can, H_can))

    # Compare on the interior where the warp is valid.
    r0, r1, c0, c1 = 10, H_can - 10, 10, W_can - 10
    diff = np.abs(
        recovered[r0:r1, c0:c1].astype(np.int32)
        - tgt[r0:r1, c0:c1].astype(np.int32)
    )
    # Small warp + bilinear roundoff; be lenient.
    assert diff.mean() < 5.0, f"mean diff too large: {diff.mean()}"
