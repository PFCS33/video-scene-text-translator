"""Tests for HiSAMSegmenter (mocked — no Hi-SAM checkpoint / GPU needed).

These tests pin the wrapper's contract:

- ``__init__`` with ``checkpoint_path=None`` does NOT load the model.
- ``__init__`` with a concrete ``checkpoint_path`` eagerly loads (same
  behaviour as ``SRNetInpainter``).
- ``load_model()`` is idempotent.
- ``load_model()`` restores the process cwd after construction (the
  ``contextlib.chdir`` workaround for build.py's hardcoded relative encoder
  path must not leak into the caller's environment).
- ``segment()`` before ``load_model()`` raises ``RuntimeError``.
- ``segment()`` returns a uint8 H*W binary mask with the same H*W as input
  and values in {0, 255}.

The Hi-SAM model is stubbed via ``unittest.mock.patch`` on the
``hi_sam.modeling.build.model_registry`` and ``hi_sam.modeling.predictor``
modules — no real weights or GPU involved.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.stages.s4_propagation.hisam_segmenter import HiSAMSegmenter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install_fake_hi_sam(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Register fake ``hi_sam.modeling.build`` + ``.predictor`` modules.

    Returns a dict with handles to the fake model and predictor factories so
    individual tests can assert against them.
    """
    import types

    # Fake model: returns itself from eval()/to() so chained calls work.
    fake_model = MagicMock(name="FakeHiSam")
    fake_model.eval.return_value = fake_model
    fake_model.to.return_value = fake_model
    fake_model.mask_threshold = 0.0

    def _make_model(args):  # noqa: ANN001
        # Capture the args passed by HiSAMSegmenter so tests can inspect them.
        _make_model.last_args = args
        return fake_model

    _make_model.last_args = None

    fake_model_registry = {"vit_b": _make_model, "vit_l": _make_model, "vit_h": _make_model}

    # Fake predictor: predict() returns a canned high-res mask shaped (1, H, W).
    class _FakePredictor:
        def __init__(self, model):  # noqa: ANN001
            self.model = model
            self._last_image: np.ndarray | None = None

        def set_image(self, image, image_format: str = "RGB"):  # noqa: ANN001
            self._last_image = image
            self._last_format = image_format

        def predict(self, multimask_output: bool = True, return_logits: bool = False):
            assert self._last_image is not None
            h, w = self._last_image.shape[:2]
            # Half-and-half mask: left half True, right half False.
            hr = np.zeros((1, h, w), dtype=bool)
            hr[:, :, : w // 2] = True
            mask = hr.copy()
            score = np.array([1.0])
            hr_score = np.array([1.0])
            if return_logits:
                # Float logits thresholded by model.mask_threshold.
                logits = np.where(hr, 1.0, -1.0).astype(np.float32)
                return mask, logits, score, hr_score
            return mask, hr, score, hr_score

    # Inject ``hi_sam`` + ``hi_sam.modeling`` + ``hi_sam.modeling.build``
    # + ``hi_sam.modeling.predictor`` as synthetic modules in ``sys.modules``
    # so the wrapper's ``import hi_sam.modeling.build`` works without cwd
    # being the Hi-SAM repo. We also patch sys.path management so the real
    # Hi-SAM repo isn't accidentally imported.
    pkg = types.ModuleType("hi_sam")
    pkg.__path__ = []  # mark as package
    modeling = types.ModuleType("hi_sam.modeling")
    modeling.__path__ = []
    build = types.ModuleType("hi_sam.modeling.build")
    build.model_registry = fake_model_registry
    predictor_mod = types.ModuleType("hi_sam.modeling.predictor")
    predictor_mod.SamPredictor = _FakePredictor

    monkeypatch.setitem(sys.modules, "hi_sam", pkg)
    monkeypatch.setitem(sys.modules, "hi_sam.modeling", modeling)
    monkeypatch.setitem(sys.modules, "hi_sam.modeling.build", build)
    monkeypatch.setitem(sys.modules, "hi_sam.modeling.predictor", predictor_mod)

    return {
        "model": fake_model,
        "make_model": _make_model,
        "predictor_cls": _FakePredictor,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInitDoesNotEagerLoadWithoutCheckpoint:
    def test_lazy_when_checkpoint_path_is_none(self):
        seg = HiSAMSegmenter(checkpoint_path=None, device="cpu")
        # No internal predictor until load_model() is called.
        assert seg._predictor is None


class TestEagerLoadWithCheckpointPath:
    def test_eager_when_checkpoint_path_given(self, monkeypatch):
        _install_fake_hi_sam(monkeypatch)
        seg = HiSAMSegmenter(
            checkpoint_path="/tmp/fake_ckpt.pth",
            device="cpu",
            model_type="vit_l",
        )
        assert seg._predictor is not None


class TestLoadModelIdempotent:
    def test_calling_load_model_twice_constructs_once(self, monkeypatch):
        _install_fake_hi_sam(monkeypatch)
        seg = HiSAMSegmenter(checkpoint_path=None, device="cpu", model_type="vit_l")
        assert seg._predictor is None

        seg._checkpoint_path = "/tmp/fake_ckpt.pth"  # satisfy internal guard
        seg.load_model()
        first_predictor = seg._predictor
        assert first_predictor is not None

        # Second call must be a no-op: same predictor instance.
        seg.load_model()
        assert seg._predictor is first_predictor


class TestCwdRestoredAfterLoad:
    def test_cwd_unchanged_across_load_model(self, monkeypatch, tmp_path):
        _install_fake_hi_sam(monkeypatch)
        # Move to a known non-repo directory to detect any leaked chdir.
        monkeypatch.chdir(tmp_path)
        before = os.getcwd()

        HiSAMSegmenter(
            checkpoint_path="/tmp/fake_ckpt.pth",
            device="cpu",
            model_type="vit_l",
        )

        after = os.getcwd()
        assert after == before, (
            f"load_model() leaked a chdir: before={before!r} after={after!r}"
        )


class TestSegmentRequiresLoadedModel:
    def test_segment_raises_when_not_loaded(self):
        seg = HiSAMSegmenter(checkpoint_path=None, device="cpu")
        roi = np.zeros((32, 48, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="no weights loaded"):
            seg.segment(roi)


class TestSegmentReturnsBinaryMask:
    def test_shape_dtype_and_values_single_pass(self, monkeypatch):
        _install_fake_hi_sam(monkeypatch)
        seg = HiSAMSegmenter(
            checkpoint_path="/tmp/fake_ckpt.pth",
            device="cpu",
            model_type="vit_l",
            use_patch_mode=False,
        )
        roi = (np.random.rand(40, 60, 3) * 255).astype(np.uint8)

        mask = seg.segment(roi)

        assert mask.shape == (40, 60)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask).tolist()).issubset({0, 255})
        # Left half is 255, right half is 0 (from fake predictor).
        assert mask[:, :30].mean() > 200
        assert mask[:, 30:].mean() < 55

    def test_patch_mode_still_returns_binary_mask(self, monkeypatch):
        _install_fake_hi_sam(monkeypatch)
        seg = HiSAMSegmenter(
            checkpoint_path="/tmp/fake_ckpt.pth",
            device="cpu",
            model_type="vit_l",
            use_patch_mode=True,
        )
        # Make it bigger than patch_size=512 so patchify_sliding actually
        # splits into multiple tiles.
        roi = (np.random.rand(600, 700, 3) * 255).astype(np.uint8)

        mask = seg.segment(roi)

        assert mask.shape == (600, 700)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask).tolist()).issubset({0, 255})
