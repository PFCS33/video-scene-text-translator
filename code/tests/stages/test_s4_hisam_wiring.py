"""Wiring tests for the ``"hisam"`` inpainter backend dispatch.

Verifies that :meth:`PropagationStage._get_inpainter` dispatches correctly to
:class:`SegmentationBasedInpainter` when
``propagation.inpainter_backend == "hisam"``. The underlying class is patched
at its import site so no real Hi-SAM checkpoint is loaded — the whole module
runs on CPU with no Hi-SAM dependencies.

See also :class:`TestS3InpainterWiring` in ``test_anytext2_editor.py`` for the
sibling SRNet wiring pattern.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from src.config import PipelineConfig
from src.stages.s4_propagation.stage import PropagationStage


def _make_config(
    *,
    backend: str = "hisam",
    checkpoint_path: str | None = "/fake/path.pth",
    device: str = "cpu",
    model_type: str = "vit_l",
    mask_dilation_px: int = 3,
    inpaint_method: str = "ns",
    use_patch_mode: bool = False,
) -> PipelineConfig:
    """Build a ``PipelineConfig`` with the given propagation inpainter knobs."""
    cfg = PipelineConfig()
    cfg.propagation.inpainter_backend = backend
    cfg.propagation.inpainter_checkpoint_path = checkpoint_path
    cfg.propagation.inpainter_device = device
    cfg.propagation.hisam_model_type = model_type
    cfg.propagation.hisam_mask_dilation_px = mask_dilation_px
    cfg.propagation.hisam_inpaint_method = inpaint_method
    cfg.propagation.hisam_use_patch_mode = use_patch_mode
    return cfg


class TestHiSAMWiring:
    """PropagationStage._get_inpainter dispatch for the Hi-SAM backend."""

    def test_hisam_backend_constructs_segmentation_inpainter(self):
        """All six constructor kwargs flow through from config to the class."""
        cfg = _make_config(
            checkpoint_path="/fake/path.pth",
            device="cpu",
            model_type="vit_b",
            mask_dilation_px=5,
            inpaint_method="telea",
            use_patch_mode=True,
        )
        patch_target = (
            "src.stages.s4_propagation.segmentation_inpainter."
            "SegmentationBasedInpainter"
        )
        with patch(patch_target) as mock_cls:
            stage = PropagationStage(cfg)
            result = stage._get_inpainter()

        assert result is mock_cls.return_value
        mock_cls.assert_called_once_with(
            checkpoint_path="/fake/path.pth",
            device="cpu",
            model_type="vit_b",
            mask_dilation_px=5,
            inpaint_method="telea",
            use_patch_mode=True,
        )

    def test_hisam_backend_no_checkpoint_returns_none_and_warns(self, caplog):
        """Missing checkpoint → returns ``None`` + emits a WARNING log."""
        cfg = _make_config(checkpoint_path=None)
        patch_target = (
            "src.stages.s4_propagation.segmentation_inpainter."
            "SegmentationBasedInpainter"
        )
        with patch(patch_target) as mock_cls:
            stage = PropagationStage(cfg)
            with caplog.at_level(logging.WARNING, logger="src.stages.s4_propagation.stage"):
                result = stage._get_inpainter()

        assert result is None
        mock_cls.assert_not_called()
        assert any(
            "hisam" in record.message.lower() and record.levelno == logging.WARNING
            for record in caplog.records
        ), f"expected a WARNING about hisam; got {caplog.records!r}"

    def test_hisam_backend_lazy_not_loaded_until_get_inpainter(self):
        """Construction alone must not trigger the Hi-SAM load."""
        cfg = _make_config()
        patch_target = (
            "src.stages.s4_propagation.segmentation_inpainter."
            "SegmentationBasedInpainter"
        )
        with patch(patch_target) as mock_cls:
            stage = PropagationStage(cfg)
            assert mock_cls.call_count == 0, (
                "SegmentationBasedInpainter should not be constructed in "
                "PropagationStage.__init__"
            )
            stage._get_inpainter()
            assert mock_cls.call_count == 1

    def test_hisam_backend_cached_after_first_call(self):
        """Second call to ``_get_inpainter`` must reuse the cached instance."""
        cfg = _make_config()
        patch_target = (
            "src.stages.s4_propagation.segmentation_inpainter."
            "SegmentationBasedInpainter"
        )
        with patch(patch_target) as mock_cls:
            stage = PropagationStage(cfg)
            first = stage._get_inpainter()
            second = stage._get_inpainter()

        assert first is second
        mock_cls.assert_called_once()

    def test_unknown_backend_raises_with_actionable_message(self):
        """Unknown backend → ValueError listing all valid choices."""
        cfg = _make_config(backend="bogus")
        stage = PropagationStage(cfg)
        with pytest.raises(ValueError) as exc_info:
            stage._get_inpainter()

        msg = str(exc_info.value)
        assert "bogus" in msg
        assert "srnet" in msg
        assert "hisam" in msg
        assert "none" in msg
