"""Tests for SegmentationBasedInpainter (mocked — no Hi-SAM checkpoint needed).

These tests pin the API contract for the Hi-SAM segmentation-based
background inpainter before the implementation lands (TDD). They exercise:

- BGR-in / BGR-out shape & dtype contract.
- Mask dilation is applied before ``cv2.inpaint`` runs.
- ``inpaint_method`` ("ns" vs "telea") maps to the matching cv2 flag.
- Input shape/dtype validation (grayscale, RGBA, float32 all raise).
- Lazy ``HiSAMSegmenter`` construction — deferred until first ``inpaint()``.
- Config fields flow into the ``HiSAMSegmenter`` constructor correctly.

Until ``src.stages.s4_propagation.segmentation_inpainter`` exists, the whole
module will fail at collection time with ``ModuleNotFoundError``. That is the
expected state at the end of plan Step 2.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.stages.s4_propagation.segmentation_inpainter import (
    SegmentationBasedInpainter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeSegmenter:
    """Test double for HiSAMSegmenter.

    Records each ``segment()`` call (input H/W + count) so tests can inspect
    how the inpainter drove it, and returns a canned mask. Default mask is a
    centred rectangle of 255s in a field of 0s. ``load_model`` counts calls
    so lazy-load tests can verify it was / wasn't invoked.
    """

    def __init__(self, mask: np.ndarray | None = None):
        self._mask = mask
        self.calls: list[tuple[int, int]] = []  # (H, W) of each input
        self.load_model_count = 0

    def segment(self, bgr_roi: np.ndarray) -> np.ndarray:
        h, w = bgr_roi.shape[:2]
        self.calls.append((h, w))
        if self._mask is not None:
            return self._mask
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 255
        return mask

    def load_model(self) -> None:
        self.load_model_count += 1


def _dummy_roi(h: int = 64, w: int = 128, fill: int = 200) -> np.ndarray:
    """Construct a valid BGR uint8 ROI for passing through the inpainter."""
    return np.full((h, w, 3), fill, dtype=np.uint8)


# ---------------------------------------------------------------------------
# TestShapeContract
# ---------------------------------------------------------------------------


class TestShapeContract:
    def test_bgr_in_bgr_out_same_shape(self):
        """Feeding a (64, 128, 3) uint8 ROI returns a (64, 128, 3) uint8 ROI."""
        fake = _FakeSegmenter()
        inpainter = SegmentationBasedInpainter(segmenter=fake)
        roi = _dummy_roi(64, 128)

        result = inpainter.inpaint(roi)

        assert result.shape == (64, 128, 3)
        assert result.dtype == np.uint8

    def test_preserves_unmasked_pixels(self):
        """All-zero mask = nothing to inpaint → output equals input."""
        h, w = 32, 64
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        fake = _FakeSegmenter(mask=empty_mask)
        inpainter = SegmentationBasedInpainter(segmenter=fake)
        roi = _dummy_roi(h, w, fill=177)

        result = inpainter.inpaint(roi)

        assert np.array_equal(result, roi)


# ---------------------------------------------------------------------------
# TestDilation
# ---------------------------------------------------------------------------


class TestDilation:
    def test_mask_is_dilated_before_inpaint(self):
        """A 1-pixel mask with dilation=5 must reach cv2.inpaint dilated."""
        h, w = 40, 40
        single_px = np.zeros((h, w), dtype=np.uint8)
        single_px[h // 2, w // 2] = 255
        fake = _FakeSegmenter(mask=single_px)
        inpainter = SegmentationBasedInpainter(
            segmenter=fake, mask_dilation_px=5
        )
        roi = _dummy_roi(h, w)

        captured = {}

        def fake_inpaint(src, mask, radius, flags):
            captured["mask"] = mask.copy()
            return src.copy()

        with patch(
            "src.stages.s4_propagation.segmentation_inpainter.cv2.inpaint",
            side_effect=fake_inpaint,
        ):
            inpainter.inpaint(roi)

        assert "mask" in captured
        nonzero = int(np.count_nonzero(captured["mask"]))
        assert nonzero > 1, (
            f"Expected dilated mask (>1 non-zero pixels), got {nonzero}"
        )

    def test_zero_dilation_preserves_mask_exactly(self):
        """mask_dilation_px=0 → mask passed to cv2.inpaint == segmenter output."""
        h, w = 40, 40
        seg_mask = np.zeros((h, w), dtype=np.uint8)
        seg_mask[10:20, 15:25] = 255
        fake = _FakeSegmenter(mask=seg_mask)
        inpainter = SegmentationBasedInpainter(
            segmenter=fake, mask_dilation_px=0
        )
        roi = _dummy_roi(h, w)

        captured = {}

        def fake_inpaint(src, mask, radius, flags):
            captured["mask"] = mask.copy()
            return src.copy()

        with patch(
            "src.stages.s4_propagation.segmentation_inpainter.cv2.inpaint",
            side_effect=fake_inpaint,
        ):
            inpainter.inpaint(roi)

        assert np.array_equal(captured["mask"], seg_mask)


# ---------------------------------------------------------------------------
# TestInpaintMethodSwitch
# ---------------------------------------------------------------------------


class TestInpaintMethodSwitch:
    def test_ns_method_maps_to_inpaint_ns(self):
        """inpaint_method='ns' → cv2.INPAINT_NS flag passed."""
        fake = _FakeSegmenter()
        inpainter = SegmentationBasedInpainter(
            segmenter=fake, inpaint_method="ns"
        )
        roi = _dummy_roi()

        with patch(
            "src.stages.s4_propagation.segmentation_inpainter.cv2.inpaint",
            return_value=roi.copy(),
        ) as mock_inpaint:
            inpainter.inpaint(roi)

        mock_inpaint.assert_called_once()
        call = mock_inpaint.call_args
        flags = call.kwargs.get("flags")
        if flags is None and len(call.args) >= 4:
            flags = call.args[3]
        assert flags == cv2.INPAINT_NS

    def test_telea_method_maps_to_inpaint_telea(self):
        """inpaint_method='telea' → cv2.INPAINT_TELEA flag passed."""
        fake = _FakeSegmenter()
        inpainter = SegmentationBasedInpainter(
            segmenter=fake, inpaint_method="telea"
        )
        roi = _dummy_roi()

        with patch(
            "src.stages.s4_propagation.segmentation_inpainter.cv2.inpaint",
            return_value=roi.copy(),
        ) as mock_inpaint:
            inpainter.inpaint(roi)

        mock_inpaint.assert_called_once()
        call = mock_inpaint.call_args
        flags = call.kwargs.get("flags")
        if flags is None and len(call.args) >= 4:
            flags = call.args[3]
        assert flags == cv2.INPAINT_TELEA

    def test_invalid_inpaint_method_raises_at_construction(self):
        """inpaint_method='bogus' → ValueError at __init__, not inpaint()."""
        with pytest.raises(ValueError, match="inpaint_method"):
            SegmentationBasedInpainter(
                segmenter=_FakeSegmenter(), inpaint_method="bogus"
            )


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_2d_input_raises(self):
        """(H, W) grayscale image → ValueError."""
        inpainter = SegmentationBasedInpainter(segmenter=_FakeSegmenter())
        gray = np.zeros((32, 32), dtype=np.uint8)
        with pytest.raises(ValueError):
            inpainter.inpaint(gray)

    def test_4channel_input_raises(self):
        """(H, W, 4) RGBA image → ValueError."""
        inpainter = SegmentationBasedInpainter(segmenter=_FakeSegmenter())
        rgba = np.zeros((32, 32, 4), dtype=np.uint8)
        with pytest.raises(ValueError):
            inpainter.inpaint(rgba)

    def test_wrong_dtype_raises(self):
        """float32 (H, W, 3) image → ValueError."""
        inpainter = SegmentationBasedInpainter(segmenter=_FakeSegmenter())
        float_roi = np.zeros((32, 32, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            inpainter.inpaint(float_roi)


# ---------------------------------------------------------------------------
# TestLazyLoad
# ---------------------------------------------------------------------------


class TestLazyLoad:
    def test_no_segmenter_construction_in_init(self):
        """Constructing the inpainter must NOT build a HiSAMSegmenter."""
        with patch(
            "src.stages.s4_propagation.segmentation_inpainter.HiSAMSegmenter"
        ) as MockSeg:
            SegmentationBasedInpainter()
            MockSeg.assert_not_called()

    def test_segmenter_constructed_on_first_inpaint(self):
        """First inpaint() call must construct the HiSAMSegmenter exactly once."""
        with patch(
            "src.stages.s4_propagation.segmentation_inpainter.HiSAMSegmenter"
        ) as MockSeg:
            mock_instance = MagicMock()
            mock_instance.segment.return_value = np.zeros(
                (16, 32), dtype=np.uint8
            )
            MockSeg.return_value = mock_instance

            inpainter = SegmentationBasedInpainter(
                checkpoint_path="/fake/ckpt.pth",
                device="cpu",
                model_type="vit_l",
                use_patch_mode=False,
            )
            MockSeg.assert_not_called()  # still lazy

            inpainter.inpaint(_dummy_roi(16, 32))

            MockSeg.assert_called_once()
            kwargs = MockSeg.call_args.kwargs
            # Fields forwarded from inpainter config to segmenter constructor:
            assert kwargs.get("checkpoint_path") == "/fake/ckpt.pth"
            assert kwargs.get("device") == "cpu"
            assert kwargs.get("model_type") == "vit_l"
            assert kwargs.get("use_patch_mode") is False
            mock_instance.load_model.assert_called_once()

    def test_segmenter_not_reconstructed_on_second_inpaint(self):
        """Second inpaint() reuses the same segmenter — no rebuild, no reload."""
        with patch(
            "src.stages.s4_propagation.segmentation_inpainter.HiSAMSegmenter"
        ) as MockSeg:
            mock_instance = MagicMock()
            mock_instance.segment.return_value = np.zeros(
                (16, 32), dtype=np.uint8
            )
            MockSeg.return_value = mock_instance

            inpainter = SegmentationBasedInpainter(
                checkpoint_path="/fake/ckpt.pth", device="cpu"
            )
            inpainter.inpaint(_dummy_roi(16, 32))
            inpainter.inpaint(_dummy_roi(16, 32))

            assert MockSeg.call_count == 1
            assert mock_instance.load_model.call_count == 1

    def test_injected_segmenter_used_as_is(self):
        """segmenter=... injection path must not touch HiSAMSegmenter at all."""
        with patch(
            "src.stages.s4_propagation.segmentation_inpainter.HiSAMSegmenter"
        ) as MockSeg:
            fake = _FakeSegmenter()
            inpainter = SegmentationBasedInpainter(segmenter=fake)
            inpainter.inpaint(_dummy_roi(32, 64))
            inpainter.inpaint(_dummy_roi(32, 64))

            MockSeg.assert_not_called()
            assert len(fake.calls) == 2


# ---------------------------------------------------------------------------
# TestConfigMapping
# ---------------------------------------------------------------------------


class TestConfigMapping:
    def test_config_fields_reach_segmenter_constructor(self):
        """All four Hi-SAM config fields flow into HiSAMSegmenter().

        The two *wrapper*-level fields (``mask_dilation_px``,
        ``inpaint_method``) must NOT be forwarded to the segmenter — they
        belong to the inpainter wrapper.
        """
        with patch(
            "src.stages.s4_propagation.segmentation_inpainter.HiSAMSegmenter"
        ) as MockSeg:
            mock_instance = MagicMock()
            mock_instance.segment.return_value = np.zeros(
                (16, 32), dtype=np.uint8
            )
            MockSeg.return_value = mock_instance

            inpainter = SegmentationBasedInpainter(
                checkpoint_path="/foo/bar.pth",
                device="cpu",
                model_type="vit_b",
                use_patch_mode=True,
                mask_dilation_px=5,
                inpaint_method="telea",
            )
            inpainter.inpaint(_dummy_roi(16, 32))

            MockSeg.assert_called_once()
            kwargs = MockSeg.call_args.kwargs
            assert kwargs.get("checkpoint_path") == "/foo/bar.pth"
            assert kwargs.get("device") == "cpu"
            assert kwargs.get("model_type") == "vit_b"
            assert kwargs.get("use_patch_mode") is True
            # Wrapper-only fields are NOT passed through to the segmenter.
            assert "mask_dilation_px" not in kwargs
            assert "inpaint_method" not in kwargs
