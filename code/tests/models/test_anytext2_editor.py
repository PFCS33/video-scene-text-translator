"""Tests for AnyText2Editor (mocked — no server needed)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.config import TextEditorConfig
from src.models.anytext2_editor import AnyText2Editor


@pytest.fixture
def editor_config() -> TextEditorConfig:
    return TextEditorConfig(
        backend="anytext2",
        server_url="http://fake-server:7860/",
        server_timeout=10,
        anytext2_ddim_steps=5,
        anytext2_cfg_scale=7.5,
        anytext2_strength=1.0,
        anytext2_img_count=1,
        anytext2_min_gen_size=512,
    )


@pytest.fixture
def editor(editor_config: TextEditorConfig) -> AnyText2Editor:
    return AnyText2Editor(editor_config)


class TestColorExtraction:
    def test_dark_text_on_light_bg(self):
        """Dark text on a white background should extract a dark color."""
        img = np.full((100, 200, 3), 240, dtype=np.uint8)  # light bg
        # Draw dark text-like pixels in center
        img[25:75, 50:150] = 30
        color = AnyText2Editor._extract_text_color(img)
        # Should be a dark color (low RGB values)
        assert color.startswith("#")
        assert len(color) == 7
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        assert r < 100 and g < 100 and b < 100

    def test_light_text_on_dark_bg(self):
        """Light text on a dark background should extract a light color."""
        img = np.full((100, 200, 3), 20, dtype=np.uint8)  # dark bg
        img[25:75, 50:150] = 220  # light text
        color = AnyText2Editor._extract_text_color(img)
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        assert r > 150 and g > 150 and b > 150

    def test_tiny_image_returns_black(self):
        """Very small image where interior is empty should return black."""
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        color = AnyText2Editor._extract_text_color(img)
        assert color == "#000000"

    def test_hex_format(self):
        """Color should always be a valid 7-char hex string."""
        img = np.random.randint(0, 256, (80, 160, 3), dtype=np.uint8)
        color = AnyText2Editor._extract_text_color(img)
        assert color.startswith("#")
        assert len(color) == 7
        # Should be parseable
        int(color[1:], 16)


class TestPrepareRoi:
    """Tests for _prepare_roi: upscale, 64-align, padding offsets."""

    def _dims(self, img: np.ndarray) -> tuple[int, int]:
        return img.shape[0], img.shape[1]

    def _is_64_aligned(self, img: np.ndarray) -> bool:
        h, w = img.shape[:2]
        return h % 64 == 0 and w % 64 == 0

    # -- Already large enough, 64-aligned --

    def test_already_large_and_aligned_unchanged(self):
        """512×512 image needs no changes."""
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        result, rect, _scale = AnyText2Editor._prepare_roi(img, min_gen_size=512)
        assert result.shape == (512, 512, 3)
        assert rect == (0, 512, 0, 512)

    def test_large_but_not_aligned_gets_padded(self):
        """600×400: already >= 512 but not 64-aligned → pad to 640×448."""
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        result, rect, _scale = AnyText2Editor._prepare_roi(img, min_gen_size=512)
        assert self._is_64_aligned(result)
        # Content should be 400×600 inside the padded image
        t, b, le, r = rect
        assert b - t == 400
        assert r - le == 600

    # -- Small ROIs: upscale to min_gen_size --

    def test_small_roi_upscaled_to_512(self):
        """150×40 ROI should be upscaled so max dim reaches 512."""
        img = np.zeros((40, 150, 3), dtype=np.uint8)
        result, rect, _scale = AnyText2Editor._prepare_roi(img, min_gen_size=512)
        assert self._is_64_aligned(result)
        # Content should be upscaled: 150*(512/150)=512, 40*(512/150)≈137
        t, b, le, r = rect
        assert r - le == 512
        assert b - t == round(40 * (512 / 150))
        # Final image should be >= 512 on longest side
        assert max(result.shape[:2]) >= 512
        # Height: 137 → 64-align → 192, but 192 < _MIN_DIM(256) → pad to 256
        assert result.shape[0] == 256

    def test_small_square_upscaled(self):
        """100×100 ROI should upscale to 512×512."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result, rect, _scale = AnyText2Editor._prepare_roi(img, min_gen_size=512)
        assert result.shape == (512, 512, 3)
        assert rect == (0, 512, 0, 512)

    # -- Too large: downscale --

    def test_too_large_scaled_down(self):
        """2000×600 should scale down so max dim ≤ 1024."""
        img = np.zeros((600, 2000, 3), dtype=np.uint8)
        result, rect, _scale = AnyText2Editor._prepare_roi(img, min_gen_size=512)
        assert self._is_64_aligned(result)
        assert max(result.shape[:2]) <= 1024
        t, b, le, r = rect
        assert r - le <= 1024

    # -- 64-alignment --

    def test_all_dimensions_64_aligned(self):
        """Output dimensions should always be multiples of 64."""
        test_cases = [
            (40, 150),   # small, wide
            (80, 300),   # medium, wide
            (200, 200),  # square
            (500, 80),   # tall-ish, narrow
            (1000, 30),  # extreme aspect ratio
            (600, 2000), # too large
        ]
        for h, w in test_cases:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            result, _, _s = AnyText2Editor._prepare_roi(img, min_gen_size=512)
            assert self._is_64_aligned(result), (
                f"Input ({w}×{h}) → output ({result.shape[1]}×{result.shape[0]}) "
                f"not 64-aligned"
            )

    # -- Content rect integrity --

    def test_content_rect_preserves_pixel_count(self):
        """Content rect dimensions should match upscaled (pre-pad) size."""
        img = np.full((40, 150, 3), 42, dtype=np.uint8)
        result, (t, b, le, r), _scale = AnyText2Editor._prepare_roi(img, min_gen_size=512)
        content = result[t:b, le:r]
        # Content should be non-zero (filled with 42)
        assert content.shape[2] == 3
        assert np.all(content == 42)

    def test_padding_region_is_border_replicate(self):
        """Padding should use BORDER_REPLICATE (edge pixels repeated)."""
        # Create image with distinct edge colors
        img = np.zeros((40, 150, 3), dtype=np.uint8)
        img[0, :] = [255, 0, 0]   # top edge red
        img[-1, :] = [0, 255, 0]  # bottom edge green
        result, (t, b, le, r), _scale = AnyText2Editor._prepare_roi(img, min_gen_size=512)
        if t > 0:
            # Top padding should replicate the top edge (red)
            assert np.all(result[0, le:r] == [255, 0, 0])
        if b < result.shape[0]:
            # Bottom padding should replicate the bottom edge (green)
            assert np.all(result[-1, le:r] == [0, 255, 0])

    # -- min_gen_size clamping --

    def test_min_gen_size_clamped_to_256_floor(self):
        """min_gen_size below 256 should be clamped to 256."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        result, _, _s = AnyText2Editor._prepare_roi(img, min_gen_size=100)
        assert self._is_64_aligned(result)
        assert max(result.shape[:2]) >= 256

    def test_min_gen_size_clamped_to_1024_ceiling(self):
        """min_gen_size above 1024 should be clamped to 1024."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        result, (t, b, le, r), _s = AnyText2Editor._prepare_roi(img, min_gen_size=2000)
        assert max(result.shape[:2]) <= 1088  # 1024 + up to 63 for alignment
        assert r - le <= 1024

    # -- Extreme aspect ratios --

    def test_extreme_wide_aspect_ratio(self):
        """1000×30 ROI: max=1000 already ≥ 512, no upscale. Pad to 64-align."""
        img = np.zeros((30, 1000, 3), dtype=np.uint8)
        result, (t, b, le, r), _scale = AnyText2Editor._prepare_roi(img, min_gen_size=512)
        assert self._is_64_aligned(result)
        # Width 1000 stays, only padded to next 64-multiple (1024)
        assert r - le == 1000
        # Height 30 padded to at least 64 (next multiple of 64)
        assert result.shape[0] >= 64

    def test_extreme_wide_small_aspect_ratio(self):
        """400×20 ROI: max=400 < 512, upscale to 512. Pad short axis."""
        img = np.zeros((20, 400, 3), dtype=np.uint8)
        result, (t, b, le, r), _scale = AnyText2Editor._prepare_roi(img, min_gen_size=512)
        assert self._is_64_aligned(result)
        assert r - le == 512  # long side upscaled to min_gen_size
        assert result.shape[0] >= 64  # short side padded to 64-align


class TestEdgeCase:
    def test_empty_roi_returns_as_is(self, editor: AnyText2Editor):
        roi = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        result = editor.edit_text(roi, "TEST")
        assert result.size == 0

    def test_tiny_roi_returns_as_is(self, editor: AnyText2Editor):
        roi = np.zeros((3, 3, 3), dtype=np.uint8)
        result = editor.edit_text(roi, "TEST")
        assert result.shape == (3, 3, 3)

    def test_no_server_url_raises(self):
        config = TextEditorConfig(backend="anytext2", server_url=None)
        editor = AnyText2Editor(config)
        with pytest.raises(ValueError, match="server_url"):
            editor._get_client()


def _make_mock_handle_file():
    """Create a mock handle_file that returns the path string as-is."""
    return lambda path: path


class TestGradioCall:
    @staticmethod
    def _fake_server_result(tmpdir: str, h: int, w: int) -> MagicMock:
        """Set up a mock Gradio client that returns a fake image of given size."""
        fake_result_path = str(Path(tmpdir) / "result.png")
        fake_img = np.full((h, w, 3), 128, dtype=np.uint8)
        cv2.imwrite(fake_result_path, fake_img)

        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.result.return_value = (
            [{"image": fake_result_path}],
            "debug info",
        )
        mock_client.submit.return_value = mock_job
        return mock_client

    @patch("src.models.anytext2_editor.AnyText2Editor._get_client")
    @patch.dict("sys.modules", {"gradio_client": MagicMock(handle_file=_make_mock_handle_file())})
    def test_edit_text_calls_predict(self, mock_get_client, editor: AnyText2Editor):
        """edit_text should call the Gradio submit endpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 300×500 → upscale to 512×854? No, max=500<512 → scale=512/500
            # Actually: 300×500, scale=512/500=1.024, → 307×512, pad to 320×512
            # Server returns the padded size
            mock_client = self._fake_server_result(tmpdir, 320, 512)
            mock_get_client.return_value = mock_client

            roi = np.full((300, 500, 3), 200, dtype=np.uint8)
            result = editor.edit_text(roi, "HOLA")

            # Verify submit was called with /process_1
            mock_client.submit.assert_called_once()
            call_kwargs = mock_client.submit.call_args
            assert call_kwargs.kwargs.get("api_name") == "/process_1"
            assert call_kwargs.kwargs.get("text_prompt") == '"HOLA"'

            # Result should match original dimensions (cropped + downscaled)
            assert result.shape == (300, 500, 3)

    @patch("src.models.anytext2_editor.AnyText2Editor._get_client")
    @patch.dict("sys.modules", {"gradio_client": MagicMock(handle_file=_make_mock_handle_file())})
    def test_empty_gallery_raises(self, mock_get_client, editor: AnyText2Editor):
        """Empty gallery response should raise RuntimeError."""
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.result.return_value = ([], "error info")
        mock_client.submit.return_value = mock_job
        mock_get_client.return_value = mock_client

        roi = np.full((300, 500, 3), 200, dtype=np.uint8)
        with pytest.raises(RuntimeError, match="empty gallery"):
            editor.edit_text(roi, "FAIL")

    @patch("src.models.anytext2_editor.AnyText2Editor._get_client")
    @patch.dict("sys.modules", {"gradio_client": MagicMock(handle_file=_make_mock_handle_file())})
    def test_small_roi_result_cropped_and_resized(self, mock_get_client, editor: AnyText2Editor):
        """Small ROI should be upscaled, and result cropped back to original."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 50×100 ROI → upscale: 100*(512/100)=512, 50*(512/100)=256
            # → 256×512, already 64-aligned → no extra padding
            mock_client = self._fake_server_result(tmpdir, 256, 512)
            mock_get_client.return_value = mock_client

            roi = np.full((50, 100, 3), 200, dtype=np.uint8)
            result = editor.edit_text(roi, "SMALL")

            assert result.shape == (50, 100, 3)

    @patch("src.models.anytext2_editor.AnyText2Editor._get_client")
    @patch.dict("sys.modules", {"gradio_client": MagicMock(handle_file=_make_mock_handle_file())})
    def test_localized_mask_written(self, mock_get_client, editor: AnyText2Editor):
        """Mask should only cover the content region, not padding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 40×150 → upscale: 150*(512/150)=512, 40*(512/150)≈137
            # → 137×512, 64-align h→192, but 192<256 (_MIN_DIM) → pad to 256×512
            mock_client = self._fake_server_result(tmpdir, 256, 512)
            mock_get_client.return_value = mock_client

            roi = np.full((40, 150, 3), 200, dtype=np.uint8)

            # Intercept imwrite to capture the mask
            written_masks = []
            original_imwrite = cv2.imwrite

            def capture_imwrite(path, img, *args, **kwargs):
                if "mask" in path:
                    written_masks.append(img.copy())
                return original_imwrite(path, img, *args, **kwargs)

            with patch("cv2.imwrite", side_effect=capture_imwrite):
                editor.edit_text(roi, "TEST")

            assert len(written_masks) == 1
            mask = written_masks[0]
            assert mask.shape[2] == 4  # RGBA

            alpha = mask[:, :, 3]
            # Padding rows should have alpha=0
            # Content region should have alpha=255
            assert alpha.max() == 255
            # Not all pixels should be 255 (padding exists since 137 < 192)
            total_pixels = alpha.shape[0] * alpha.shape[1]
            masked_pixels = np.count_nonzero(alpha)
            assert masked_pixels < total_pixels, "Mask should not cover entire image"


class TestParseResult:
    def test_unexpected_gallery_format_raises(self, editor: AnyText2Editor):
        """Gallery with unexpected entry format should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="Unexpected gallery format"):
            editor._parse_result(([{"not_image": "value"}], "debug"))

    def test_failed_image_read_raises(self, editor: AnyText2Editor):
        """Gallery pointing to non-existent file should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="Failed to read"):
            editor._parse_result(
                ([{"image": "/nonexistent/file.png"}], "debug")
            )


class TestConnectionError:
    @patch.dict("sys.modules", {"gradio_client": MagicMock()})
    def test_connection_error_wraps(self):
        """Failed connection should raise ConnectionError with clear message."""
        import sys
        mock_gc = sys.modules["gradio_client"]
        mock_gc.Client.side_effect = Exception("Connection refused")

        config = TextEditorConfig(
            backend="anytext2", server_url="http://bad-host:9999/"
        )
        ed = AnyText2Editor(config)
        with pytest.raises(ConnectionError, match="Cannot connect"):
            ed._get_client()


class TestConfigValidation:
    def test_anytext2_without_url_fails_validation(self):
        """Pipeline config with anytext2 backend but no URL should fail."""
        from src.config import PipelineConfig

        config = PipelineConfig()
        config.text_editor.backend = "anytext2"
        config.text_editor.server_url = None
        config.input_video = "test.mp4"
        config.output_video = "out.mp4"
        errors = config.validate()
        assert any("server_url" in e for e in errors)

    def test_anytext2_with_url_passes_validation(self):
        """Pipeline config with anytext2 backend and URL should pass."""
        from src.config import PipelineConfig

        config = PipelineConfig()
        config.text_editor.backend = "anytext2"
        config.text_editor.server_url = "http://localhost:45843/"
        config.input_video = "test.mp4"
        config.output_video = "out.mp4"
        errors = config.validate()
        assert not any("server_url" in e for e in errors)


class TestPrepareRoiScaleReturn:
    """Verify _prepare_roi returns the correct scale factor."""

    def test_scale_1_when_no_resize(self):
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        _, _, scale = AnyText2Editor._prepare_roi(img, min_gen_size=512)
        assert scale == 1.0

    def test_scale_upscale(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, _, scale = AnyText2Editor._prepare_roi(img, min_gen_size=512)
        assert scale == pytest.approx(512 / 100)

    def test_scale_downscale(self):
        img = np.zeros((600, 2000, 3), dtype=np.uint8)
        _, _, scale = AnyText2Editor._prepare_roi(img, min_gen_size=512)
        assert scale == pytest.approx(1024 / 2000)


class TestEditRegionMask:
    """Verify mask targets only the edit_region when provided."""

    @patch("src.models.anytext2_editor.AnyText2Editor._get_client")
    @patch.dict("sys.modules", {"gradio_client": MagicMock(handle_file=_make_mock_handle_file())})
    def test_mask_covers_edit_region_only(self, mock_get_client, editor: AnyText2Editor):
        """When edit_region is given, mask should be smaller than content_rect."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 400×400 expanded ROI, text is in the center 200×200
            mock_client = MagicMock()
            # Result image: 448×448 (400 padded to 64-align)
            fake_result_path = str(Path(tmpdir) / "result.png")
            cv2.imwrite(fake_result_path, np.full((448, 448, 3), 128, dtype=np.uint8))
            mock_job = MagicMock()
            mock_job.result.return_value = (
                [{"image": fake_result_path}], "debug",
            )
            mock_client.submit.return_value = mock_job
            mock_get_client.return_value = mock_client

            roi = np.full((400, 400, 3), 200, dtype=np.uint8)
            edit_region = (100, 300, 100, 300)  # center 200×200

            written_masks = []
            original_imwrite = cv2.imwrite

            def capture_imwrite(path, img, *args, **kwargs):
                if "mask" in path:
                    written_masks.append(img.copy())
                return original_imwrite(path, img, *args, **kwargs)

            with patch("cv2.imwrite", side_effect=capture_imwrite):
                editor.edit_text(roi, "TEST", edit_region=edit_region)

            assert len(written_masks) == 1
            alpha = written_masks[0][:, :, 3]
            masked_pixels = np.count_nonzero(alpha)
            total_pixels = alpha.size
            # With edit_region, masked area should be much less than total
            # (200×200 = 40000 vs 400×400 = 160000 content area)
            assert masked_pixels < total_pixels * 0.5


class TestS3Integration:
    def test_anytext2_backend_init(self):
        """S3 stage should create AnyText2Editor when backend is 'anytext2'."""
        from src.config import PipelineConfig
        from src.stages.s3_text_editing import TextEditingStage

        config = PipelineConfig()
        config.text_editor.backend = "anytext2"
        config.text_editor.server_url = "http://fake:7860/"
        stage = TextEditingStage(config)

        # _init_editor should create an AnyText2Editor (won't connect yet — lazy)
        editor = stage._init_editor()
        assert isinstance(editor, AnyText2Editor)


class _FakeInpainter:
    """Mock BaseBackgroundInpainter: marks every pixel it sees as 42.

    Lets tests distinguish "touched by inpainter" from "original pixels"
    without needing an actual model or checkpoint.
    """

    def __init__(self):
        self.call_count = 0
        self.last_shape: tuple[int, ...] | None = None

    def inpaint(self, canonical_roi: np.ndarray) -> np.ndarray:
        self.call_count += 1
        self.last_shape = canonical_roi.shape
        return np.full_like(canonical_roi, 42)


class TestAdaptiveMask:
    """Adaptive mask sizing flow: shrink mask + pre-inpaint for long→short.

    Covers:
    - Trigger when target aspect is much narrower than source canonical
    - Skip (fast path) when within tolerance
    - Skip when config flag is off
    - Skip (with warning) when no inpainter is provided
    - Graceful fallback on inpainter exception
    - Caller's roi_image is not mutated
    """

    @staticmethod
    def _server_returning(tmpdir: str, h: int, w: int):
        fake_path = str(Path(tmpdir) / "result.png")
        cv2.imwrite(fake_path, np.full((h, w, 3), 128, dtype=np.uint8))
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.result.return_value = ([{"image": fake_path}], "debug")
        mock_client.submit.return_value = mock_job
        return mock_client

    @staticmethod
    def _capture_masks(editor_call):
        """Run *editor_call* while recording imwrites with 'mask' in path.

        Returns a dict keyed by basename so callers can distinguish the
        main edit mask (``mask.png``) from the font-mimic mask
        (``mimic_mask.png``).
        """
        import os

        captured: dict[str, np.ndarray] = {}
        original_imwrite = cv2.imwrite

        def capture(path, img, *args, **kwargs):
            if "mask" in path:
                captured[os.path.basename(path)] = img.copy()
            return original_imwrite(path, img, *args, **kwargs)

        with patch("cv2.imwrite", side_effect=capture):
            editor_call()
        return captured

    def _adaptive_config(self, **overrides) -> TextEditorConfig:
        cfg = TextEditorConfig(
            backend="anytext2",
            server_url="http://fake-server:7860/",
            server_timeout=10,
            anytext2_ddim_steps=5,
            anytext2_adaptive_mask=True,
            anytext2_mask_aspect_tolerance=0.15,
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    @patch("src.models.anytext2_editor.AnyText2Editor._get_client")
    @patch.dict("sys.modules", {"gradio_client": MagicMock(handle_file=_make_mock_handle_file())})
    def test_long_to_short_triggers_inpaint_and_narrows_mask(self, mock_get_client):
        """7:1 canonical + 3-char CJK target → mask should shrink + inpainter runs."""
        cfg = self._adaptive_config(roi_context_expansion=0.3)
        inpainter = _FakeInpainter()
        editor = AnyText2Editor(cfg, inpainter=inpainter)

        mock_client = None
        with tempfile.TemporaryDirectory() as tmpdir:
            # Canonical-ish 7:1 ROI: 560×80 scaled up to AnyText2's grid
            mock_client = self._server_returning(tmpdir, 512, 3584)
            mock_get_client.return_value = mock_client
            roi = np.full((80, 560, 3), 200, dtype=np.uint8)

            captured = self._capture_masks(
                lambda: editor.edit_text(roi, "我是示"),
            )

        # Inpainter was called exactly once on the full ROI
        assert inpainter.call_count == 1
        assert inpainter.last_shape == (80, 560, 3)

        # Both main (narrow) and font-mimic (wide) masks written
        assert "mask.png" in captured
        assert "mimic_mask.png" in captured

        # Main edit mask is narrowed, sent on a tighter (cropped) canvas.
        # With crop, mask covers ~62% of the smaller canvas — still less
        # than 100% but higher than the pre-crop ~43%.
        main_alpha = captured["mask.png"][:, :, 3]
        main_cols = int((main_alpha[main_alpha.shape[0] // 2] == 255).sum())
        main_total = main_alpha.shape[1]
        assert main_cols < 0.75 * main_total

        # Font-mimic mask: on its own (larger) canvas, covers nearly all
        # columns so the font encoder sees complete source glyphs.
        mimic_alpha = captured["mimic_mask.png"][:, :, 3]
        mimic_cols = int((mimic_alpha[mimic_alpha.shape[0] // 2] == 255).sum())
        mimic_total = mimic_alpha.shape[1]
        assert mimic_cols > main_cols
        assert mimic_cols >= 0.9 * mimic_total

    @patch("src.models.anytext2_editor.AnyText2Editor._get_client")
    @patch.dict("sys.modules", {"gradio_client": MagicMock(handle_file=_make_mock_handle_file())})
    def test_within_tolerance_skips_inpaint(self, mock_get_client):
        """Close aspect → adaptive flow skipped, inpainter untouched."""
        cfg = self._adaptive_config()
        inpainter = _FakeInpainter()
        editor = AnyText2Editor(cfg, inpainter=inpainter)

        with tempfile.TemporaryDirectory() as tmpdir:
            # 3:1 canonical + 3-char CJK target → exact aspect match
            mock_get_client.return_value = self._server_returning(tmpdir, 512, 1536)
            roi = np.full((80, 240, 3), 200, dtype=np.uint8)

            editor.edit_text(roi, "我是示")

        assert inpainter.call_count == 0

    @patch("src.models.anytext2_editor.AnyText2Editor._get_client")
    @patch.dict("sys.modules", {"gradio_client": MagicMock(handle_file=_make_mock_handle_file())})
    def test_adaptive_flag_false_skips_inpaint(self, mock_get_client):
        """anytext2_adaptive_mask=False → adaptive flow disabled entirely."""
        cfg = self._adaptive_config(anytext2_adaptive_mask=False)
        inpainter = _FakeInpainter()
        editor = AnyText2Editor(cfg, inpainter=inpainter)

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_get_client.return_value = self._server_returning(tmpdir, 512, 3584)
            roi = np.full((80, 560, 3), 200, dtype=np.uint8)
            editor.edit_text(roi, "我是示")

        assert inpainter.call_count == 0

    @patch("src.models.anytext2_editor.AnyText2Editor._get_client")
    @patch.dict("sys.modules", {"gradio_client": MagicMock(handle_file=_make_mock_handle_file())})
    def test_no_inpainter_logs_warning_and_skips(self, mock_get_client, caplog):
        """adaptive_mask=True but no inpainter provided → warning + fallback."""
        cfg = self._adaptive_config()
        editor = AnyText2Editor(cfg, inpainter=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_get_client.return_value = self._server_returning(tmpdir, 512, 3584)
            roi = np.full((80, 560, 3), 200, dtype=np.uint8)
            with caplog.at_level("WARNING"):
                editor.edit_text(roi, "我是示")

        assert any(
            "adaptive_mask" in rec.message and "no inpainter" in rec.message
            for rec in caplog.records
        )

    @patch("src.models.anytext2_editor.AnyText2Editor._get_client")
    @patch.dict("sys.modules", {"gradio_client": MagicMock(handle_file=_make_mock_handle_file())})
    def test_no_inpainter_warning_only_once(self, mock_get_client, caplog):
        """Warning should be rate-limited to once per editor instance."""
        cfg = self._adaptive_config()
        editor = AnyText2Editor(cfg, inpainter=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_get_client.return_value = self._server_returning(tmpdir, 512, 3584)
            roi = np.full((80, 560, 3), 200, dtype=np.uint8)
            with caplog.at_level("WARNING"):
                editor.edit_text(roi, "我是示")
                editor.edit_text(roi, "你好")

        # Exactly one warning across both calls
        msgs = [r.message for r in caplog.records if "adaptive_mask" in r.message]
        assert len(msgs) == 1

    @patch("src.models.anytext2_editor.AnyText2Editor._get_client")
    @patch.dict("sys.modules", {"gradio_client": MagicMock(handle_file=_make_mock_handle_file())})
    def test_inpainter_exception_falls_back_gracefully(self, mock_get_client, caplog):
        """Inpainter raises → log warning, use non-adaptive mask, don't crash."""
        cfg = self._adaptive_config()

        class _BrokenInpainter:
            def inpaint(self, x):
                raise RuntimeError("OOM simulated")

        editor = AnyText2Editor(cfg, inpainter=_BrokenInpainter())

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_get_client.return_value = self._server_returning(tmpdir, 512, 3584)
            roi = np.full((80, 560, 3), 200, dtype=np.uint8)

            with caplog.at_level("WARNING"):
                result = editor.edit_text(roi, "我是示")

        # Should not crash; result is the server's output, re-cropped
        assert result.shape == (80, 560, 3)
        assert any("inpainter failed" in r.message for r in caplog.records)

    @patch("src.models.anytext2_editor.AnyText2Editor._get_client")
    @patch.dict("sys.modules", {"gradio_client": MagicMock(handle_file=_make_mock_handle_file())})
    def test_caller_roi_not_mutated(self, mock_get_client):
        """Adaptive flow must not modify the caller's input array."""
        cfg = self._adaptive_config()
        editor = AnyText2Editor(cfg, inpainter=_FakeInpainter())

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_get_client.return_value = self._server_returning(tmpdir, 512, 3584)
            roi = np.full((80, 560, 3), 200, dtype=np.uint8)
            roi_snapshot = roi.copy()

            editor.edit_text(roi, "我是示")

        assert np.array_equal(roi, roi_snapshot)

    @patch.dict("sys.modules", {"gradio_client": MagicMock(handle_file=_make_mock_handle_file())})
    def test_adaptive_crop_sends_smaller_canvas(self):
        """When adaptive fires with expanded ROI, server receives a tighter canvas.

        Expanded ROI is 1120×128 (canonical 700×80 + 30% expansion).
        Target "我是示" → adaptive mask 240px → crop ~384×128.
        After _prepare_roi: ~512×256 instead of ~1024×256.
        """
        cfg = self._adaptive_config(roi_context_expansion=0.3)
        inpainter = _FakeInpainter()
        editor = AnyText2Editor(cfg, inpainter=inpainter)

        # Simulate expanded ROI: 700×80 canonical + 30% expansion
        roi = np.full((128, 1120, 3), 200, dtype=np.uint8)
        edit_region = (24, 104, 210, 910)  # canonical within expanded

        server_calls = []

        def fake_call_server(
            ori_path, mask_path, target_text, text_color, w, h, **kwargs,
        ):
            server_calls.append({"w": w, "h": h})
            return np.full((h, w, 3), 128, dtype=np.uint8)

        with patch.object(editor, "_call_server", side_effect=fake_call_server):
            result = editor.edit_text(roi, "我是示", edit_region=edit_region)

        assert len(server_calls) == 1
        w_sent = server_calls[0]["w"]
        # Without crop: _prepare_roi(1120×128) → w ≈ 1024
        # With crop (~384×128): _prepare_roi → w ≈ 512
        assert w_sent <= 640, f"Expected smaller canvas w, got w={w_sent}"

        # Result must still match the original expanded ROI shape
        assert result.shape == (128, 1120, 3)


class TestS3InpainterWiring:
    """TextEditingStage creates and forwards an inpainter to AnyText2Editor."""

    def test_s3_passes_inpainter_when_configured(self, tmp_path, monkeypatch):
        """When propagation.inpainter_backend='srnet' + checkpoint + anytext2 +
        adaptive_mask=True, the editor should receive an inpainter instance."""
        # Stub out SRNetInpainter to avoid loading a real checkpoint
        import src.stages.s4_propagation.srnet_inpainter as srnet_mod
        from src.config import PipelineConfig
        from src.stages.s3_text_editing import TextEditingStage

        class _StubSRNetInpainter:
            def __init__(self, checkpoint_path, device):
                self.checkpoint_path = checkpoint_path
                self.device = device

            def inpaint(self, x):
                return x

        monkeypatch.setattr(srnet_mod, "SRNetInpainter", _StubSRNetInpainter)

        # Any non-empty path works since we stubbed the class
        fake_ckpt = tmp_path / "fake.pth"
        fake_ckpt.write_bytes(b"not a real checkpoint")

        config = PipelineConfig()
        config.text_editor.backend = "anytext2"
        config.text_editor.server_url = "http://fake:7860/"
        config.text_editor.anytext2_adaptive_mask = True
        config.propagation.inpainter_backend = "srnet"
        config.propagation.inpainter_checkpoint_path = str(fake_ckpt)
        config.propagation.inpainter_device = "cpu"

        stage = TextEditingStage(config)
        editor = stage._init_editor()

        assert isinstance(editor, AnyText2Editor)
        assert editor._inpainter is not None
        assert isinstance(editor._inpainter, _StubSRNetInpainter)

    def test_s3_no_inpainter_when_adaptive_off(self):
        """adaptive_mask=False → don't even touch the inpainter path."""
        from src.config import PipelineConfig
        from src.stages.s3_text_editing import TextEditingStage

        config = PipelineConfig()
        config.text_editor.backend = "anytext2"
        config.text_editor.server_url = "http://fake:7860/"
        config.text_editor.anytext2_adaptive_mask = False
        config.propagation.inpainter_backend = "srnet"
        config.propagation.inpainter_checkpoint_path = "/whatever/ckpt"

        stage = TextEditingStage(config)
        editor = stage._init_editor()

        assert isinstance(editor, AnyText2Editor)
        assert editor._inpainter is None

    def test_s3_no_inpainter_when_backend_unset(self):
        """adaptive_mask=True but no propagation inpainter → editor gets None."""
        from src.config import PipelineConfig
        from src.stages.s3_text_editing import TextEditingStage

        config = PipelineConfig()
        config.text_editor.backend = "anytext2"
        config.text_editor.server_url = "http://fake:7860/"
        config.text_editor.anytext2_adaptive_mask = True
        # propagation.inpainter_backend stays at default (None/empty)

        stage = TextEditingStage(config)
        editor = stage._init_editor()

        assert isinstance(editor, AnyText2Editor)
        assert editor._inpainter is None

    def test_s3_passes_hisam_inpainter_when_configured(self, tmp_path, monkeypatch):
        """backend='hisam' + checkpoint + adaptive_mask → editor gets a
        SegmentationBasedInpainter. Regression guard for the S3 fallback that
        silently disabled adaptive mask when users set inpainter_backend to
        'hisam' (S3 only knew about 'srnet')."""
        import src.stages.s4_propagation.segmentation_inpainter as seg_mod
        from src.config import PipelineConfig
        from src.stages.s3_text_editing import TextEditingStage

        class _StubHiSAMInpainter:
            def __init__(self, checkpoint_path, device, model_type,
                         mask_dilation_px, inpaint_method, use_patch_mode):
                self.checkpoint_path = checkpoint_path
                self.device = device
                self.model_type = model_type
                self.mask_dilation_px = mask_dilation_px
                self.inpaint_method = inpaint_method
                self.use_patch_mode = use_patch_mode

            def inpaint(self, x):
                return x

        monkeypatch.setattr(
            seg_mod, "SegmentationBasedInpainter", _StubHiSAMInpainter,
        )

        fake_ckpt = tmp_path / "hisam.pth"
        fake_ckpt.write_bytes(b"not a real checkpoint")

        config = PipelineConfig()
        config.text_editor.backend = "anytext2"
        config.text_editor.server_url = "http://fake:7860/"
        config.text_editor.anytext2_adaptive_mask = True
        config.propagation.inpainter_backend = "hisam"
        config.propagation.inpainter_checkpoint_path = str(fake_ckpt)
        config.propagation.inpainter_device = "cpu"
        config.propagation.hisam_model_type = "vit_b"
        config.propagation.hisam_mask_dilation_px = 5
        config.propagation.hisam_inpaint_method = "telea"
        config.propagation.hisam_use_patch_mode = True

        stage = TextEditingStage(config)
        editor = stage._init_editor()

        assert isinstance(editor, AnyText2Editor)
        assert isinstance(editor._inpainter, _StubHiSAMInpainter)
        assert editor._inpainter.checkpoint_path == str(fake_ckpt)
        assert editor._inpainter.device == "cpu"
        assert editor._inpainter.model_type == "vit_b"
        assert editor._inpainter.mask_dilation_px == 5
        assert editor._inpainter.inpaint_method == "telea"
        assert editor._inpainter.use_patch_mode is True

    def test_s3_hisam_no_checkpoint_falls_back_to_none(self, caplog):
        """backend='hisam' + empty checkpoint path → warn + editor gets None.
        Same graceful-fallback contract as the srnet branch."""
        import logging

        from src.config import PipelineConfig
        from src.stages.s3_text_editing import TextEditingStage

        config = PipelineConfig()
        config.text_editor.backend = "anytext2"
        config.text_editor.server_url = "http://fake:7860/"
        config.text_editor.anytext2_adaptive_mask = True
        config.propagation.inpainter_backend = "hisam"
        config.propagation.inpainter_checkpoint_path = None

        stage = TextEditingStage(config)
        with caplog.at_level(logging.WARNING, logger="src.stages.s3_text_editing"):
            editor = stage._init_editor()

        assert isinstance(editor, AnyText2Editor)
        assert editor._inpainter is None
        assert any("hisam" in r.message.lower() for r in caplog.records)
