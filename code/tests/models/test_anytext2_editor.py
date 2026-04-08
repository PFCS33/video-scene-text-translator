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
