"""AnyText2 text editor backend via Gradio HTTP API.

Calls an external AnyText2 Gradio server to perform style-preserving
cross-language scene text editing.  The server runs in a separate conda
env (Python 3.10) to avoid dependency conflicts.

Requires:
    pip install gradio_client
    A running AnyText2 Gradio server (see third_party/install_anytext2.sh)
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.config import TextEditorConfig
from src.models.base_text_editor import BaseTextEditor

logger = logging.getLogger(__name__)

# AnyText2 dimension constraints.
MIN_DIM = 256   # Hard minimum accepted by the model
MAX_DIM = 1024  # Hard maximum accepted by the model
ALIGN = 64      # Dimensions must be multiples of this (SD VAE + U-Net)


class AnyText2Editor(BaseTextEditor):
    """Scene text editor that delegates to an AnyText2 Gradio server.

    Usage::

        editor = AnyText2Editor(config)
        edited = editor.edit_text(roi_image, "PELIGRO")
    """

    def __init__(self, config: TextEditorConfig):
        self.config = config
        self._client = None  # Lazy-init gradio Client

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _get_client(self):
        """Connect to the Gradio server on first use."""
        if self._client is not None:
            return self._client

        if not self.config.server_url:
            raise ValueError(
                "AnyText2 backend requires text_editor.server_url in config. "
                "Set it to the Gradio server URL, e.g. 'http://host:port/'."
            )

        from gradio_client import Client  # lazy import

        url = self.config.server_url.rstrip("/") + "/"
        timeout = self.config.server_timeout
        logger.info(
            "Connecting to AnyText2 server at %s (timeout=%ds)", url, timeout,
        )
        try:
            self._client = Client(
                url,
                httpx_kwargs={"timeout": timeout},
            )
        except Exception as exc:
            raise ConnectionError(
                f"Cannot connect to AnyText2 server at {url}. "
                "Is the server running? See third_party/install_anytext2.sh."
            ) from exc

        logger.info("Connected to AnyText2 server")
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def edit_text(
        self,
        roi_image: np.ndarray,
        target_text: str,
        edit_region: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """Replace text in *roi_image* with *target_text* via AnyText2.

        Args:
            roi_image: BGR image of the text region (H x W x 3, uint8).
            target_text: The translated text to render.
            edit_region: Optional (top, bottom, left, right) pixel coords
                within *roi_image* marking the area to edit.  When the ROI
                has been expanded with scene context, this restricts the
                AnyText2 mask to the text area only.  If *None*, the entire
                content region is edited.

        Returns:
            Edited BGR image, same shape as *roi_image*.
        """
        if roi_image.size == 0:
            logger.warning("AnyText2Editor: empty ROI, returning as-is")
            return roi_image

        h_orig, w_orig = roi_image.shape[:2]
        if h_orig < 5 or w_orig < 5:
            logger.warning("AnyText2Editor: ROI too small (%dx%d), returning as-is", w_orig, h_orig)
            return roi_image

        # Upscale, 64-align, and pad — returns prepared image + content region
        min_gen = self.config.anytext2_min_gen_size
        roi_prepared, content_rect, scale = self._prepare_roi(roi_image, min_gen)
        h_send, w_send = roi_prepared.shape[:2]
        ct, cb, cl, cr = content_rect  # top, bottom, left, right

        # Compute mask_rect: the region AnyText2 should edit.
        # When edit_region is provided, mask only that sub-area (text only).
        # Otherwise, mask the entire content rectangle.
        if edit_region is not None:
            et, eb, el, er = edit_region
            pad_top = ct
            pad_left = cl
            mask_rect = (
                max(0, pad_top + int(round(et * scale))),
                min(h_send, pad_top + int(round(eb * scale))),
                max(0, pad_left + int(round(el * scale))),
                min(w_send, pad_left + int(round(er * scale))),
            )
        else:
            mask_rect = content_rect

        mt, mb, ml, mr = mask_rect

        # Extract dominant text color from the text area only
        if edit_region is not None:
            et, eb, el, er = edit_region
            color_region = roi_image[et:eb, el:er]
        else:
            color_region = roi_image
        text_color = self._extract_text_color(color_region)

        # Save images to temp files (Gradio API needs file paths)
        with tempfile.TemporaryDirectory() as tmpdir:
            ori_path = str(Path(tmpdir) / "ori.png")
            mask_path = str(Path(tmpdir) / "mask.png")

            if not cv2.imwrite(ori_path, roi_prepared):
                raise RuntimeError(f"Failed to write temp ROI image to {ori_path}")

            # RGBA mask: alpha channel marks the edit region.
            # AnyText2 extracts the mask from layers[0][..., 3:] (alpha).
            mask = np.zeros((h_send, w_send, 4), dtype=np.uint8)
            mask[mt:mb, ml:mr, 3] = 255
            if not cv2.imwrite(mask_path, mask):
                raise RuntimeError(f"Failed to write temp mask image to {mask_path}")

            result_image = self._call_server(
                ori_path, mask_path, target_text, text_color, w_send, h_send,
            )

        # Crop out the content region (strip padding), then resize to original
        result_content = result_image[ct:cb, cl:cr]
        if result_content.shape[:2] != (h_orig, w_orig):
            result_content = cv2.resize(
                result_content, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4,
            )

        return result_content

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _call_server(
        self,
        ori_path: str,
        mask_path: str,
        target_text: str,
        text_color: str,
        w: int,
        h: int,
    ) -> np.ndarray:
        """Send an edit request to the AnyText2 Gradio server."""
        from gradio_client import handle_file  # lazy import

        client = self._get_client()

        # ref_img: background is the original, layers[0] is an RGBA image
        # where the alpha channel marks the edit region.
        ref_img = {
            "background": handle_file(ori_path),
            "layers": [handle_file(mask_path)],
            "composite": None,
            "id": None,
        }
        ori_img = handle_file(ori_path)

        # m1: ROI image for "Mimic From Image" font extraction.
        # Background is the source image; layer alpha marks font region.
        mimic_img = {
            "background": handle_file(ori_path),
            "layers": [handle_file(mask_path)],
            "composite": None,
            "id": None,
        }
        # Null placeholder for unused font image editors (m2-m5)
        null_img = {"background": None, "layers": [], "composite": None, "id": None}

        # AnyText2 text_prompt: text must be wrapped in literal double quotes
        # so that modify_prompt() regex can find it.
        quoted_text = f'"{target_text}"'

        logger.debug(
            "AnyText2 request: text=%r, size=%dx%d, color=%s, steps=%d",
            target_text, w, h, text_color, self.config.anytext2_ddim_steps,
        )

        timeout = self.config.server_timeout

        job = client.submit(
            img_prompt="Text with some background",
            text_prompt=quoted_text,
            sort_radio="↕",
            revise_pos=False,
            base_model_path="",
            lora_path_ratio="",
            f1="Mimic From Image(模仿图中字体)",
            f2="No Font(不指定字体)",
            f3="No Font(不指定字体)",
            f4="No Font(不指定字体)",
            f5="No Font(不指定字体)",
            m1=mimic_img,
            m2=null_img,
            m3=null_img,
            m4=null_img,
            m5=null_img,
            c1=text_color,
            c2="#000000",
            c3="#000000",
            c4="#000000",
            c5="#000000",
            show_debug=False,
            draw_img=null_img,
            ref_img=ref_img,
            ori_img=ori_img,
            img_count=self.config.anytext2_img_count,
            ddim_steps=self.config.anytext2_ddim_steps,
            w=w,
            h=h,
            strength=self.config.anytext2_strength,
            attnx_scale=1.0,
            font_hollow=True,
            cfg_scale=self.config.anytext2_cfg_scale,
            seed=-1,
            eta=0,
            a_prompt=(
                "best quality, extremely detailed, 4k, HD, "
                "super legible text, clear text edges, clear strokes, "
                "neat writing, no watermarks"
            ),
            n_prompt=(
                "low-res, bad anatomy, extra digit, fewer digits, cropped, "
                "worst quality, low quality, watermark, unreadable text, "
                "messy words, distorted text, disorganized writing, "
                "advertising picture"
            ),
            api_name="/process_1",
        )
        result = job.result(timeout=timeout)

        return self._parse_result(result)

    def _parse_result(self, result: tuple) -> np.ndarray:
        """Extract the first image from the Gradio gallery response.

        Args:
            result: ``(gallery_list, debug_markdown)`` from ``/process_1``.

        Returns:
            BGR numpy array of the edited image.
        """
        gallery, debug_info = result

        if not gallery:
            raise RuntimeError(
                f"AnyText2 returned empty gallery. Debug: {debug_info}"
            )

        # Gallery entry format varies by gradio_client version:
        #   - {"image": "/path/to/file.webp", "caption": ...}
        #   - {"image": {"path": "/path/..."}, "caption": ...}
        #   - {"path": "/path/..."}
        first_entry = gallery[0]
        if isinstance(first_entry, dict) and "image" in first_entry:
            img_val = first_entry["image"]
            image_path = img_val["path"] if isinstance(img_val, dict) else img_val
        elif isinstance(first_entry, dict) and "path" in first_entry:
            image_path = first_entry["path"]
        else:
            raise RuntimeError(
                f"Unexpected gallery format: {type(first_entry)}. "
                f"Debug: {debug_info}"
            )

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(
                f"Failed to read AnyText2 result image at {image_path}"
            )

        logger.debug("AnyText2 result: shape=%s", img.shape)
        return img

    @staticmethod
    def _prepare_roi(
        image: np.ndarray, min_gen_size: int = 512,
    ) -> tuple[np.ndarray, tuple[int, int, int, int], float]:
        """Upscale, 64-align, and pad an ROI for AnyText2.

        Steps:
            1. **Downscale** if ``max(h, w) > 1024`` (AnyText2 hard max).
            2. **Upscale** if ``max(h, w) < min_gen_size`` so the longest
               side reaches *min_gen_size* (quality floor, default 512).
            3. **Pad** both dimensions UP to the next multiple of 64 using
               ``BORDER_REPLICATE``.  This prevents AnyText2's server-side
               ``resize_image`` from cropping content pixels.

        Args:
            image: BGR input ROI (H × W × 3, uint8).
            min_gen_size: Target minimum for ``max(h, w)``.  Clamped to
                [``MIN_DIM``, ``MAX_DIM``] internally.

        Returns:
            ``(prepared_image, content_rect, scale)`` where *content_rect*
            is ``(top, bottom, left, right)`` — the slice coordinates of
            the original content within the padded image, and *scale* is
            the resize factor applied to the input.
        """
        min_gen_size = max(MIN_DIM, min(min_gen_size, MAX_DIM))
        h, w = image.shape[:2]

        # 1. Downscale if above hard max
        scale = 1.0
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)

        # 2. Upscale if below quality floor
        if max(h, w) * scale < min_gen_size:
            scale = min_gen_size / max(h, w)

        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        # Resize if scale changed
        if new_w != w or new_h != h:
            image = cv2.resize(
                image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4,
            )
            logger.debug(
                "Scaled ROI from %dx%d to %dx%d (scale=%.3f)",
                w, h, new_w, new_h, scale,
            )

        # 3. Pad to next multiple of ALIGN (64) on each axis
        def _pad_to_align(dim: int) -> int:
            remainder = dim % ALIGN
            return (ALIGN - remainder) if remainder else 0

        pad_w = _pad_to_align(new_w)
        pad_h = _pad_to_align(new_h)

        # Also ensure both axes meet MIN_DIM after alignment.
        # NOTE: This preserves 64-alignment because MIN_DIM (256) is
        # itself a multiple of ALIGN (64).  If either constant changes,
        # verify that MIN_DIM % ALIGN == 0 still holds.
        aligned_w = new_w + pad_w
        aligned_h = new_h + pad_h
        if aligned_w < MIN_DIM:
            pad_w += MIN_DIM - aligned_w
        if aligned_h < MIN_DIM:
            pad_h += MIN_DIM - aligned_h

        # Content rectangle within the padded image
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        if pad_w > 0 or pad_h > 0:
            image = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REPLICATE,
            )
            logger.debug(
                "Padded ROI by (top=%d, bottom=%d, left=%d, right=%d) "
                "to %dx%d (64-aligned)",
                pad_top, pad_bottom, pad_left, pad_right,
                image.shape[1], image.shape[0],
            )

        content_rect = (pad_top, pad_top + new_h, pad_left, pad_left + new_w)

        logger.debug(
            "Prepared ROI: original %dx%d → final %dx%d, content_rect=%s, scale=%.3f",
            w, h, image.shape[1], image.shape[0], content_rect, scale,
        )
        return image, content_rect, scale

    @staticmethod
    def _extract_text_color(image: np.ndarray) -> str:
        """Estimate dominant text color from the ROI as a hex string.

        Strategy: text is typically darker or lighter than the background.
        Sample border pixels (background) and interior pixels, then pick
        the cluster that differs most from the border median.
        """
        h, w = image.shape[:2]
        border_h = max(1, h // 8)
        border_w = max(1, w // 8)

        # Border pixels → background estimate
        border_pixels = np.concatenate([
            image[:border_h].reshape(-1, 3),
            image[-border_h:].reshape(-1, 3),
            image[border_h:-border_h, :border_w].reshape(-1, 3),
            image[border_h:-border_h, -border_w:].reshape(-1, 3),
        ])
        bg_median = np.median(border_pixels, axis=0)

        # Interior pixels
        interior = image[border_h:h - border_h, border_w:w - border_w]
        if interior.size == 0:
            return "#000000"

        interior_flat = interior.reshape(-1, 3).astype(np.float64)

        # Pixels that differ most from background are likely text
        diffs = np.linalg.norm(interior_flat - bg_median, axis=1)
        threshold = np.percentile(diffs, 75)
        text_mask = diffs >= threshold
        text_pixels = interior_flat[text_mask]

        if len(text_pixels) == 0:
            return "#000000"

        # Median of text pixels (BGR)
        text_color_bgr = np.median(text_pixels, axis=0).astype(int)
        b, g, r = text_color_bgr
        return f"#{r:02x}{g:02x}{b:02x}"
