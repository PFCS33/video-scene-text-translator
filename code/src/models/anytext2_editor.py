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
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.config import TextEditorConfig
from src.models.anytext2_mask import (
    compute_adaptive_mask_rect,
    restore_middle_strip,
)
from src.models.base_text_editor import BaseTextEditor

if TYPE_CHECKING:
    from src.stages.s4_propagation.base_inpainter import BaseBackgroundInpainter

logger = logging.getLogger(__name__)

# Feather width at the middle-strip boundary (pixels). Small fixed value —
# enough to hide the SRNet seam, small enough to not bleed into new text.
_ADAPTIVE_STRIP_FEATHER_PX = 3

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

    def __init__(
        self,
        config: TextEditorConfig,
        inpainter: BaseBackgroundInpainter | None = None,
    ):
        self.config = config
        self._client = None  # Lazy-init gradio Client
        # Optional inpainter for adaptive mask sizing. When present and
        # config.anytext2_adaptive_mask is True, long-to-short cases
        # pre-inpaint the canonical and shrink the mask. See plan.md.
        self._inpainter = inpainter
        self._warned_no_inpainter = False

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

        # Extract dominant text color from the ORIGINAL roi_image BEFORE
        # the adaptive mask flow rewrites any pixels — otherwise we'd
        # read the inpainted background color.
        if edit_region is not None:
            et, eb, el, er = edit_region
            color_region = roi_image[et:eb, el:er]
        else:
            color_region = roi_image
        text_color = self._extract_text_color(color_region)

        # Save pre-adaptive references for the font-mimic input. When the
        # adaptive flow fires, m1 should see the ORIGINAL source glyphs
        # (not the SRNet-rewritten hybrid) so AnyText2's font encoder
        # extracts the correct upright style.
        mimic_roi_image = roi_image
        mimic_edit_region = edit_region

        # Apply adaptive mask sizing if configured + inpainter available.
        # May rewrite roi_image and narrow edit_region to a shrunk, centered
        # strip inside the canonical text area. Silent no-op otherwise.
        roi_image, edit_region = self._apply_adaptive_mask(
            roi_image, target_text, edit_region,
        )
        adaptive_fired = roi_image is not mimic_roi_image

        # Upscale, 64-align, and pad — returns prepared image + content region
        min_gen = self.config.anytext2_min_gen_size
        roi_prepared, content_rect, scale = self._prepare_roi(roi_image, min_gen)
        h_send, w_send = roi_prepared.shape[:2]
        ct, cb, cl, cr = content_rect  # top, bottom, left, right

        def _rect_for_region(
            region: tuple[int, int, int, int] | None,
        ) -> tuple[int, int, int, int]:
            """Translate an ROI-space edit region into send-space coords."""
            if region is None:
                return content_rect
            r_t, r_b, r_l, r_r = region
            return (
                max(0, ct + int(round(r_t * scale))),
                min(h_send, ct + int(round(r_b * scale))),
                max(0, cl + int(round(r_l * scale))),
                min(w_send, cl + int(round(r_r * scale))),
            )

        # Main mask: the (possibly narrowed) region AnyText2 should edit.
        mask_rect = _rect_for_region(edit_region)
        mt, mb, ml, mr = mask_rect

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

            # When adaptive fired, give m1 (font-mimic) the pre-adaptive
            # ROI + original-extent mask so AnyText2's font encoder sees
            # complete source glyphs and extracts the correct style.
            # ref_img keeps the hybrid + narrow mask for the actual edit.
            mimic_ori_path: str | None = None
            mimic_mask_path: str | None = None
            if adaptive_fired:
                mimic_ori_path = str(Path(tmpdir) / "mimic_ori.png")
                mimic_mask_path = str(Path(tmpdir) / "mimic_mask.png")

                mimic_prepared, _, _ = self._prepare_roi(
                    mimic_roi_image, min_gen,
                )
                if not cv2.imwrite(mimic_ori_path, mimic_prepared):
                    raise RuntimeError(
                        f"Failed to write mimic ROI to {mimic_ori_path}"
                    )

                mmt, mmb, mml, mmr = _rect_for_region(mimic_edit_region)
                mimic_mask_arr = np.zeros((h_send, w_send, 4), dtype=np.uint8)
                mimic_mask_arr[mmt:mmb, mml:mmr, 3] = 255
                if not cv2.imwrite(mimic_mask_path, mimic_mask_arr):
                    raise RuntimeError(
                        f"Failed to write mimic mask to {mimic_mask_path}"
                    )

            result_image = self._call_server(
                ori_path, mask_path, target_text, text_color, w_send, h_send,
                mimic_ori_path=mimic_ori_path,
                mimic_mask_path=mimic_mask_path,
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

    def _apply_adaptive_mask(
        self,
        roi_image: np.ndarray,
        target_text: str,
        edit_region: tuple[int, int, int, int] | None,
    ) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
        """Shrink the mask + pre-inpaint source text for long-to-short cases.

        When the target text's natural aspect ratio is much narrower than
        the canonical text area, this:

        1. Extracts the canonical region (either the whole ROI or the inner
           sub-rect marked by *edit_region*)
        2. Calls the configured inpainter to erase source text
        3. Restores a centered middle strip of original pixels matching the
           target text's natural aspect ratio, so AnyText2 still sees a
           valid "text to replace" anchor inside the mask
        4. Pastes the hybrid canonical back into *roi_image* and narrows
           *edit_region* to the new (shrunk) mask rectangle

        Returns the (possibly modified) *roi_image* and *edit_region*. If
        the adaptive flow should be skipped (config off, no inpainter,
        within tolerance, target wider than source, inpainter failure),
        returns the inputs unchanged.
        """
        if not self.config.anytext2_adaptive_mask:
            return roi_image, edit_region

        if self._inpainter is None:
            if not self._warned_no_inpainter:
                logger.warning(
                    "AnyText2 adaptive_mask is enabled but no inpainter "
                    "was provided. Long-to-short translations may produce "
                    "gibberish fill. Configure propagation.inpainter_backend "
                    "to enable the adaptive flow."
                )
                self._warned_no_inpainter = True
            return roi_image, edit_region

        # Identify the canonical region within roi_image
        if edit_region is not None:
            et, eb, el, er = edit_region
            canonical = roi_image[et:eb, el:er]
        else:
            et, eb = 0, roi_image.shape[0]
            el, er = 0, roi_image.shape[1]
            canonical = roi_image

        canonical_h, canonical_w = canonical.shape[:2]
        if canonical_h <= 0 or canonical_w <= 0:
            return roi_image, edit_region

        # Pure-logic computation — may return None to signal skip
        adaptive_rect = compute_adaptive_mask_rect(
            canonical_w=canonical_w,
            canonical_h=canonical_h,
            target_text=target_text,
            tolerance=self.config.anytext2_mask_aspect_tolerance,
        )
        if adaptive_rect is None:
            return roi_image, edit_region

        # Inpaint the entire canonical via the configured backend
        try:
            clean_canonical = self._inpainter.inpaint(canonical)
        except Exception as exc:  # noqa: BLE001 - log + fall back
            logger.warning(
                "AnyText2 adaptive_mask: inpainter failed (%s); falling "
                "back to non-adaptive mask for this track.",
                exc,
            )
            return roi_image, edit_region

        # Restore the middle strip (original pixels under the new mask)
        hybrid_canonical = restore_middle_strip(
            inpainted=clean_canonical,
            original=canonical,
            mask_rect=adaptive_rect,
            feather_px=_ADAPTIVE_STRIP_FEATHER_PX,
        )

        # Paste hybrid back into a copy of roi_image (don't mutate caller)
        new_roi = roi_image.copy()
        new_roi[et:eb, el:er] = hybrid_canonical

        # Translate adaptive_rect from canonical coords to roi_image coords
        _, _, strip_l, strip_r = adaptive_rect
        new_edit_region = (et, eb, el + strip_l, el + strip_r)

        logger.debug(
            "AnyText2 adaptive mask triggered: canonical %dx%d → mask "
            "width %d (centered), target_text=%r",
            canonical_w, canonical_h, strip_r - strip_l, target_text,
        )

        return new_roi, new_edit_region

    def _call_server(
        self,
        ori_path: str,
        mask_path: str,
        target_text: str,
        text_color: str,
        w: int,
        h: int,
        mimic_ori_path: str | None = None,
        mimic_mask_path: str | None = None,
    ) -> np.ndarray:
        """Send an edit request to the AnyText2 Gradio server.

        When *mimic_ori_path* / *mimic_mask_path* are provided (adaptive
        mask path), ``m1`` uses a separate pre-adaptive ROI + wide mask
        so the font encoder sees complete source glyphs. ``ref_img`` and
        ``ori_img`` still use *ori_path* (the adaptive hybrid canvas).
        """
        from gradio_client import handle_file  # lazy import

        client = self._get_client()

        # ref_img: background is the (possibly adaptive-hybrid) image,
        # layers[0] is an RGBA image where the alpha channel marks the
        # edit region.
        ref_img = {
            "background": handle_file(ori_path),
            "layers": [handle_file(mask_path)],
            "composite": None,
            "id": None,
        }
        ori_img = handle_file(ori_path)

        # Null placeholder for unused font image editors (m2-m5)
        null_img = {"background": None, "layers": [], "composite": None, "id": None}

        # m1: ROI image for "Mimic From Image" font extraction. When the
        # adaptive path provides separate mimic files, use them so the
        # font encoder reads the full original source glyphs instead of
        # a narrow middle strip (which would cause italic slant).
        mimic_bg = mimic_ori_path or ori_path
        mimic_layer = mimic_mask_path or mask_path
        mimic_img = {
            "background": handle_file(mimic_bg),
            "layers": [handle_file(mimic_layer)],
            "composite": None,
            "id": None,
        }

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
