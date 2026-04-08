"""Placeholder text editor for Stage B pipeline testing.

Renders target text using Pillow (PIL) for full Unicode support.
Produces a crude result but allows end-to-end pipeline testing
without a real Stage A model.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.models.base_text_editor import BaseTextEditor

# DejaVu Sans shipped with matplotlib supports Latin/accented characters.
_FONT_PATH = None
try:
    import matplotlib
    import os
    _candidate = os.path.join(
        os.path.dirname(matplotlib.__file__),
        "mpl-data", "fonts", "ttf", "DejaVuSans.ttf",
    )
    if os.path.isfile(_candidate):
        _FONT_PATH = _candidate
except ImportError:
    pass


class PlaceholderTextEditor(BaseTextEditor):
    """Renders target text onto the ROI using Pillow.

    Strategy:
    1. Estimate dominant background color from border pixels.
    2. Fill the ROI center with that background color.
    3. Render target text in a contrasting color, auto-scaled to fit.
    """

    def edit_text(
        self,
        roi_image: np.ndarray,
        target_text: str,
        edit_region: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        result = roi_image.copy()
        h, w = result.shape[:2]

        if h < 5 or w < 5:
            return result

        # When edit_region is given, render only within that sub-area
        if edit_region is not None:
            et, eb, el, er = edit_region
            sub = result[et:eb, el:er].copy()
            sub = self._render_text(sub, target_text)
            result[et:eb, el:er] = sub
            return result

        return self._render_text(result, target_text)

    def _render_text(self, result: np.ndarray, target_text: str) -> np.ndarray:
        """Render target_text onto the image with auto-scaled font."""
        h, w = result.shape[:2]

        if h < 5 or w < 5:
            return result

        # Estimate background from border pixels (top/bottom 10% rows)
        border_h = max(1, h // 10)
        border_pixels = np.concatenate([
            result[:border_h].reshape(-1, 3),
            result[-border_h:].reshape(-1, 3),
        ])
        bg_color = tuple(int(c) for c in np.median(border_pixels, axis=0))

        # Contrasting text color
        text_color = tuple(255 - c for c in bg_color)

        # Fill center with background
        margin_h = max(1, h // 8)
        margin_w = max(1, w // 8)
        result[margin_h:h - margin_h, margin_w:w - margin_w] = bg_color

        # Available area for text
        avail_w = w - 2 * margin_w
        avail_h = h - 2 * margin_h

        # Auto-scale font to fit within ROI
        font_size = 8
        best_font = self._load_font(font_size)
        for size in range(8, max(9, h * 2)):
            font = self._load_font(size)
            bbox = font.getbbox(target_text)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if tw > avail_w or th > avail_h:
                break
            best_font = font
            font_size = size

        # Convert BGR -> RGB for Pillow
        pil_img = Image.fromarray(result[:, :, ::-1])
        draw = ImageDraw.Draw(pil_img)

        bbox = best_font.getbbox(target_text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        text_x = (w - tw) // 2 - bbox[0]
        text_y = (h - th) // 2 - bbox[1]

        # PIL text color is RGB (result is BGR, but we converted)
        draw.text(
            (text_x, text_y), target_text,
            fill=(text_color[2], text_color[1], text_color[0]),
            font=best_font,
        )

        # Convert RGB -> BGR back to OpenCV format
        return np.array(pil_img)[:, :, ::-1].copy()

    @staticmethod
    def _load_font(size: int) -> ImageFont.FreeTypeFont:
        if _FONT_PATH:
            return ImageFont.truetype(_FONT_PATH, size)
        return ImageFont.load_default(size=size)
