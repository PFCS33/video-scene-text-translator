"""Placeholder text editor for Stage B pipeline testing.

Renders target text using OpenCV putText. Produces a crude result
but allows end-to-end pipeline testing without a real Stage A model.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.models.base_text_editor import BaseTextEditor


class PlaceholderTextEditor(BaseTextEditor):
    """Renders target text onto the ROI using OpenCV.

    Strategy:
    1. Estimate dominant background color from border pixels.
    2. Fill the ROI center with that background color.
    3. Render target text in a contrasting color, auto-scaled to fit.
    """

    def edit_text(self, roi_image: np.ndarray, target_text: str) -> np.ndarray:
        result = roi_image.copy()
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

        # Auto-scale font to fit within ROI
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = max(1, h // 30)
        font_scale = 0.3
        for scale in np.arange(0.3, 5.0, 0.1):
            (tw, th), _ = cv2.getTextSize(target_text, font, scale, thickness)
            if tw > (w - 2 * margin_w) or th > (h - 2 * margin_h):
                font_scale = max(0.3, scale - 0.1)
                break
            font_scale = scale

        (tw, th), baseline = cv2.getTextSize(
            target_text, font, font_scale, thickness
        )
        text_x = (w - tw) // 2
        text_y = (h + th) // 2

        cv2.putText(
            result, target_text, (text_x, text_y), font,
            font_scale, text_color, thickness, cv2.LINE_AA,
        )
        return result
