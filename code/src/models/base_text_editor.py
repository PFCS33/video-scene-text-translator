"""Abstract base class for scene text editing models (Stage A interface).

Any Stage A implementation (RS-STE, AnyText2, etc.) must subclass
BaseTextEditor and implement edit_text(). This decouples the pipeline
from any specific model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseTextEditor(ABC):
    """Abstract interface for scene text editing models."""

    @abstractmethod
    def edit_text(
        self,
        roi_image: np.ndarray,
        target_text: str,
        edit_region: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """Replace text in an ROI image with target_text.

        Args:
            roi_image: BGR image of the text region (H x W x 3, uint8).
            target_text: The translated text to render.
            edit_region: Optional (top, bottom, left, right) pixel coords
                within *roi_image* marking the area to edit. If None,
                the entire image is the edit target.  Used when the ROI
                has been expanded with surrounding scene context.

        Returns:
            Edited BGR image of the same shape as roi_image (H x W x 3, uint8).
            The returned image should preserve the original scene style
            (font, color, background) while displaying target_text.
        """
        ...

    def load_model(self, model_path: str, device: str = "cpu") -> None:  # noqa: B027
        """Optionally load model weights. Not needed for placeholder."""
        pass
