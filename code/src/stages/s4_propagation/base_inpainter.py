"""Abstract base class for background inpainting models.

Any inpainter that produces a text-removed background ROI in canonical
frontal space must subclass BaseBackgroundInpainter. The s4 propagation
stage and the LCM consume the output via TextDetection.inpainted_background.

Multiple backends can plug in here (SRNet's B sub-network, LaMa, MAT, etc.)
without changing the rest of the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseBackgroundInpainter(ABC):
    """Abstract interface for text-removal/background inpainting models."""

    @abstractmethod
    def inpaint(self, canonical_roi: np.ndarray) -> np.ndarray:
        """Remove text from a canonical-frontal ROI and return its background.

        Args:
            canonical_roi: BGR image of the frontalized text region
                (H x W x 3, uint8). Already warped to canonical space by S2.

        Returns:
            BGR image of the same shape as `canonical_roi` (H x W x 3, uint8)
            with the text strokes erased and replaced by plausible
            background texture.
        """
        ...

    def load_model(self, model_path: str, device: str = "cpu") -> None:  # noqa: B027
        """Optionally load model weights. No-op default for stub backends."""
        pass
