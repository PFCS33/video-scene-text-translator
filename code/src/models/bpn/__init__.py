"""Background Prediction Network (BPN) from STRIVE (arXiv:2109.02762).

Predicts differential blur parameters to match blur characteristics
between a reference frame ROI and target frame ROIs.
"""

from .model import BPN
from .blur import DifferentiableBlur

__all__ = ["BPN", "DifferentiableBlur"]
