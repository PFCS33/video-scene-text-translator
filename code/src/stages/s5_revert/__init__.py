"""Stage 5: Revert (de-frontalize + optional alignment refinement + composite).

Package layout mirrors s4_propagation/: ``stage.py`` holds the orchestrator
and ``refiner.py`` (Step 2.4) will host the ROI alignment refiner inference
wrapper.
"""

from .stage import RevertStage

__all__ = ["RevertStage"]
