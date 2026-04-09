"""ROI Alignment Refiner.

Predicts a residual homography (4-corner offset parameterization) between two
almost-aligned canonical-frontal text ROIs. Used by S5 to correct CoTracker
residual tracking error before compositing the edited ROI back into the frame.

See plan.md Part 1 for the full design.
"""
