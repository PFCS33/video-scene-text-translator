"""Core data structures for the video text replacement pipeline.

All dataclasses that flow between pipeline stages are defined here
to avoid circular imports and provide a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


@dataclass
class BBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x: int  # top-left x
    y: int  # top-left y
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    def to_slice(self) -> tuple[slice, slice]:
        """Return (row_slice, col_slice) for numpy array indexing."""
        return slice(self.y, self.y2), slice(self.x, self.x2)

    def area(self) -> int:
        return self.width * self.height


@dataclass
class Quad:
    """Four corner points defining a text region polygon.

    Points are ordered: top-left, top-right, bottom-right, bottom-left.
    Shape: (4, 2) as numpy array, each row is (x, y).
    """

    points: np.ndarray  # shape (4, 2), dtype float32

    def to_bbox(self) -> BBox:
        """Compute axis-aligned bounding box enclosing this quad."""
        x_min, y_min = self.points.min(axis=0).astype(int)
        x_max, y_max = self.points.max(axis=0).astype(int)
        return BBox(x=int(x_min), y=int(y_min),
                     width=int(x_max - x_min), height=int(y_max - y_min))

    def aspect_ratio(self) -> float:
        """Width/height ratio based on average edge lengths."""
        top_w = np.linalg.norm(self.points[1] - self.points[0])
        bot_w = np.linalg.norm(self.points[2] - self.points[3])
        left_h = np.linalg.norm(self.points[3] - self.points[0])
        right_h = np.linalg.norm(self.points[1] - self.points[2])
        avg_w = (top_w + bot_w) / 2
        avg_h = (left_h + right_h) / 2
        return float(avg_w / max(avg_h, 1e-6))


@dataclass
class TextDetection:
    """A single text detection in one frame."""

    frame_idx: int
    quad: Quad
    bbox: BBox
    text: str
    ocr_confidence: float  # 0.0 to 1.0
    sharpness_score: float = 0.0
    contrast_score: float = 0.0
    frontality_score: float = 0.0
    composite_score: float = 0.0


@dataclass
class TextTrack:
    """A tracked text region across multiple frames.

    Groups TextDetections that refer to the same physical text instance
    across the video. This is the central data structure that flows
    through all pipeline stages.
    """

    track_id: int
    source_text: str
    target_text: str
    source_lang: str
    target_lang: str
    detections: dict[int, TextDetection]  # frame_idx -> detection
    reference_frame_idx: int = -1
    reference_quad: Optional[Quad] = None
    edited_roi: Optional[np.ndarray] = None  # Result from Stage A (H x W x 3)


@dataclass
class FrameHomography:
    """Homography data for one frame relative to the reference frame."""

    frame_idx: int
    H_to_ref: Optional[np.ndarray] = None  # 3x3: frame -> reference
    H_from_ref: Optional[np.ndarray] = None  # 3x3: reference -> frame
    is_valid: bool = True


@dataclass
class PropagatedROI:
    """A translated ROI adapted for a specific frame."""

    frame_idx: int
    track_id: int
    roi_image: np.ndarray  # Color-adjusted translated ROI (H x W x 3)
    alpha_mask: np.ndarray  # Alpha mask for blending (H x W, float 0-1)
    target_quad: Quad  # Where to place it in the original frame


@dataclass
class PipelineResult:
    """Final output of the full pipeline."""

    tracks: list[TextTrack]
    output_frames: list[np.ndarray]
    fps: float
    frame_size: tuple[int, int]  # (width, height)
