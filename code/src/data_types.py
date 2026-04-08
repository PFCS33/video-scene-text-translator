"""Core data structures for the video text replacement pipeline.

All dataclasses that flow between pipeline stages are defined here
to avoid circular imports and provide a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    """A single text detection in one frame.

    Geometry fields (quad, bbox) are set by S1 detection/tracking.
    Homography fields (H_to_frontal, H_from_frontal) are set by S2.
    """

    frame_idx: int
    quad: Quad
    bbox: BBox
    text: str
    ocr_confidence: float  # 0.0 to 1.0
    sharpness_score: float = 0.0
    contrast_score: float = 0.0
    frontality_score: float = 0.0
    composite_score: float = 0.0
    # Homography fields — populated by S2 frontalization
    H_to_frontal: np.ndarray | None = None  # 3x3: frame → canonical frontal
    H_from_frontal: np.ndarray | None = None  # 3x3: canonical frontal → frame
    homography_valid: bool = False
    # Inpainted background ROI in canonical frontal space (text removed).
    # Populated by an inpainting step before S4; consumed by S4's LCM.
    inpainted_background: np.ndarray | None = None  # H x W x 3, uint8


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
    detections: dict[int, TextDetection] = field(default_factory=dict)
    reference_frame_idx: int = -1
    canonical_size: tuple[int, int] | None = None  # (width, height) of canonical frontal rect
    edited_roi: np.ndarray | None = None  # Result from Stage A (H x W x 3)

    @property
    def reference_quad(self) -> Quad | None:
        """Quad from the reference frame's detection."""
        det = self.detections.get(self.reference_frame_idx)
        return det.quad if det else None
    
    def to_json_serializable(self) -> dict:
        """Convert to a JSON-serializable dict (e.g. for logging or output)."""
        return {
            "track_id": self.track_id,
            "source_text": self.source_text,
            "target_text": self.target_text,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "reference_frame_idx": self.reference_frame_idx,
            "canonical_size": self.canonical_size,
            "detections": {
                idx: {
                    "frame_idx": det.frame_idx,
                    "quad": det.quad.points.tolist(),
                    "bbox": {
                        "x": det.bbox.x,
                        "y": det.bbox.y,
                        "width": det.bbox.width,
                        "height": det.bbox.height,
                    },
                    "text": det.text,
                    "ocr_confidence": det.ocr_confidence,
                    "sharpness_score": det.sharpness_score,
                    "contrast_score": det.contrast_score,
                    "frontality_score": det.frontality_score,
                    "composite_score": det.composite_score,
                    # Homography fields are not included in JSON output for brevity
                }
                for idx, det in self.detections.items()
            },
        }
    
    @classmethod
    def from_json_serializable(cls, data: dict) -> TextTrack:
        """Create a TextTrack from a JSON-deserialized dict."""
        detections = {
            int(idx): TextDetection(
                frame_idx=det_data["frame_idx"],
                quad=Quad(points=np.array(det_data["quad"], dtype=np.float32)),
                bbox=BBox(
                    x=det_data["bbox"]["x"],
                    y=det_data["bbox"]["y"],
                    width=det_data["bbox"]["width"],
                    height=det_data["bbox"]["height"],
                ),
                text=det_data["text"],
                ocr_confidence=det_data["ocr_confidence"],
                sharpness_score=det_data.get("sharpness_score", 0.0),
                contrast_score=det_data.get("contrast_score", 0.0),
                frontality_score=det_data.get("frontality_score", 0.0),
                composite_score=det_data.get("composite_score", 0.0),
            )
            for idx, det_data in data["detections"].items()
        }
        return cls(
            track_id=data["track_id"],
            source_text=data["source_text"],
            target_text=data["target_text"],
            source_lang=data["source_lang"],
            target_lang=data["target_lang"],
            reference_frame_idx=data["reference_frame_idx"],
            canonical_size=tuple(data["canonical_size"]) if data.get("canonical_size") else None,
            detections=detections,
        )


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
