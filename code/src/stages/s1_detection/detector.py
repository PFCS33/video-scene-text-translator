"""Text detection via EasyOCR with quality scoring."""

from __future__ import annotations

import numpy as np

from src.config import DetectionConfig
from src.data_types import Quad, TextDetection
from src.utils.geometry import quad_bbox_area_ratio
from src.utils.image_processing import (
    compute_contrast_otsu,
    compute_sharpness,
)


class TextDetector:
    """Detects text in frames using EasyOCR and computes quality scores."""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self._reader = None  # Lazy-init EasyOCR

    def _init_ocr(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(
                self.config.ocr_languages, gpu=True
            )

    def detect_text_in_frame(
        self, frame: np.ndarray, frame_idx: int
    ) -> list[TextDetection]:
        """Detect all text regions in a single frame via EasyOCR."""
        self._init_ocr()
        results = self._reader.readtext(frame)

        detections = []
        for bbox_points, text, confidence in results:
            if confidence < self.config.ocr_confidence_threshold:
                continue

            quad = Quad(points=np.array(bbox_points, dtype=np.float32))
            bbox = quad.to_bbox()

            if bbox.area() < self.config.min_text_area:
                continue

            roi = frame[bbox.to_slice()]
            if roi.size == 0:
                continue

            detection = TextDetection(
                frame_idx=frame_idx,
                quad=quad,
                bbox=bbox,
                text=text.strip(),
                ocr_confidence=confidence,
                sharpness_score=compute_sharpness(roi),
                contrast_score=compute_contrast_otsu(roi),
                frontality_score=quad_bbox_area_ratio(quad),
            )
            detection.composite_score = self.compute_composite_score(detection)
            detections.append(detection)

        return detections

    def compute_composite_score(self, det: TextDetection) -> float:
        """Weighted combination of quality metrics for reference selection."""
        return (
            self.config.weight_ocr_confidence * det.ocr_confidence
            + self.config.weight_sharpness * det.sharpness_score
            + self.config.weight_contrast * det.contrast_score
            + self.config.weight_frontality * det.frontality_score
        )
