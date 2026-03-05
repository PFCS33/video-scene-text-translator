"""Stage 1: Detection & Selection.

Detects text ROIs in video frames, performs OCR, translates text,
groups detections into tracks, and selects reference frames.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from src.config import PipelineConfig
from src.data_types import BBox, Quad, TextDetection, TextTrack
from src.utils.geometry import quad_frontality_score
from src.utils.image_processing import compute_contrast, compute_sharpness

logger = logging.getLogger(__name__)


class DetectionStage:
    def __init__(self, config: PipelineConfig):
        self.config = config.detection
        self.translation_config = config.translation
        self._reader = None  # Lazy-init EasyOCR
        self._translator = None  # Lazy-init translator

    def _init_ocr(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(
                self.config.ocr_languages, gpu=False
            )

    def _init_translator(self):
        if self._translator is None:
            if self.translation_config.backend == "googletrans":
                from googletrans import Translator
                self._translator = Translator()
            else:
                from google.cloud import translate_v2 as translate
                self._translator = translate.Client()

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
                contrast_score=compute_contrast(roi),
                frontality_score=quad_frontality_score(quad),
            )
            detection.composite_score = self._compute_composite_score(detection)
            detections.append(detection)

        return detections

    def _compute_composite_score(self, det: TextDetection) -> float:
        """Weighted combination of quality metrics for reference selection."""
        return (
            self.config.weight_ocr_confidence * det.ocr_confidence
            + self.config.weight_sharpness * det.sharpness_score
            + self.config.weight_contrast * det.contrast_score
            + self.config.weight_frontality * det.frontality_score
        )

    def translate_text(self, text: str) -> str:
        """Translate text from source_lang to target_lang."""
        self._init_translator()
        if self.translation_config.backend == "googletrans":
            result = self._translator.translate(
                text,
                src=self.translation_config.source_lang,
                dest=self.translation_config.target_lang,
            )
            return result.text
        else:
            result = self._translator.translate(
                text,
                source_language=self.translation_config.source_lang,
                target_language=self.translation_config.target_lang,
            )
            return result["translatedText"]

    def group_detections_into_tracks(
        self, all_detections: dict[int, list[TextDetection]]
    ) -> list[TextTrack]:
        """Group detections across frames into tracks by spatial proximity.

        Uses IoU of bounding boxes between consecutive frames
        with greedy matching.
        """
        tracks: list[TextTrack] = []
        next_track_id = 0
        active: dict[int, TextDetection] = {}  # track_id -> last detection

        for frame_idx in sorted(all_detections.keys()):
            dets = all_detections[frame_idx]
            matched_det_idxs: set[int] = set()
            matched_track_ids: set[int] = set()

            # Match detections to existing tracks
            for det_i, det in enumerate(dets):
                best_iou = 0.3  # Minimum IoU threshold
                best_track_id = None
                for track_id, last_det in active.items():
                    if track_id in matched_track_ids:
                        continue
                    iou = _bbox_iou(det.bbox, last_det.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_track_id = track_id

                if best_track_id is not None:
                    for track in tracks:
                        if track.track_id == best_track_id:
                            track.detections[frame_idx] = det
                            break
                    active[best_track_id] = det
                    matched_det_idxs.add(det_i)
                    matched_track_ids.add(best_track_id)

            # Create new tracks for unmatched detections
            for det_i, det in enumerate(dets):
                if det_i in matched_det_idxs:
                    continue
                translated = self.translate_text(det.text)
                track = TextTrack(
                    track_id=next_track_id,
                    source_text=det.text,
                    target_text=translated,
                    source_lang=self.translation_config.source_lang,
                    target_lang=self.translation_config.target_lang,
                    detections={frame_idx: det},
                )
                tracks.append(track)
                active[next_track_id] = det
                next_track_id += 1

        return tracks

    def select_reference_frames(
        self, tracks: list[TextTrack]
    ) -> list[TextTrack]:
        """For each track, pick the frame with the highest composite score."""
        for track in tracks:
            if not track.detections:
                continue
            best_idx = max(
                track.detections.keys(),
                key=lambda idx: track.detections[idx].composite_score,
            )
            track.reference_frame_idx = best_idx
            track.reference_quad = track.detections[best_idx].quad
        return tracks

    def run(
        self, frames: list[tuple[int, np.ndarray]]
    ) -> list[TextTrack]:
        """Full S1: detect -> group -> translate -> select reference."""
        logger.info("S1: Starting detection on %d frames", len(frames))
        sample_rate = self.config.frame_sample_rate
        all_detections: dict[int, list[TextDetection]] = {}

        for frame_idx, frame in frames:
            if frame_idx % sample_rate != 0:
                continue
            dets = self.detect_text_in_frame(frame, frame_idx)
            if dets:
                all_detections[frame_idx] = dets
            logger.debug("S1: Frame %d -> %d detections", frame_idx, len(dets))

        tracks = self.group_detections_into_tracks(all_detections)
        tracks = self.select_reference_frames(tracks)
        logger.info("S1: Found %d text tracks", len(tracks))
        return tracks


def _bbox_iou(a: BBox, b: BBox) -> float:
    """Compute IoU (Intersection over Union) between two bounding boxes."""
    x_overlap = max(0, min(a.x2, b.x2) - max(a.x, b.x))
    y_overlap = max(0, min(a.y2, b.y2) - max(a.y, b.y))
    intersection = x_overlap * y_overlap
    union = a.area() + b.area() - intersection
    return intersection / max(union, 1e-6)
