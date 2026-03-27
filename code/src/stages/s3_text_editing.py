"""Stage 3: Cross-Language Text Editing.

Wraps the Stage A text editing model. Extracts the reference frame ROI
and passes it through the editor to produce a translated ROI.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.config import PipelineConfig
from src.data_types import TextTrack
from src.models.base_text_editor import BaseTextEditor
from src.models.placeholder_editor import PlaceholderTextEditor

logger = logging.getLogger(__name__)


class TextEditingStage:
    def __init__(self, config: PipelineConfig):
        self.config = config.text_editor
        self._editor: BaseTextEditor | None = None

    def _init_editor(self) -> BaseTextEditor:
        if self._editor is None:
            if self.config.backend == "placeholder":
                self._editor = PlaceholderTextEditor()
            elif self.config.backend == "stage_a":
                raise NotImplementedError(
                    "Stage A model integration not yet implemented. "
                    "Use 'placeholder' backend for Stage B testing."
                )
            else:
                raise ValueError(
                    f"Unknown text editor backend: {self.config.backend}"
                )
        return self._editor

    def run(
        self,
        tracks: list[TextTrack],
        frames: dict[int, np.ndarray],
    ) -> list[TextTrack]:
        """Edit text in each track's reference frame ROI.

        Extracts the ROI, passes it through the text editor,
        and stores the result in track.edited_roi.
        """
        editor = self._init_editor()
        logger.info(
            "S3: Editing text for %d tracks using '%s' backend",
            len(tracks), self.config.backend,
        )

        for track in tracks:
            ref_idx = track.reference_frame_idx
            if ref_idx < 0 or ref_idx not in frames:
                logger.warning(
                    "S3: Track %d has no valid reference frame", track.track_id
                )
                continue

            ref_frame = frames[ref_idx]
            ref_det = track.detections[ref_idx]

            if (ref_det.H_to_frontal is not None
                    and ref_det.homography_valid
                    and track.canonical_size is not None):
                w, h = track.canonical_size
                roi = cv2.warpPerspective(ref_frame, ref_det.H_to_frontal, (w, h))
            else:
                roi = ref_frame[ref_det.bbox.to_slice()].copy()

            if roi.size == 0:
                logger.warning(
                    "S3: Track %d reference ROI is empty", track.track_id
                )
                continue

            edited_roi = editor.edit_text(roi, track.target_text)
            track.edited_roi = edited_roi
            logger.debug(
                "S3: Track %d: '%s' -> '%s' (ROI shape: %s)",
                track.track_id, track.source_text,
                track.target_text, edited_roi.shape,
            )

        return tracks
