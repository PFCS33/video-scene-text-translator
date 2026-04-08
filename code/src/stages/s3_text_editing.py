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


def _clamp_expansion_ratio(ratio: float, w: int, h: int) -> float:
    """Cap expansion so expanded dimensions stay within AnyText2 limits.

    Returns 0.0 if no expansion is possible (ROI already at/above max).
    """
    from src.models.anytext2_editor import MAX_DIM

    if ratio <= 0:
        return 0.0
    max_dim = max(w, h)
    if max_dim >= MAX_DIM:
        return 0.0
    max_ratio = (MAX_DIM / max_dim - 1) / 2
    return min(ratio, max_ratio)


def _expanded_warp(
    frame: np.ndarray,
    H_to_frontal: np.ndarray,
    w: int,
    h: int,
    ratio: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Warp a larger-than-canonical region to include scene context.

    Shifts the homography destination by a margin so the canonical text
    area is centered in a larger output.  ``warpPerspective`` naturally
    fills the margins with real scene pixels from the source frame.

    Args:
        frame: Full video frame (BGR).
        H_to_frontal: 3x3 homography (frame → canonical).
        w, h: Canonical text dimensions.
        ratio: Expansion ratio per side (e.g. 0.3 = 30%).

    Returns:
        (expanded_roi, edit_region) where *edit_region* is
        ``(top, bottom, left, right)`` of the text area within the
        expanded ROI.
    """
    margin_x = int(round(w * ratio))
    margin_y = int(round(h * ratio))

    # Translation matrix: shift canonical space into center of expanded output
    T = np.array(
        [[1, 0, margin_x], [0, 1, margin_y], [0, 0, 1]],
        dtype=np.float64,
    )
    H_expanded = T @ H_to_frontal

    w_exp = w + 2 * margin_x
    h_exp = h + 2 * margin_y

    expanded_roi = cv2.warpPerspective(
        frame, H_expanded, (w_exp, h_exp),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    edit_region = (margin_y, margin_y + h, margin_x, margin_x + w)

    logger.debug(
        "Expanded warp: canonical %dx%d → expanded %dx%d (ratio=%.2f, "
        "margin=%dx%d)",
        w, h, w_exp, h_exp, ratio, margin_x, margin_y,
    )

    return expanded_roi, edit_region


class TextEditingStage:
    def __init__(self, config: PipelineConfig):
        self.config = config.text_editor
        self._editor: BaseTextEditor | None = None

    def _init_editor(self) -> BaseTextEditor:
        if self._editor is None:
            if self.config.backend == "placeholder":
                self._editor = PlaceholderTextEditor()
            elif self.config.backend == "anytext2":
                from src.models.anytext2_editor import AnyText2Editor
                self._editor = AnyText2Editor(self.config)
            elif self.config.backend == "stage_a":
                raise NotImplementedError(
                    "Stage A model integration not yet implemented. "
                    "Use 'placeholder' or 'anytext2' backend."
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

        expansion = self.config.roi_context_expansion

        for track in tracks:
            ref_idx = track.reference_frame_idx
            if ref_idx < 0 or ref_idx not in frames:
                logger.warning(
                    "S3: Track %d has no valid reference frame", track.track_id
                )
                continue

            ref_frame = frames[ref_idx]
            ref_det = track.detections[ref_idx]

            has_homography = (
                ref_det.H_to_frontal is not None
                and ref_det.homography_valid
                and track.canonical_size is not None
            )

            if has_homography:
                w, h = track.canonical_size
                effective_ratio = _clamp_expansion_ratio(expansion, w, h)

                if effective_ratio > 0:
                    roi, edit_region = _expanded_warp(
                        ref_frame, ref_det.H_to_frontal, w, h, effective_ratio,
                    )
                else:
                    roi = cv2.warpPerspective(
                        ref_frame, ref_det.H_to_frontal, (w, h),
                    )
                    edit_region = None
            else:
                if expansion > 0:
                    logger.debug(
                        "S3: Track %d has no homography, skipping ROI expansion",
                        track.track_id,
                    )
                roi = ref_frame[ref_det.bbox.to_slice()].copy()
                edit_region = None

            if roi.size == 0:
                logger.warning(
                    "S3: Track %d reference ROI is empty", track.track_id
                )
                continue

            edited_roi = editor.edit_text(
                roi, track.target_text, edit_region=edit_region,
            )

            # When expanded, crop back to the original canonical area
            if edit_region is not None:
                et, eb, el, er = edit_region
                edited_roi = edited_roi[et:eb, el:er]

            track.edited_roi = edited_roi
            logger.debug(
                "S3: Track %d: '%s' -> '%s' (ROI shape: %s, edit_region=%s)",
                track.track_id, track.source_text,
                track.target_text, edited_roi.shape, edit_region,
            )

        return tracks
