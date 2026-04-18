"""Reference frame selection and text translation."""

from __future__ import annotations

import logging

from src.config import DetectionConfig, TranslationConfig
from src.data_types import TextTrack

logger = logging.getLogger(__name__)


class ReferenceSelector:
    """Selects reference frames and handles text translation."""

    def __init__(
        self,
        detection_config: DetectionConfig,
        translation_config: TranslationConfig,
    ):
        self.config = detection_config
        self.translation_config = translation_config
        self._translator = None  # Lazy-init translator

    def _init_translator(self):
        if self._translator is None and self.translation_config.backend == "google-cloud":
            from google.cloud import translate_v2 as translate
            self._translator = translate.Client()

    def translate_text(self, text: str) -> str:
        """Translate text from source_lang to target_lang."""
        if not text or not text.strip():
            return text

        self._init_translator()
        src = self.translation_config.source_lang
        tgt = self.translation_config.target_lang
        try:
            if self.translation_config.backend == "deep-translator":
                return self._translate_deep(text, src, tgt)

            result = self._translator.translate(
                text,
                source_language=src,
                target_language=tgt,
            )
            return result["translatedText"]
        except Exception as exc:
            logger.warning(
                "Translation failed for text '%s' with backend '%s': %s",
                text,
                self.translation_config.backend,
                exc,
            )
            return text

    # MyMemoryTranslator requires full locale codes (e.g. "en-GB", "es-ES")
    # while GoogleTranslator accepts the short forms we use everywhere else.
    # Map short → locale on fallback so the two backends stay interchangeable.
    # Entries are short → MyMemory locale code EXCEPT "zh-CN" which is already
    # the canonical MyMemory locale (not a short form). New Chinese variants
    # (zh-TW, zh-HK, …) need an explicit entry here — they won't auto-expand.
    _MYMEMORY_LOCALE: dict[str, str] = {
        "en": "en-GB",
        "es": "es-ES",
        "zh-CN": "zh-CN",
        "fr": "fr-FR",
        "de": "de-DE",
        "ja": "ja-JP",
        "ko": "ko-KR",
    }

    def _translate_deep(self, text: str, src: str, tgt: str) -> str:
        """Translate using deep-translator with MyMemory fallback."""
        from deep_translator import GoogleTranslator, MyMemoryTranslator

        try:
            return GoogleTranslator(source=src, target=tgt).translate(text)
        except Exception as exc:
            logger.debug(
                "GoogleTranslator failed for '%s', trying MyMemory: %s",
                text,
                exc,
            )
            my_src = self._MYMEMORY_LOCALE.get(src, src)
            my_tgt = self._MYMEMORY_LOCALE.get(tgt, tgt)
            return MyMemoryTranslator(source=my_src, target=my_tgt).translate(text)

    def select_reference_frames(
        self,
        tracks: list[TextTrack],
        max_frame_offset: int | None = None,
    ) -> list[TextTrack]:
        """Select reference frame per track using STRIVE-aligned criteria.

        Pipeline (hard pre-filters, then 2-metric composite):
        0. Filter: Detections must have the longest text in the track
        1. Filter: OCR confidence >= ref_ocr_min_confidence
        2. Filter: Keep top-K by sharpness (ref_sharpness_top_k)
        3. Score: 0.7 * contrast (Otsu) + 0.3 * frontality (bbox area ratio)
        4. Select frame with highest composite score

        If all candidates are filtered out, falls back to highest
        composite_score among all detections.

        Args:
            tracks: list of TextTrack to process.
            max_frame_offset: if set, only consider detections within the
                first ``max_frame_offset`` frames of each track. Useful for
                CoTracker online mode where the reference must fall inside the
                first sliding window.
        """
        for track in tracks:
            if not track.detections:
                continue

            candidates = list(track.detections.items())

            # Optional: restrict to first N frames of the track
            if max_frame_offset is not None and candidates:
                track_start = min(idx for idx, _ in candidates)
                candidates = [
                    (idx, det) for idx, det in candidates
                    if idx < track_start + max_frame_offset
                ]
                if not candidates:
                    # Fallback: use all detections if none in the window
                    candidates = list(track.detections.items())

            # Hard filter 0: keep only detections with the longest text in the track
            max_len = max(len(det.text) for _, det in candidates)
            candidates = [
                (idx, det) for idx, det in candidates
                if len(det.text) == max_len
            ]

            # Hard filter 1: OCR confidence
            ocr_min = self.config.ref_ocr_min_confidence
            filtered = [
                (idx, det) for idx, det in candidates
                if det.ocr_confidence >= ocr_min
            ]

            # Hard filter 2: Top-K sharpness
            if filtered:
                top_k = self.config.ref_sharpness_top_k
                if len(filtered) > top_k:
                    filtered.sort(key=lambda x: x[1].sharpness_score, reverse=True)
                    filtered = filtered[:top_k]

            # Fallback: if all candidates were filtered out, use all detections
            if not filtered:
                logger.debug(
                    "Track %d: no candidates passed pre-filters, falling back",
                    track.track_id,
                )
                filtered = candidates

            # 2-metric composite: 0.7 contrast + 0.3 frontality
            w_contrast = self.config.ref_weight_contrast
            w_frontality = self.config.ref_weight_frontality
            best_idx = max(
                filtered,
                key=lambda x: (
                    w_contrast * x[1].contrast_score
                    + w_frontality * x[1].frontality_score
                ),
            )[0]

            track.reference_frame_idx = best_idx
        return tracks
