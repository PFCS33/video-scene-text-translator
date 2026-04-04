"""Pipeline configuration management.

Loads config from YAML, provides typed defaults, and validates parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DetectionConfig:
    ocr_backend: str = "easyocr"  # "easyocr" or "paddleocr"
    ocr_languages: list[str] = field(default_factory=lambda: ["en"])
    ocr_confidence_threshold: float = 0.3
    min_text_area: int = 100
    # Per-detection composite scoring weights (must sum to 1.0)
    weight_ocr_confidence: float = 0.3
    weight_sharpness: float = 0.3
    weight_contrast: float = 0.2
    weight_frontality: float = 0.2
    # Reference frame selection: hard pre-filters (STRIVE-aligned)
    ref_ocr_min_confidence: float = 0.7
    ref_sharpness_top_k: int = 10
    # Reference frame selection: 2-metric composite weights
    ref_weight_contrast: float = 0.7   # Otsu interclass variance
    ref_weight_frontality: float = 0.3  # bbox area ratio
    # Process every N-th frame for detection (1 = every frame)
    frame_sample_rate: int = 1
    # Optical flow for tracking quads between frames
    optical_flow_method: str = "farneback"  # "farneback", "lucas_kanade", or "cotracker"
    # "gaps_only": only fill frames missing OCR detections (original behavior)
    # "full_propagation": propagate reference quad to ALL frames, overwriting OCR quads
    flow_fill_strategy: str = "gaps_only"
    # CoTracker3 settings (only used when optical_flow_method == "cotracker")
    cotracker_checkpoint: str = "third_party/co-tracker/checkpoints/scaled_offline.pth"
    cotracker_online_checkpoint: str = "third_party/co-tracker/checkpoints/scaled_online.pth"
    cotracker_window_len: int = 60
    cotracker_online_window_len: int = 16
    farneback_pyr_scale: float = 0.5
    farneback_levels: int = 3
    farneback_winsize: int = 15
    farneback_iterations: int = 3
    farneback_poly_n: int = 5
    farneback_poly_sigma: float = 1.2
    lk_win_size: list[int] = field(default_factory=lambda: [21, 21])
    lk_max_level: int = 3
    # Optional word whitelist — if set, only keep detections whose words are all in this set
    word_whitelist: set[str] | None = None


@dataclass
class TranslationConfig:
    source_lang: str = "en"
    target_lang: str = "es"
    backend: str = "googletrans"  # "googletrans" or "google-cloud"
    api_key: str | None = None


@dataclass
class FrontalizationConfig:
    # Homography computation (frame quad → canonical frontal rectangle)
    homography_method: str = "RANSAC"
    ransac_reproj_threshold: float = 5.0


@dataclass
class PropagationConfig:
    color_space: str = "YCrCb"
    histogram_bins: int = 256
    clip_limit: float = 2.0
    blend_blur_kernel: int = 5


@dataclass
class RevertConfig:
    blend_border_size: int = 3
    blend_method: str = "gaussian"


@dataclass
class TextEditorConfig:
    backend: str = "placeholder"  # "placeholder" or "stage_a"
    model_path: str | None = None
    device: str = "cpu"


@dataclass
class PipelineConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    frontalization: FrontalizationConfig = field(default_factory=FrontalizationConfig)
    propagation: PropagationConfig = field(default_factory=PropagationConfig)
    revert: RevertConfig = field(default_factory=RevertConfig)
    text_editor: TextEditorConfig = field(default_factory=TextEditorConfig)
    # Global
    input_video: str = ""
    output_video: str = ""
    output_dir: str = "" # For TPM data gen: directory to save extracted ROIs and metadata instead of video output
    log_level: str = "INFO"
    debug_output_dir: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load config from YAML file, merging with defaults."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            detection=DetectionConfig(**raw.get("detection", {})),
            translation=TranslationConfig(**raw.get("translation", {})),
            frontalization=FrontalizationConfig(**raw.get("frontalization", {})),
            propagation=PropagationConfig(**raw.get("propagation", {})),
            revert=RevertConfig(**raw.get("revert", {})),
            text_editor=TextEditorConfig(**raw.get("text_editor", {})),
            input_video=raw.get("input_video", ""),
            output_video=raw.get("output_video", ""),
            output_dir=raw.get("output_dir", ""),
            log_level=raw.get("log_level", "INFO"),
            debug_output_dir=raw.get("debug_output_dir"),
        )

    def validate(self) -> list[str]:
        """Return list of validation errors. Empty list means valid."""
        errors = []
        if not self.input_video:
            errors.append("input_video is required")
        if not self.output_video and not self.output_dir:
            errors.append("output_video or output_dir is required")
        if not (0 <= self.detection.ocr_confidence_threshold <= 1):
            errors.append("ocr_confidence_threshold must be in [0, 1]")
        det_weights = [
            self.detection.weight_ocr_confidence,
            self.detection.weight_sharpness,
            self.detection.weight_contrast,
            self.detection.weight_frontality,
        ]
        if abs(sum(det_weights) - 1.0) > 0.01:
            errors.append(
                f"Detection scoring weights must sum to 1.0, got {sum(det_weights):.2f}"
            )
        ref_weights = [
            self.detection.ref_weight_contrast,
            self.detection.ref_weight_frontality,
        ]
        if abs(sum(ref_weights) - 1.0) > 0.01:
            errors.append(
                f"Reference selection weights must sum to 1.0, got {sum(ref_weights):.2f}"
            )
        if self.detection.ref_sharpness_top_k < 1:
            errors.append("ref_sharpness_top_k must be >= 1")
        if self.detection.frame_sample_rate < 1:
            errors.append("frame_sample_rate must be >= 1")
        return errors
