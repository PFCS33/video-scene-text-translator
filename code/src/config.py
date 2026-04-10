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
    # track_break_threshold: maximum number of frames to allow between detections in the same track
    track_break_threshold: int = 5
    # Optical flow for tracking quads between frames
    optical_flow_method: str = "farneback"  # "farneback", "lucas_kanade", or "cotracker"
    # "gaps_only": only fill frames missing OCR detections (original behavior)
    # "full_propagation": propagate reference quad to ALL frames, overwriting OCR quads
    flow_fill_strategy: str = "gaps_only"
    # CoTracker3 settings (only used when optical_flow_method == "cotracker")
    cotracker_checkpoint: str = "../third_party/co-tracker/checkpoints/scaled_offline.pth"
    cotracker_online_checkpoint: str = "../third_party/co-tracker/checkpoints/scaled_online.pth"
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
    backend: str = "deep-translator"  # "deep-translator" or "google-cloud"
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

    # Lighting Correction Module (TPM/LCM) — applied per-frame when each
    # detection has an inpainted_background populated. Falls back to the
    # legacy histogram-matching path when backgrounds are missing.
    use_lcm: bool = False
    lcm_eps: float = 1e-3
    lcm_ratio_clip_min: float = 0.5
    lcm_ratio_clip_max: float = 2.0
    lcm_ratio_blur_ksize: int = 9
    lcm_use_log_domain: bool = True
    lcm_temporal_alpha: float = 1.0  # 1.0 = no temporal smoothing
    lcm_neighbor_self_weight: float = 2.0

    # Background inpainter for LCM. Only loaded when use_lcm=True.
    # Backends: "srnet" (lksshw/SRNet wrapper) or "none".
    inpainter_backend: str = "none"
    inpainter_checkpoint_path: str | None = None
    inpainter_device: str = "cuda"

    # Blur Prediction Network (TPM/BPN). Applies a per-frame differential
    # blur to the LCM-corrected ROI to match each frame's blur level.
    # Only invoked when both use_lcm and use_bpn are True (paper order:
    # LCM then BPN). bpn_image_size must match the resolution the
    # checkpoint was trained at.
    use_bpn: bool = False
    bpn_checkpoint_path: str | None = None
    bpn_device: str = "cuda"
    bpn_n_neighbors: int = 3
    bpn_image_size: tuple[int, int] = (64, 128)  # (H, W) at training time
    bpn_kernel_size: int = 41


@dataclass
class RevertConfig:
    blend_border_size: int = 3
    blend_method: str = "gaussian"


@dataclass
class TextEditorConfig:
    backend: str = "placeholder"  # "placeholder", "anytext2", or "stage_a"
    model_path: str | None = None
    device: str = "cpu"
    # AnyText2 Gradio server settings (only used when backend == "anytext2")
    server_url: str | None = None  # e.g. "http://localhost:45843/"
    server_timeout: int = 120  # seconds to wait for Gradio response
    anytext2_ddim_steps: int = 20
    anytext2_cfg_scale: float = 7.5
    anytext2_strength: float = 1.0
    anytext2_img_count: int = 1  # number of result images (1 = fastest)
    # Minimum generation size: upscale small ROIs so max(h,w) >= this value.
    # AnyText2 was trained at 512×512 — smaller inputs degrade quality.
    # Range: 256–1024. Both dimensions are also padded to multiples of 64.
    anytext2_min_gen_size: int = 512
    # ROI context expansion: fraction of each dimension to add as margin
    # around the text region, filled with real scene pixels from the frame.
    # Gives AnyText2 visual context for better style matching.
    # 0.0 = no expansion (current behavior), 0.3 = 30% margin on each side.
    roi_context_expansion: float = 0.0
    # Adaptive mask sizing: when target text is much narrower than source
    # canonical (e.g. 7 CJK chars → 3 CJK chars), shrink the AnyText2 mask
    # to match target aspect and pre-inpaint source text outside the new
    # mask. Avoids gibberish-fill characters. Requires an inpainter backend
    # configured under `propagation.inpainter_backend` to take effect;
    # otherwise the adaptive flow is silently skipped.
    anytext2_adaptive_mask: bool = True
    # Skip adaptive flow if |target_aspect - source_aspect| / source_aspect
    # is below this fraction. Common translation cases with similar visual
    # width (DANGER→PELIGRO, STOP→ALTO) bypass the inpaint + shrink path.
    anytext2_mask_aspect_tolerance: float = 0.15


@dataclass
class TPMDataGenConfig:
    # For generating data for text propagation model training instead of video output
    save_detected_tracks: bool = True  # Whether to save S1 tracks to JSON for reuse
    load_detected_tracks: bool = False  # Whether to load S1 tracks from JSON instead of re-running detection
    max_frames_per_track: int = 20
    frame_sample_rate: int = 1  # Sample every N-th frame from each track


@dataclass
class PipelineConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    frontalization: FrontalizationConfig = field(default_factory=FrontalizationConfig)
    propagation: PropagationConfig = field(default_factory=PropagationConfig)
    revert: RevertConfig = field(default_factory=RevertConfig)
    text_editor: TextEditorConfig = field(default_factory=TextEditorConfig)
    tpm_data_gen: TPMDataGenConfig = field(default_factory=TPMDataGenConfig)
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
            tpm_data_gen=TPMDataGenConfig(**raw.get("tpm_data_gen", {})),
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
        if self.text_editor.backend == "anytext2" and not self.text_editor.server_url:
            errors.append(
                "text_editor.server_url is required when backend is 'anytext2'"
            )
        return errors
