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
    track_break_threshold: int = 30
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
    # S1 quad smoothing filters applied during gap-filling. These operate
    # on the raw optical-flow quads before S2 frontalization. Stacking
    # multiple filters introduces positional lag — disable if S5 temporal
    # smoothing is used instead.
    use_kalman_smoothing: bool = False
    use_ema_smoothing: bool = False
    ema_alpha: float = 0.6  # EMA weight on previous state (higher = more lag)


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

    # Attach each detection's canonical-frontal frame ROI to its PropagatedROI.
    # Needed by the S5 alignment refiner so it can predict ΔH against the
    # reference canonical ROI. Adds memory per detection (~track.canonical_size
    # * 3 bytes per detection), so default off.
    save_target_canonical_roi: bool = False


@dataclass
class RevertConfig:
    blend_border_size: int = 3
    blend_method: str = "gaussian"

    # S5 alignment refiner. See code/src/models/refiner/README.md for the
    # network design and code/src/stages/s5_revert/refiner.py for the
    # inference wrapper. When enabled, predicts a residual homography
    # (ΔH) between the reference and target canonical ROIs and composes
    # it into the warp chain to correct residual CoTracker tracking
    # error. Requires propagation.save_target_canonical_roi=True so S4
    # populates PropagatedROI.target_roi_canonical.
    use_refiner: bool = False
    refiner_checkpoint_path: str = "checkpoints/refiner/refiner_v0.pt"
    refiner_device: str = "cuda"
    refiner_image_size: tuple[int, int] = (64, 128)  # (H, W) at network input
    refiner_max_corner_offset_px: float = 16.0
    # If more than this fraction of refiner predictions per video are
    # rejected by the sanity checks, escalate the log line from DEBUG
    # to INFO so we notice without spamming the logs on clean runs.
    refiner_rejection_warn_threshold: float = 0.1

    # Do-no-harm gate (Tier 1 of refiner improvements). After the model
    # produces a sane ΔH, score the alignment under identity vs ΔH using
    # masked NCC on luminance. Only apply ΔH if it strictly improves the
    # score by `refiner_score_margin`. Catches the failure mode where the
    # network over-corrects on already-aligned pairs and adds visible
    # jitter. Set use_refiner_gate=False to disable the gate entirely
    # (legacy behavior — the model's own ΔH is always applied if it
    # passes the sanity checks).
    use_refiner_gate: bool = True
    refiner_score_margin: float = 0.01

    # Temporal smoothing of the final projected quad corners across
    # frames within each track. Applies a center-weighted (Gaussian)
    # moving average to the 4 corner trajectories in frame space,
    # reducing both CoTracker tracking jitter and refiner prediction
    # noise. Works with or without the refiner — when the refiner is
    # off, smooths the raw H_from_frontal projections.
    # Set to 1 to disable (no smoothing). Minimum effective value is 3.
    temporal_smooth_window: int = 1
    # Gaussian sigma for the smoothing kernel, in frames. A good
    # starting value is window_size / 4 (σ≈2 for window=7).
    # Smaller σ → sharper center weight → less smoothing.
    temporal_smooth_sigma: float = 2.0


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
        if self.revert.use_refiner:
            if not self.revert.refiner_checkpoint_path:
                errors.append(
                    "revert.refiner_checkpoint_path is required when "
                    "revert.use_refiner is True"
                )
            if not self.propagation.save_target_canonical_roi:
                errors.append(
                    "propagation.save_target_canonical_roi must be True "
                    "when revert.use_refiner is True (the refiner needs "
                    "S4 to attach target_roi_canonical to each PropagatedROI)"
                )
            if not (0.0 <= self.revert.refiner_rejection_warn_threshold <= 1.0):
                errors.append(
                    "revert.refiner_rejection_warn_threshold must be in [0, 1]"
                )
            if self.revert.refiner_max_corner_offset_px <= 0:
                errors.append(
                    "revert.refiner_max_corner_offset_px must be > 0"
                )
        return errors
