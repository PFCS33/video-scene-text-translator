# Architecture: Cross-Language Scene Text Replacement

## Overview
5-stage video pipeline that replaces scene text across languages (e.g., English "DANGER" → Spanish "PELIGRO"), preserving font style, perspective, and lighting. Stage B (current) uses classical CV. Stage C (future) replaces key stages with learned models (STTN, TPM).

## Module Map
| Module | Responsibility | Interfaces |
|--------|---------------|------------|
| `pipeline.py` | Orchestrator: wires S1→S5, manages frame I/O | Calls all stages; uses VideoReader/Writer |
| `data_types.py` | Core dataclasses: BBox, Quad, TextDetection, TextTrack, FrameHomography, PropagatedROI, PipelineResult | Consumed by all stages |
| `config.py` | YAML config loading, validation, CLI override support | Loaded by pipeline, passed to all stages |
| `video_io.py` | VideoReader / VideoWriter with context manager support | Used by pipeline.py |
| `s1_detection.py` | EasyOCR detection → IoU tracking → translation → composite scoring → reference frame selection | In: frames → Out: list[TextTrack] |
| `s2_frontalization.py` | Optical flow quad tracking → homography computation per frame | In: TextTrack + frames → Out: dict[frame_idx → FrameHomography] |
| `s3_text_editing.py` | Stage A model wrapper via BaseTextEditor | In: reference ROI + target_text → Out: edited ROI |
| `s4_propagation/` | LCM (per-pixel ratio map from inpainted backgrounds, when available) or YCrCb luminance histogram matching as fallback, + feathered alpha mask creation. BPN integration in progress. | In: edited ROI + frame ROIs (+ inpainted backgrounds when available) → Out: dict[frame_idx → PropagatedROI] |
| `s5_revert.py` | Inverse homography warp + alpha blending + compositing | In: PropagatedROIs + frames → Out: final output frames |
| `base_text_editor.py` | ABC for Stage A models (edit_text, load_model) | Subclassed by concrete model backends |
| `placeholder_editor.py` | OpenCV putText placeholder for pipeline testing | Implements BaseTextEditor |
| `geometry.py` | Homography computation, quad metrics (area, frontality, bbox ratio), point warping | Used by S1, S2, S5 |
| `image_processing.py` | Sharpness (Laplacian), contrast (Otsu interclass variance), histogram matching | Used by S1, S4 |
| `optical_flow.py` | Farneback (dense) + Lucas-Kanade (sparse) optical flow wrappers | Used by S2 |

## Implementation Stages

### Stage A — Cross-Language Text Editing Model
Separate work, not in `code/`. Models consumed via `BaseTextEditor` interface.
- **RS-STE** (main focus): Transformer with recognition branch for implicit style separation. Training loop re-implemented, fine-tuning for cross-language (en→zh character alphabet expansion ~95→~6000).
- **AnyText2**: Diffusion-based (SD 1.5 + WriteNet + AttnX). Supports multilingual. Inference code imported, debugging dependencies.
- **CLASTE**: GAN-based cross-language specific. Would require full re-implementation — high effort, uncertain outcome.

### Stage B — Classical Video Pipeline (IMPLEMENTED)
Uses classical CV methods, aligned with STRIVE's frontalization-first design. 5-stage pipeline:
1. **S1 Detection + Tracking + Selection**: Split into submodules (`s1_detection/detector.py`, `tracker.py`, `selector.py`, `stage.py`). EasyOCR → detect text on sampled frames → IoU tracking → optical flow gap-filling (bidirectional from reference, covers all frames) → Google Translate → reference frame selection (4-metric scoring → 2-stage pre-filter → 2-metric composite). Updates source_text from reference frame's OCR.
2. **S2 Frontalization**: Computes homography from each frame's quad to a canonical frontal rectangle (axis-aligned, derived from reference quad dimensions). Stores `H_to_frontal` / `H_from_frontal` directly on TextDetection. Pure geometry — no pixels warped.
3. **S3 Text Editing**: Warps reference frame to canonical frontal via `H_to_frontal` → passes clean frontal ROI to `BaseTextEditor.edit_text()` → stores edited_roi. Falls back to bbox crop if no homography.
4. **S4 Propagation**: Warps each frame to canonical frontal via `H_to_frontal` → histogram matches luminance (CDF-based, YCrCb Y channel) against the frontalized edited ROI — pixel-aligned comparison. Creates feathered alpha mask. Falls back to bbox crop if no homography.
5. **S5 Revert**: Reads `H_from_frontal` from TextDetection → warps edited ROI to bounded target bbox region (not full frame) via `T @ H_from_frontal` → alpha blends only within bbox slice.

### Stage C — Full STRIVE Pipeline (NOT YET IMPLEMENTED)
- Replace S2 homography with STTN (Spatial-Temporal Transformer Network) — learned frontalization with temporal consistency
- Replace S4 histogram matching with TPM (LCM per-pixel lighting ratio + BPN differential blur prediction)

**Stage B vs STRIVE frontalization:**
- Stage B now frontalizes to a canonical rectangle (same flow as STRIVE: frontalize → edit → propagate → de-frontalize), but uses classical homography instead of learned STTN
- Classical homography is the exact solution for planar text but doesn't handle non-planar surfaces or temporal consistency
- STTN processes frame stacks jointly for temporal smoothness; Stage B computes each frame's homography independently

## Cross-Cutting Concerns
- **Config**: All parameters in `config/default.yaml`. Nested dataclasses with validation. CLI overrides via argparse in `run_pipeline.py`.
- **Error handling**: Config validation gates bad input early. Fallback paths in reference selection (if pre-filters eliminate all candidates, use all detections). Empty-ROI guards in S1. Invalid homography checks in S2/S5.
- **Logging**: Python logging module at INFO level per stage, DEBUG for detailed tracing. Level set via config.
- **Memory**: All frames held in dict keyed by frame_idx. Works for short clips. Needs sliding-window refactor for long videos (>500 frames).

## Key Design Decisions
- **Central data structure**: `TextTrack` flows through all 5 stages. S1 creates it, S2-S5 enrich it. Prevents scattered state across stages.
- **Stage A abstraction**: `BaseTextEditor` ABC with `edit_text(roi, target_text) → ndarray` contract. Swap models via `text_editor.backend` config — no pipeline code changes needed.
- **Config-driven**: All tunable parameters in YAML. Validation enforces weight sums, range bounds, and file existence.
- **Detections keyed by frame_idx**: Dict (not list) for O(1) lookup. Handles sparse detections naturally.
- **Lazy initialization**: EasyOCR, translator, and text editor init on first use — avoids import failures when deps not installed.
- **Bidirectional propagation**: S1 propagates quads forward and backward from reference frame via optical flow — more stable than one-direction.
- **Canonical frontalization**: All stages operate in a shared canonical frontal space (axis-aligned rectangle). H_to_frontal / H_from_frontal stored on TextDetection — single source of truth, no parallel data structures.
- **Matrices only, warp on-the-fly**: S2 stores 3×3 matrices, downstream stages warp when they need pixels. Keeps S2 as pure geometry, avoids memory bloat.

## Known Limitations
- **Tracking**: IoU-based greedy matching breaks with large camera motion. Hungarian algorithm would improve but Stage C's STTN would replace entirely.
- **Optical flow drift**: Accumulates over long sequences. Bidirectional propagation helps but not perfect.
- **Gap-filling propagates to all frames**: Optical flow fills quads for ALL video frames, including frames where text is genuinely absent (occluded, out of view, camera cut). This causes false replacements. Fix options: (1) validate tracked quads — reject if area changes >50% between frames or quad becomes degenerate/self-intersecting, (2) appearance verification — check contrast/sharpness in tracked region to confirm text is still present.
- **Lighting adaptation**: Global histogram matching on aligned ROIs — better than misaligned, but still doesn't handle spatially varying lighting (shadows, specular highlights). Per-pixel lighting ratio (classical LCM) would require inpainting to isolate background before computing the ratio, otherwise artifacts at text edges.
- **No blur modeling**: Pipeline ignores motion blur and focus blur entirely. STRIVE's BPN predicts differential blur between frames — no classical equivalent without a learned model.
- **Temporal jitter**: Each frame's homography is computed independently — no smoothing across time. Temporal smoothing would require decomposing homography into translation/rotation/scale before averaging (can't average raw 3×3 matrices).
- **Placeholder editor**: cv2.putText produces crude output with no style matching. Awaiting real Stage A model integration.
- **Translation backend**: googletrans is unofficial/unreliable. Config supports google-cloud-translate but requires API key setup.
- **Memory**: All frames in memory. Not viable for videos >500 frames without sliding-window refactor.

## Tech Stack
- **Python 3.11 + conda**: Chosen for OpenCV/numpy ecosystem compatibility and team familiarity
- **OpenCV**: Core CV operations — homography, optical flow, color space conversion, alpha blending
- **EasyOCR**: Scene text detection — chosen over Tesseract for multi-language and scene-text robustness
- **Farneback optical flow**: Default dense method, with Lucas-Kanade as sparse alternative (both classical, no GPU needed)
- **ruff**: Linting and formatting — fast, all-in-one Python linter

## Open Questions
- Stage C: STTN model integration approach (replaces S2 homography with learned spatial transformer)
- Stage C: TPM model integration approach (replaces S4 histogram matching with LCM + BPN)
- Sliding-window frame loading strategy for long videos
