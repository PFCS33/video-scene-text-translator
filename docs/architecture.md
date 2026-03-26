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
| `s4_propagation.py` | YCrCb luminance histogram matching + feathered alpha mask creation | In: edited ROI + frame ROIs → Out: dict[frame_idx → PropagatedROI] |
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

### Stage B — Basic Video Pipeline (IMPLEMENTED)
Uses classical CV methods. 5-stage pipeline:
1. **S1 Detection**: EasyOCR → detect text → IoU-based tracking → Google Translate → score frames (4-metric composite: OCR confidence, sharpness, contrast, frontality) → 2-stage pre-filter (OCR threshold + top-K sharpness) → 2-metric reference selection (0.7 contrast + 0.3 frontality)
2. **S2 Frontalization**: Optical flow (Farneback default, LK alternative) → bidirectional quad propagation from reference → `cv2.findHomography(RANSAC)` per frame → H_to_ref / H_from_ref
3. **S3 Text Editing**: Extract reference ROI → call `BaseTextEditor.edit_text(roi, target_text)` → store edited_roi on TextTrack
4. **S4 Propagation**: YCrCb luminance histogram matching (CDF-based) → adapt edited ROI to each frame's lighting → create feathered alpha mask
5. **S5 Revert**: Inverse homography warp via H_from_ref → alpha blending → composite into original frames

### Stage C — Full STRIVE Pipeline (NOT YET IMPLEMENTED)
- Replace S2 optical flow + homography with STTN (Spatial-Temporal Transformer Network)
- Replace S4 histogram matching with TPM (Temporal Propagation Module)

**Frontalization difference (Stage B vs STRIVE):**
- Stage B does NOT do true frontalization. The reference frame's natural perspective is treated as "frontal". S2 only computes the geometric mapping (H_to_ref / H_from_ref) between frames. The edited ROI is propagated outward from reference via H_from_ref in S5.
- STRIVE uses STTN to warp every frame's ROI to a canonical frontal rectangle, edits text in that normalized space, then transfers back. STTN sees multiple frames jointly for temporal consistency.
- Classical frontalization (getPerspectiveTransform quad→rect) is feasible for planar text but doesn't handle non-planar surfaces, motion blur, or temporal smoothness like STTN does.
- If implementing Stage C: replace S2's optical flow + homography with STTN, and the pipeline needs an explicit frontalize→edit→de-frontalize flow instead of the current ref-centric propagation.

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
- **Bidirectional propagation**: S2 propagates quads forward and backward from reference frame — more stable than one-direction.

## Known Limitations
- **Tracking**: IoU-based greedy matching breaks with large camera motion. Hungarian algorithm would improve but Stage C's STTN would replace entirely.
- **Optical flow drift**: Accumulates over long sequences. Bidirectional propagation helps but not perfect.
- **Lighting adaptation**: Global histogram matching per ROI — doesn't handle spatially varying lighting (shadows, specular highlights).
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
- Stage C: STTN model integration approach (replaces S2 optical flow + homography)
- Stage C: TPM model integration approach (replaces S4 histogram matching)
- Whether to split `s1_detection.py` (256 lines, 4 responsibilities: detection, translation, tracking, selection)
- Sliding-window frame loading strategy for long videos
