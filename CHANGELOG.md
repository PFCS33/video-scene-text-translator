# Changelog

## 2026-04-06 — AnyText2 Integration (feat/anytext2-integration)

### Stage A Model
- Integrate AnyText2 (ICLR 2025) as a real Stage A text editing backend, replacing the placeholder for cross-language scene text replacement
- `AnyText2Editor` subclass of `BaseTextEditor` communicates with an external AnyText2 Gradio server via `gradio_client`
- Supports style-preserving editing: uses "Mimic From Image" font extraction and auto-detected text color from the ROI
- Handles ROI dimension clamping (256-1024px range), auto-resize back to original dimensions

### Configuration
- Add `text_editor.server_url`, `server_timeout`, and AnyText2-specific params (`ddim_steps`, `cfg_scale`, `strength`, `img_count`) to `TextEditorConfig`
- `adv.yaml` defaults to `backend: "anytext2"` with server URL pre-configured
- `default.yaml` keeps `backend: "placeholder"` for offline testing

### Testing
- 15 new unit tests for AnyText2Editor: color extraction, dimension clamping, edge cases, mocked Gradio calls, S3 integration
- All tests run without AnyText2 server (fully mocked)

### Misc
- Add `third_party/install_anytext2.sh` for setting up AnyText2 server (clone, conda env, model download)
- Add `gradio_client` to `requirements/base.txt`

## 2026-04-05 — TPM Data Generation Pipeline (experiment/tpm_data_gen)

### Core
- Initial implementation of TPM data generation pipeline (`tpm_data_gen_pipeline.py`, `run_tpm_data_gen_pipeline.py`) with CLI entry point, reusing Stage 1 detection and tracker

### Streaming Architecture (TPM data gen pipeline only)
- Replace all-frames-in-memory loading with a streaming 2-pass pipeline: Pass 1 streams frames for OCR detection/grouping/reference selection, Pass 2 runs per-track optical flow gap-fill + frontalization + ROI extraction (the main translation pipeline still uses the original in-memory approach)
- Add `StreamingDetectionStage` and `StreamingTextTracker` that read frames on demand via `VideoReader`
- Add `CoTrackerOnlineFlowTracker` wrapping the online predictor for chunked tracking with sliding-window GPU memory management

### Detection Improvements
- Add PaddleOCR as a configurable OCR backend alongside EasyOCR (`detection.ocr_backend`)
- Filter gibberish OCR detections using `wordfreq` zipf frequency thresholds
- Add configurable word whitelist (`--word-whitelist` CSV) to bypass gibberish filter for domain-specific text
- Add hard filter for longest text in reference frame selection
- Restrict quad propagation to track's frame range to reduce spurious detections
- Add track break threshold and text similarity check for tracking

### Performance
- Skip redundant seeks for sequential frame reads in `VideoReader` by tracking decoder position
- Replace `iter_frames()` with `read_frame()` at sample_rate intervals to avoid decoding skipped frames
- Add tqdm progress bars to CoTracker online processing and OCR detection loops

### CoTracker Online Fixes
- Rewrite CoTracker online to stream frames forward with overlapping windows matching official `online_demo.py` pattern
- Pad last online chunk when shorter than step×2 to prevent dropped frames
- Keep partially occluded frames (partial occlusion on 4 corners is rarely meaningful)
- Add `max_frame_offset` to `ReferenceSelector` to constrain reference frame to first window of each track

### Configuration
- Add `adv.yaml` with advanced configuration options for CoTracker and PaddleOCR
- Add options to save and load detected tracks from JSON for pipeline debugging
- Update default config path to `adv.yaml` for TPM data gen pipeline

### Misc
- Add PaddleOCR install script (`third_party/`)

## 2026-04-01 — CoTracker Integration (experiment/cotracker)

- Add `flow_fill_strategy` config option: `gaps_only` (original) vs `full_propagation` (overwrite all OCR quads with optical-flow-tracked quads from reference frame)
- Integrate Meta CoTracker3 as a new `optical_flow_method` option, replacing pairwise Farneback/LK with batch point tracking (~25x faster, smoother trajectories)
- Add timing logs for OCR detection and optical flow steps in Stage 1
- Add `third_party/install_cotracker.sh` for cloning and installing CoTracker
