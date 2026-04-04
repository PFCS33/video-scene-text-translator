# Plan: Sliding-Window Streaming Detection for TPM Data Gen Pipeline

## Goal
Remove the all-frames-in-memory bottleneck from `tpm_data_gen_pipeline.py` by introducing a `StreamingDetectionStage` that processes video frames through a `VideoReader` with bounded memory (~16-32 frames max). Enables processing arbitrarily long videos.

## Approach
Create a parallel `StreamingDetectionStage` with the same output contract (`list[TextTrack]`) but accepting a `VideoReader` instead of a preloaded frames list. Reuses existing `TextDetector` and `ReferenceSelector` unchanged. New streaming tracker handles optical flow gap-filling by seeking through the video per-track. New `CoTrackerOnlineFlowTracker` wraps `CoTrackerOnlinePredictor` for chunked online tracking.

**Key decisions:**
- **No edits to existing S1 code** — `stage.py`, `tracker.py`, `detector.py`, `selector.py` all stay untouched. Main pipeline unaffected.
- **2-pass architecture in `tpm_data_gen_pipeline.py`:**
  - Pass 1: Stream all frames sequentially → OCR detect → group → select references (no frame retention)
  - Pass 2: Per-track seek + sequential read → streaming optical flow gap-fill → S2 homographies → warp + save ROIs
- **Pairwise flow (Farneback/LK):** Read 2 frames at a time via `VideoReader`, compute flow, discard prev frame. Max 2 frames in memory.
- **CoTracker online:** Feed chunks of `step * 2` frames (~16) via `CoTrackerOnlinePredictor`. Max ~16 frames in VRAM per chunk. Uses `scaled_online.pth` checkpoint.
- **ROI extraction folds into Pass 2** — after gap-filling a track, immediately compute homographies and extract ROIs while frames are still being read. No third pass needed.
- **Short track fallback:** When `optical_flow_method == "cotracker"` but a track has fewer than `step * 2` frames, fall back to pairwise flow (Farneback) automatically. No padding or special windowing needed.
- **Future main pipeline migration:** Swap `DetectionStage` → `StreamingDetectionStage`, pass `VideoReader` instead of frames list.

## Files to Change
- [ ] `code/src/config.py` — Add `cotracker_online_checkpoint` field to `DetectionConfig`
- [ ] (new) `code/src/stages/s1_detection/streaming_tracker.py` — `StreamingTextTracker` with `fill_gaps_streaming(tracks, video_reader)`: per-track seek + pairwise or online flow
- [ ] (new) `code/src/utils/cotracker_online.py` — `CoTrackerOnlineFlowTracker` wrapping `CoTrackerOnlinePredictor` for chunked point tracking
- [ ] (new) `code/src/stages/s1_detection/streaming_stage.py` — `StreamingDetectionStage.run(video_reader) -> list[TextTrack]`: streaming orchestrator reusing `TextDetector` and `ReferenceSelector`
- [ ] `code/src/tpm_data_gen_pipeline.py` — Rewrite to 2-pass: use `StreamingDetectionStage` + per-track ROI extraction via `VideoReader`
- [ ] `code/tests/test_streaming_detection.py` — Tests for streaming tracker and stage

## Risks
- **CoTracker online vs offline quality:** Online mode processes in sliding windows and may produce slightly different tracks than offline batch mode. Need to verify visually.
- **VideoCapture seek performance:** `cv2.CAP_PROP_POS_FRAMES` seeking speed depends on video codec. For poorly-indexed codecs (e.g., some MPEG-4 files), seeking to each track's start could be slow. Mitigation: process tracks in frame-order to minimize backward seeks.
- **Short tracks with CoTracker online:** Tracks shorter than `step * 2` frames (~16) automatically fall back to pairwise flow (Farneback/LK). CoTracker's temporal consistency advantage is negligible for <16 frames, and pairwise flow handles short ranges well.
- **CoTracker online state is per-video-session:** `init_video_online_processing()` must be called per track. Each track is a separate tracking session.

## Done When
- [ ] `StreamingDetectionStage.run(video_reader)` produces identical tracks to `DetectionStage.run(frames_list)` for pairwise flow methods
- [ ] CoTracker online flow tracker produces reasonable tracks (visual verification on test video)
- [ ] `tpm_data_gen_pipeline.py` processes a 1080p video without loading all frames — peak memory stays bounded
- [ ] All existing tests still pass (zero regressions)
- [ ] New tests cover streaming tracker and stage
- [ ] Code review approved (@reviewer)
- [ ] Changes committed as atomic commits

## Progress
- [x] Step 1: Add `cotracker_online_checkpoint` to `DetectionConfig` in `config.py`
- [x] Step 2: Implement `CoTrackerOnlineFlowTracker` in `code/src/utils/cotracker_online.py`
- [x] Step 3: Implement `StreamingTextTracker` in `code/src/stages/s1_detection/streaming_tracker.py`
- [x] Step 4: Implement `StreamingDetectionStage` in `code/src/stages/s1_detection/streaming_stage.py`
- [x] Step 5: Rewrite `tpm_data_gen_pipeline.py` to use streaming 2-pass architecture
- [x] Step 6: Write tests for streaming components (4 tests, all passing)
- [x] Step 7: Manual verification on test_data/dryer2.mp4 — 34 tracks, 3020 ROIs extracted successfully
