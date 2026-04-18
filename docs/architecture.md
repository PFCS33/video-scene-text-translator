# Architecture: Cross-Language Scene Text Replacement

## Overview
5-stage video pipeline that replaces scene text across languages (e.g., English "DANGER" → Spanish "PELIGRO"), preserving font style, perspective, and lighting. Stage B (current) uses classical CV. Stage C (future) replaces key stages with learned models (STTN, TPM).

## Module Map
| Module | Responsibility | Interfaces |
|--------|---------------|------------|
| `pipeline.py` | Main orchestrator: wires S1→S5, manages frame I/O (in-memory) | Calls all stages; uses VideoReader/Writer |
| `tpm_data_gen_pipeline.py` | TPM data gen orchestrator: streaming 2-pass, extracts ROIs for TPM training | Uses StreamingDetectionStage + S2; VideoReader |
| `data_types.py` | Core dataclasses: BBox, Quad, TextDetection, TextTrack, FrameHomography, PropagatedROI, PipelineResult | Consumed by all stages |
| `config.py` | YAML config loading, validation, CLI override support. Includes `TPMDataGenConfig`. | Loaded by pipeline, passed to all stages |
| `video_io.py` | VideoReader / VideoWriter with context manager support; sequential seek optimization | Used by both pipelines |
| `s1_detection/detector.py` | EasyOCR or PaddleOCR detection with quality scoring + wordfreq gibberish filtering | In: frame → Out: list[TextDetection] |
| `s1_detection/tracker.py` | IoU tracking with track break threshold + text similarity check + frame-range-bounded propagation | In: detections → Out: list[TextTrack] |
| `s1_detection/selector.py` | Reference frame selection: hard pre-filters → 2-metric composite + max_frame_offset constraint | In: TextTrack → Out: reference_frame_idx |
| `s1_detection/stage.py` | In-memory S1 orchestrator: detect → track → translate → gap-fill → select | In: frames → Out: list[TextTrack] |
| `s1_detection/streaming_stage.py` | Streaming S1 orchestrator: 2-pass (OCR → gap-fill per-track) via VideoReader | In: VideoReader → Out: list[TextTrack] |
| `s1_detection/streaming_tracker.py` | Streaming gap-filler: per-track seek + pairwise or CoTracker online flow | In: tracks + VideoReader → Out: tracks with filled gaps |
| `s2_frontalization.py` | Homography computation: frame quad → canonical frontal rectangle | In: TextTrack + frames → Out: H stored on TextDetection |
| `s3_text_editing.py` | Stage A model wrapper via BaseTextEditor. When AnyText2 backend is used with adaptive mask, also lazy-loads a `BaseBackgroundInpainter` (same config as S4) for the long-to-short inpaint path. | In: reference ROI + target_text → Out: edited ROI |
| `models/anytext2_mask.py` | Pure helpers for AnyText2 adaptive mask sizing: target text width estimation (character-class heuristic), centered mask rect computation, feathered middle-strip restoration. No I/O, fully unit-tested. | Used by `anytext2_editor.AnyText2Editor` |
| `s4_propagation/` | LCM (per-pixel ratio map from inpainted backgrounds, when available) or YCrCb luminance histogram matching as fallback, + feathered alpha mask creation. BPN integration in progress. Two pluggable inpainter backends behind `BaseBackgroundInpainter`: `srnet` (learned text-removal network) and `hisam` (Hi-SAM pixel-level stroke segmentation + cv2.inpaint Navier-Stokes). | In: edited ROI + frame ROIs (+ inpainted backgrounds when available) → Out: dict[frame_idx → PropagatedROI] |
| `s5_revert.py` | Inverse homography warp + alpha blending + compositing | In: PropagatedROIs + frames → Out: final output frames |
| `base_text_editor.py` | ABC for Stage A models (edit_text, load_model) | Subclassed by concrete model backends |
| `placeholder_editor.py` | Pillow-based placeholder for pipeline testing (supports accented chars) | Implements BaseTextEditor |
| `geometry.py` | Homography computation, quad metrics (area, frontality, bbox ratio), point warping | Used by S1, S2, S5 |
| `image_processing.py` | Sharpness (Laplacian), contrast (Otsu interclass variance), histogram matching | Used by S1, S4 |
| `optical_flow.py` | Farneback (dense) + Lucas-Kanade (sparse) optical flow wrappers | Used by S1 tracker |
| `cotracker_online.py` | CoTracker3 online mode wrapper: chunked sliding-window GPU point tracking | Used by streaming tracker |

## Implementation Stages

### Stage A — Cross-Language Text Editing Model
Separate work, not in `code/`. Models consumed via `BaseTextEditor` interface.
- **RS-STE** (main focus): Transformer with recognition branch for implicit style separation. Training loop re-implemented, fine-tuning for cross-language (en→zh character alphabet expansion ~95→~6000).
- **AnyText2**: Diffusion-based (SD 1.5 + WriteNet + AttnX). Supports multilingual. Inference code imported, debugging dependencies.
- **CLASTE**: GAN-based cross-language specific. Would require full re-implementation — high effort, uncertain outcome.

### Stage B — Classical Video Pipeline (IMPLEMENTED)
Uses classical CV methods, aligned with STRIVE's frontalization-first design. 5-stage pipeline:
1. **S1 Detection + Tracking + Selection**: Split into submodules (`s1_detection/detector.py`, `tracker.py`, `selector.py`, `stage.py`). Two OCR backends: EasyOCR or PaddleOCR (configurable via `detection.ocr_backend`). Detections filtered by wordfreq gibberish scoring + optional word whitelist. IoU tracking with configurable track break threshold + text similarity checks. Optical flow gap-filling: Farneback (classical), Lucas-Kanade, or CoTracker3 (learned, GPU). Gap-fill bounded to track's frame range. Two strategies: `gaps_only` (fill missing frames) or `full_propagation` (overwrite all quads from reference). Reference selection: 4-metric scoring → 2-stage pre-filter → 2-metric composite, with `max_frame_offset` constraint for CoTracker online mode.
2. **S2 Frontalization**: Computes homography from each frame's quad to a canonical frontal rectangle (axis-aligned, derived from reference quad dimensions). Stores `H_to_frontal` / `H_from_frontal` directly on TextDetection. Pure geometry — no pixels warped.
3. **S3 Text Editing**: Warps reference frame to canonical frontal via `H_to_frontal` → passes clean frontal ROI to `BaseTextEditor.edit_text()` → stores edited_roi. Falls back to bbox crop if no homography. When using the AnyText2 backend with `anytext2_adaptive_mask: true`, S3 additionally lazy-loads a `BaseBackgroundInpainter` (reusing S4's config at `propagation.inpainter_backend`, but as an independent instance — S3 and S4 stay decoupled). For long-to-short translations (e.g. 7 CJK chars → 3 CJK chars) where target aspect diverges from source canonical by ≥ 15%, the editor pre-inpaints the canonical, restores a centered middle strip of original pixels matching the target's natural width, and sends this hybrid + a shrunk mask to AnyText2. This avoids AnyText2's known length-mismatch gibberish-fill behaviour (a documented limitation across the entire scene text editing literature — see GLASTE 2024). Non-AnyText2 backends are unaffected.
4. **S4 Propagation**: Warps each frame to canonical frontal via `H_to_frontal` → histogram matches luminance (CDF-based, YCrCb Y channel) against the frontalized edited ROI — pixel-aligned comparison. Creates feathered alpha mask. Falls back to bbox crop if no homography.
5. **S5 Revert**: Reads `H_from_frontal` from TextDetection → warps edited ROI to bounded target bbox region (not full frame) via `T @ H_from_frontal` → alpha blends only within bbox slice.

### TPM Data Generation Pipeline (IMPLEMENTED)
Streaming 2-pass pipeline for generating training data for the TPM (Text Propagation Model). Uses `StreamingDetectionStage` instead of in-memory `DetectionStage` to process arbitrarily long videos with bounded memory.
- **Pass 1 (streaming)**: Stream frames sequentially via `VideoReader` → OCR detect at sample_rate intervals → group into tracks → select reference frames. No frame retention in memory.
- **Pass 2 (per-track)**: For each track, seek to start frame → streaming optical flow gap-fill (pairwise Farneback or CoTracker online) → S2 homography → warp + extract canonical ROIs → save to disk.
- **CoTracker online mode**: `CoTrackerOnlineFlowTracker` wraps `CoTrackerOnlinePredictor` for chunked sliding-window tracking (~16 frames in VRAM). Short tracks (<step×2 frames) fall back to pairwise Farneback automatically.
- **Track serialization**: Supports save/load detected tracks as JSON for pipeline debugging (`tpm_data_gen.save_detected_tracks` / `load_detected_tracks`).

### Stage C — Full STRIVE Pipeline (NOT YET IMPLEMENTED)
- Replace S2 homography with STTN (Spatial-Temporal Transformer Network) — learned frontalization with temporal consistency
- Replace S4 histogram matching with TPM (LCM per-pixel lighting ratio + BPN differential blur prediction)

**Stage B vs STRIVE frontalization:**
- Stage B now frontalizes to a canonical rectangle (same flow as STRIVE: frontalize → edit → propagate → de-frontalize), but uses classical homography instead of learned STTN
- Classical homography is the exact solution for planar text but doesn't handle non-planar surfaces or temporal consistency
- STTN processes frame stacks jointly for temporal smoothness; Stage B computes each frame's homography independently

## Cross-Cutting Concerns
- **Config**: Two configs: `config/default.yaml` (classical Farneback + EasyOCR) and `config/adv.yaml` (CoTracker + PaddleOCR). Nested dataclasses with validation. CLI overrides via argparse.
- **Error handling**: Config validation gates bad input early. Fallback paths in reference selection (if pre-filters eliminate all candidates, use all detections). Empty-ROI guards in S1. Invalid homography checks in S2/S5.
- **Logging**: Python logging module at INFO level per stage, DEBUG for detailed tracing. Timing logs for OCR and optical flow. tqdm progress bars for CoTracker and OCR loops.
- **Memory**: Main pipeline holds all frames in dict (works for short clips, breaks >500 frames). TPM data gen pipeline uses streaming VideoReader with bounded memory (~2-16 frames max).
- **Third-party deps**: CoTracker3 and PaddleOCR installed via scripts in `third_party/`. Both are optional — pipeline falls back to Farneback/EasyOCR if not available. Hi-SAM is vendored at `third_party/Hi-SAM/` for the segmentation-based background inpainter. Its inference deps overlap with what the main venv already ships (torch, torchvision, einops, shapely, pyclipper, scikit-image) — no extra `pip install` required. The wrapper uses `contextlib.chdir(third_party/Hi-SAM)` during `load_model()` because upstream `build.py` hardcodes a relative path to the SAM ViT encoder weights; the chdir is scoped to construction only and cwd is restored immediately after.

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
- **Tracking**: IoU-based greedy matching with configurable break threshold and text similarity checks. Breaks with large camera motion. Hungarian algorithm would improve but Stage C's STTN would replace entirely.
- **Optical flow drift**: Accumulates over long sequences for classical methods. CoTracker3 produces smoother trajectories but requires GPU. Bidirectional propagation helps for classical methods.
- **Gap-filling propagates to all frames**: Optical flow fills quads for ALL video frames, including frames where text is genuinely absent (occluded, out of view, camera cut). This causes false replacements. Fix options: (1) validate tracked quads — reject if area changes >50% between frames or quad becomes degenerate/self-intersecting, (2) appearance verification — check contrast/sharpness in tracked region to confirm text is still present.
- **Lighting adaptation**: Global histogram matching on aligned ROIs — better than misaligned, but still doesn't handle spatially varying lighting (shadows, specular highlights). Per-pixel lighting ratio (classical LCM) would require inpainting to isolate background before computing the ratio, otherwise artifacts at text edges.
- **No blur modeling**: Pipeline ignores motion blur and focus blur entirely. STRIVE's BPN predicts differential blur between frames — no classical equivalent without a learned model.
- **Temporal jitter**: Each frame's homography is computed independently — no smoothing across time. Temporal smoothing would require decomposing homography into translation/rotation/scale before averaging (can't average raw 3×3 matrices).
- **Placeholder editor**: Pillow-based text rendering (supports accented characters) but no style matching. Awaiting real Stage A model integration.
- **Translation backend**: googletrans is unofficial/unreliable. Config supports google-cloud-translate but requires API key setup.
- **Memory**: Main pipeline holds all frames in memory — not viable for >500 frames. TPM data gen pipeline uses streaming 2-pass (bounded memory), but main pipeline has not been migrated yet.

## Tech Stack
- **Python 3.11 + conda**: Chosen for OpenCV/numpy ecosystem compatibility and team familiarity
- **OpenCV**: Core CV operations — homography, optical flow, color space conversion, alpha blending
- **EasyOCR**: Scene text detection — default OCR backend, good multi-language support
- **PaddleOCR**: Alternative OCR backend — faster, better accuracy on some scenes, configurable via `detection.ocr_backend`
- **CoTracker3**: Meta's learned point tracker — ~25x faster than Farneback, smoother trajectories. Offline mode for batch, online mode for streaming. Requires GPU.
- **Farneback optical flow**: Classical dense method (CPU-only), default fallback. Lucas-Kanade as sparse alternative.
- **wordfreq**: Gibberish detection — filters OCR noise using zipf frequency thresholds
- **tqdm**: Progress bars for long-running OCR and CoTracker loops
- **Pillow**: Image I/O and text rendering (supports accented/Unicode characters, replaced cv2.putText in placeholder editor)
- **ruff**: Linting and formatting — fast, all-in-one Python linter

## Open Questions
- Stage C: STTN model integration approach (replaces S2 homography with learned spatial transformer)
- Stage C: TPM model integration approach (replaces S4 histogram matching with LCM + BPN) — TPM data gen pipeline now exists to produce training data
- ~~Sliding-window frame loading strategy for long videos~~ — **Resolved**: streaming 2-pass architecture in TPM data gen pipeline. Main pipeline still in-memory.

## Web Application

Browser-based live-demo frontend for the pipeline. Single FastAPI process
serves the React SPA and the `/api/*` surface; pipeline runs in-process on
one worker thread.

```
┌─────────────────────────┐
│  Browser                │
│  React + Vite + TS      │
│  Tailwind + shadcn/ui   │
└──────────┬──────────────┘
           │ HTTP + SSE (same origin)
           ▼
┌──────────────────────────────────────────────┐
│  FastAPI (uvicorn) on GPU box                │
│  ├── /api/*    REST + SSE                    │
│  └── /        static React bundle            │
│                                              │
│  JobManager (1 worker thread, in-mem dict)   │
│    │                                         │
│    ▼                                         │
│  PipelineRunner                              │
│    ├── attaches logging.Handler              │
│    │   → asyncio.Queue → SSE                 │
│    ├── passes progress_callback              │
│    │   → VideoPipeline                       │
│    └── calls VideoPipeline(config).run()     │
└──────────┬───────────────────────────────────┘
           │ HTTP
           ▼
┌────────────────────────┐
│  AnyText2 Gradio server│   (already external,
│  (separate process)    │    unchanged)
└────────────────────────┘
```

### What lives where
- `server/` — FastAPI app (`app/main.py`), routes (`app/routes.py`), job
  registry + single-worker executor (`app/jobs.py`), pipeline wrapper
  (`app/pipeline_runner.py`), storage (`app/storage.py`), Pydantic
  schemas (`app/schemas.py`), curated language list
  (`app/languages.py`). Tests in `server/tests/`. Nested conventions
  doc at `server/CLAUDE.md`.
- `web/` — React 18 + Vite + TS SPA. Components in `src/components/`
  (shadcn primitives under `components/ui/`), API client + SSE helper
  in `src/api/`, job-lifecycle hook in `src/hooks/useJobStream.ts`,
  design tokens in `src/styles/globals.css`. Nested conventions doc at
  `web/CLAUDE.md`.

### API surface

| Method | Path                            | Purpose |
|--------|---------------------------------|---------|
| POST   | `/api/jobs`                     | multipart: `video` file + `source_lang` + `target_lang` → `{job_id}` |
| GET    | `/api/jobs/{job_id}/status`     | `{status, current_stage?, created_at, finished_at?, error?, output_available}` |
| GET    | `/api/jobs/{job_id}/events`     | SSE stream of events (see below) |
| GET    | `/api/jobs/{job_id}/output`     | streams the output MP4, `Content-Disposition: attachment` |
| DELETE | `/api/jobs/{job_id}`            | deletes job + files (409 if running) |
| GET    | `/api/languages`                | `[{code, label}, ...]` — curated list for the dropdown |
| GET    | `/` (and other static paths)    | serves the built React bundle |

### SSE event shapes

```ts
type Event =
  | { type: "stage_start",    stage: "s1"|"s2"|"s3"|"s4"|"s5", ts: number }
  | { type: "stage_complete", stage: "s1"|"s2"|"s3"|"s4"|"s5", duration_ms: number, ts: number }
  | { type: "log",            level: "info"|"warning"|"error", message: string, ts: number }
  | { type: "done",           output_url: string, ts: number }
  | { type: "error",          message: string, traceback?: string, ts: number }
```

### Job lifecycle
1. `POST /api/jobs` (multipart) → upload streamed to
   `server/storage/uploads/{job_id}/`; pipeline queued. Response
   `{job_id}`. Second submit while one is active → 409 with
   `{error: "concurrent_job", active_job_id}` so the client can render
   a rejoin link (R8) instead of a hard error.
2. Queued → `running` when the worker picks the job up.
3. `running` — the SSE stream emits `stage_start` / `stage_complete`
   events (10 per run, 5 stages × start/done) plus interleaved `log`
   events forwarded from the pipeline's `src.*` logger tree.
4. Terminal — either `done` (with `output_url = /api/jobs/{id}/output`)
   or `error` (with `message` + `traceback`). The event stream closes
   after the terminal event.

### Concurrency + persistence model
- Single `ThreadPoolExecutor(max_workers=1)` owned by `JobManager`.
  Only one `VideoPipeline.run()` is ever in flight. A second submit
  raises `ConcurrentJobError` (→ 409).
- In-memory `dict[str, _JobRecord]` keyed by UUID4. No database, no
  disk persistence. Server restart loses all job state.
- Cleanup paths:
  (a) explicit `DELETE /api/jobs/{id}` removes the record + files;
  (b) TTL sweep on boot (`storage.sweep_old_jobs(ttl_hours=2)`) purges
  job dirs older than 2 hours from the last crash.

### Key design decisions
- **In-process import, not subprocess (D3).** `from src.pipeline import
  VideoPipeline; pipeline.run()` on a `ThreadPoolExecutor` worker. Lazy
  imports inside `pipeline_runner.run_pipeline_job` keep FastAPI boot
  free of torch/paddle/cv2.
- **SSE over WebSocket (D5).** One-way pipeline → browser only. Browser
  `EventSource` handles auto-reconnect; no framing layer to maintain.
  Implemented via `sse_starlette.EventSourceResponse`.
- **Same-origin, no CORS (D9).** FastAPI `app.mount("/",
  StaticFiles(...))` serves the built SPA after
  `app.include_router(router)` (order matters — static must come last).
  Dev uses Vite proxy forwarding `/api/*` to `:8000`.
- **Structured progress via `progress_callback` (D11).** 5-line
  pipeline change: `VideoPipeline.__init__` gained an optional
  `progress_callback: Callable[[str], None]`; called with
  `"stage_N_{start|done}"`. Runner adapts strings → `StageStartEvent` /
  `StageCompleteEvent`. Chosen over log-message parsing for stability.
- **D16 terminal-state-flip invariant.** The `emit` closure in
  `JobManager._run_job` flips `record.status` to `succeeded` / `failed`
  *before* enqueueing the terminal event. A client that reads
  `DoneEvent` over SSE and races to `/status` cannot observe a stale
  `"running"`. Symmetric for error path.
- **Output MP4 transcoded via ffmpeg (R3 / D15).** OpenCV's
  `VideoWriter` with `mp4v` fourcc emits `FMP4` (MPEG-4 Part 2), which
  browsers refuse to play. `pipeline_runner._transcode_to_browser_safe`
  shells out to `ffmpeg -c:v libx264 -pix_fmt yuv420p -movflags
  +faststart` after `VideoPipeline.run()` and atomic-swaps the file.

### Known limitations
- **No real cancellation (D14, R1).** `DELETE /api/jobs/{id}` on a
  running job returns 409. Cooperative cancellation would require
  stop-flag checks inside the pipeline — out of scope for MVP.
- **Single-worker queue-of-one (D4, R8).** A second submit during a
  running job is rejected. Mitigated client-side with the rejoin link
  in the 409 body, not by actually queueing.
- **Long videos still break (>500 frames).** The main `VideoPipeline`
  holds all frames in memory (see Known Limitations above). The web
  app inherits this; TPM data gen's streaming architecture has not
  been ported to the main pipeline. Upload size is capped at 200 MiB
  (R2) as a rough bound.
- **Browser codec assumption.** Relies on ffmpeg being on PATH for the
  transcode step. Validated at integration-test time (R3); missing
  ffmpeg surfaces as a `RuntimeError` → `ErrorEvent`.

### Running the app
Dev mode — hot-reload backend + Vite dev server:
```bash
./server/scripts/dev.sh
# uvicorn --reload on :8000, Vite on :5173, /api proxied to :8000
```

Prod/demo mode — build the SPA, serve everything from FastAPI:
```bash
./server/scripts/build_frontend.sh
# → web/dist → server/app/static/
python -m uvicorn server.app.main:app --host 0.0.0.0 --port 8000
# browse http://localhost:8000/
```
