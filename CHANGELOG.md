# Changelog

## 2026-04-19 — Retrain BPN on S2-Aligned Dataset, Enable by Default (feat/bpn_aligned_dataset)

### Why

The prior BPN checkpoint (`bpn_v0.pt`) was trained against two broken
inputs: (1) the old `DifferentiableBlur` that zero-padded its
convolution — producing artificial border darkness the optimizer had to
explain away — and (2) raw S1 canonical ROIs whose (ref, target) pairs
were not pixel-aligned, so the Stage 2 reconstruction loss pushed the
network to explain geometric misalignment as blur. Both are fixed now:
the blur is reflect-padded and a fresh `/workspace/bpn_dataset` was
extracted with the S2 refiner folded into each `H_to_frontal`. This
changelog entry covers the retrained checkpoint and the config flip
that turns BPN back on in `adv.yaml`.

### Retraining Pipeline Ready

- `code/src/models/bpn/dataset.py` — top docstring updated; metadata
  JSON lookup now tries `corrected_track_info.json` (S2-refined) first
  and falls back to legacy `s1_tracks.json`. Both files expose the
  only two fields the loader needs (`track_id`, `reference_frame_idx`),
  so the same `BPNDataset` works against either layout.
- `code/src/models/bpn/config.yaml` + the two test configs — default
  `data_root` points at `/workspace/bpn_dataset` with a comment
  pointing back at `scripts/generate_bpn_dataset.py` for regeneration
  and noting how to swap in `/workspace/tpm_dataset` for a
  pre-refinement baseline A/B.

### bpn_v1 Checkpoint

- Trained Stage 1 (synthetic blur supervision) → Stage 2
  (self-supervised on S2-refined `/workspace/bpn_dataset`) with the
  fixed reflect-padded `DifferentiableBlur`.
- Evaluation on the held-out validation split:
  - Reconstruction MSE: 0.00383 ± 0.00544 (std > mean — cluttered /
    long sentences drive the upper tail).
  - `sigma_x`: mean 1.71, range [0.85, 8.6] — back inside the Stage 1
    training prior of [0.3, 1.8] (vs `bpn_v0`'s median 15 up to 35).
  - `sigma_y`: mean 1.44, range [0.38, 8.7].
  - `rho`: mean ≈ 0, std 0.0024 — known limitation, the model defaults
    to axis-aligned anisotropic Gaussians rather than learning
    rotations. Self-supervised reconstruction doesn't disentangle rho
    from content well.
  - `w`: mean -0.30, range [-1.0, +0.68] — blur-dominant but does
    reach the sharpen half-space. Positive-`w` predictions are rare in
    practice because S1 reference selection picks near-peak-sharpness
    frames, so targets are rarely sharper than the reference.

### Config Flip in adv.yaml

- `propagation.use_bpn: true` (was `false`).
- `propagation.bpn_checkpoint_path: ../checkpoints/bpn/bpn_v1.pt` (was
  `../checkpoints/bpn/bpn_v0.pt`).
- `checkpoints/download.sh` updated to pull `bpn_v1.pt` from the
  corresponding Drive ID.

### Known Limitations (tracked, not blocking)

- Oriented motion blur is not modeled (rho collapse). Acceptable
  tradeoff for this project's scope.
- De-blurring (target sharper than ref) is rarely applied with
  visible effect. Driven by reference-selection's sharpness bias
  in the training data rather than the network itself.
- Cluttered long sentences are the failure mode pulling reconstruction
  MSE's upper tail — matches the alignment-quality signal already
  flagged elsewhere.

## 2026-04-18 — Fix BPN Border Halo via Reflect-Padding (feat/mv_refine_to_s2)

### Symptom

With BPN enabled, every propagated ROI in the output video rendered as
a visibly brighter blob than its surrounding unedited pixels. The
brightness scaled monotonically with how much blur BPN predicted.

### Root Cause

`DifferentiableBlur.forward` called `F.conv2d(img, kern, padding=pad)`,
which uses **zero-padding**. Convolving against zero-padded borders
darkens the blurred output near the ROI edges: a pixel at the boundary
integrates its normalized kernel over a window that's partially
outside the image, and the "outside" contributes zero.

The differential-blur formula `I_out = (1+w)*I - w*blurred` with `w<0`
propagates this: interior pixels are unaffected, but border pixels pick
up the artificial darkness.

S5 then composites via `cv2.seamlessClone`, which solves Poisson with
the source's gradient field and the destination frame's boundary
values. The Poisson solve:
- Anchors the composite to the frame's boundary (matches surroundings).
- Integrates the source's gradient field inward from that boundary.

Because BPN had artificially darkened the source's border, the "source
interior minus source border" was positive, so the integrated composite
interior sat **above** the frame's ambient — a bright halo. Stronger
BPN (bigger effective kernel, larger |w|) → bigger border darkening →
bigger halo.

### Investigation

Per-call diagnostic logging was added to `BPNPredictor.apply_blur`
(since removed) measuring `(sigma_x, sigma_y, w, mean_before,
mean_after, Δmean, clamp fractions)`. Initial read of 482 calls on one
video showed all 482 Δ values negative (BPN darkening per-pixel mean
by ~2.5% on average) — which was a red herring. User confirmed two
key observations that pinned the mechanism:

1. With sigma=31 and |w|=0.28 the ROI only looked slightly blurred
   (kernel truncation at 41 px caps effective blur; the huge sigmas
   from an out-of-distribution BPN checkpoint weren't the primary
   issue).
2. Composite border pixels matched the frame's ambient perfectly; the
   interior sat above. This rules out a constant brightness shift in
   the source and points at a gradient integrated from a biased
   boundary.

Together: the boundary condition was correct (Poisson enforces frame's
ambient at the border), but the source had an artificially dark border
that translated into a positive inward gradient integral.

### Fix

[code/src/models/bpn/blur.py:86-92](code/src/models/bpn/blur.py#L86-L92) — replace

    F.conv2d(img_flat, kern_flat, padding=pad, groups=B * C)

with

    img_padded = F.pad(img_flat, (pad, pad, pad, pad), mode="reflect")
    F.conv2d(img_padded, kern_flat, padding=0, groups=B * C)

Reflect-padding mirrors the image's own pixels into the kernel support
at the boundary, so the convolution is mean-preserving all the way to
the edge. No more artificial dark rim; no more Poisson halo.

### Verification

- Per-call Δmean dropped from ~-0.025 (zero-pad) to ~-0.0001 (reflect-
  pad) — two orders of magnitude — on the same video with identical
  predicted params. User confirmed the halo is gone in the composite.
- On a flat 0.5 input, reflect-padded output is 0.5000 at every pixel
  including the corners. Zero-padded output dropped below 0.5 near the
  border depending on sigma.
- Border-vs-interior mean difference on a random image dropped from
  ~0.06 at |w|=0.5 to ~0.002 (sampling noise).

### Testing

- New `test_bpn_blur.py` with 5 regression tests under
  `TestBorderPreservation` (flat-image exact preservation, random-image
  border-vs-interior bound, identity at w=0) and `TestShapeContract`
  (shape, [0, 1] range under aggressive sharpen).
- 438 → 443 tests passing.

### Note on BPN Predictions

The diagnostic surfaced a secondary issue worth revisiting later: the
current checkpoint predicts `sigma_x` up to 35 (median 15) and `|w|`
up to 0.88, far outside the Stage 1 training range (sigma ∈ [0.3, 1.8],
|w| ≤ 0.4). The fixed 41-px kernel absorbs most of the damage by
truncating the Gaussian support, but the predictions still suggest
Stage 2 self-supervised training drifted to explain misalignment as
blur. With S2 refinement now handling geometric error, retraining BPN
(or optionally clamping sigma/|w| to Stage 1 range at inference) is a
worthwhile follow-up. Not done in this commit; `adv.yaml` leaves
`use_bpn: false` by default.

## 2026-04-18 — Alignment Refiner Moved from S5 to S2 (feat/mv_refine_to_s2)

### Why

The trained ROI alignment refiner predicts a residual homography ΔH
that corrects CoTracker drift between the reference and target
canonical ROIs. Previously applied at S5 (compositing), the correction
only benefited the final warp — S3 (text editing) and S4 (LCM, BPN)
still read the uncorrected `H_to_frontal`, so S4's per-pixel ratio map
between `ref_canonical` and `target_canonical` was computed on a
misaligned pair. Moving the refiner into S2 folds ΔH into
`H_to_frontal` / `H_from_frontal` up front, so every downstream stage
reads the corrected geometry automatically — no per-stage wiring.

### Direction Convention

ΔH is a forward homography in canonical pixel space mapping
ref-canonical → target-canonical (same contract as the training
dataset). At S2 we fold it in as:
- `H_to_frontal_corrected = inv(ΔH) @ H_to_frontal_unrefined`
- `H_from_frontal_corrected = H_from_frontal_unrefined @ ΔH`

Pinned by `test_refiner_direction_pinning` in
`test_s2_frontalization.py`: a 3-px canonical x-translate ΔH shifts a
frame-space midpoint's corrected projection by exactly 3 canonical
pixels in the opposite direction.

### FrontalizationStage

- `compute_homographies(track, frames=None)` — when
  `frontalization.use_refiner` is on and frames are provided, builds
  `ref_canonical` once per track, warps each target frame through the
  unrefined `H_to_frontal` to produce `target_canonical`, calls
  `RefinerInference.predict_delta_H`, and folds ΔH into both `H`
  matrices on the detection. Reference frame is skipped
- `run(tracks, frames=None)` — new `frames` kwarg threaded in from
  `pipeline.py`. Backward-compatible: TPM data gen pipeline's
  `s2.run(tracks)` still works because `frames=None` silently bypasses
  refinement
- Rejection counters logged at track-aggregate level (DEBUG by
  default, INFO when rejection rate ≥ `refiner_rejection_warn_threshold`)

### Configuration

- Add 8 `FrontalizationConfig.refiner_*` fields mirroring the S5
  shape: `use_refiner`, `refiner_checkpoint_path`, `refiner_device`,
  `refiner_image_size`, `refiner_max_corner_offset_px`,
  `refiner_rejection_warn_threshold`, `use_refiner_gate`,
  `refiner_score_margin`
- New validator rule: `frontalization.use_refiner` and
  `revert.use_refiner` cannot both be True — would double-correct
  the homography
- `adv.yaml` — refiner block moved to `frontalization:` with
  `use_refiner: true`; `revert.use_refiner` flipped to `false` while
  the S5 refiner block remains as fallback (no code removed)

### S5 and S4 Unchanged

- S5 refiner code path is intact — runs only when
  `revert.use_refiner: true` in config. Identical logic to before
- S5's existing composition `T @ H_from_frontal @ ΔH` naturally becomes
  `T @ H_from_frontal_corrected` when S2 does the correction — zero
  S5 code changes needed to consume the new geometry
- S4's `_warp_to_canonical` (stage.py:154) reads `det.H_to_frontal`
  directly, so the LCM ratio map now sees a pixel-aligned
  `(ref_canonical, target_canonical)` pair — the motivating quality
  win of this migration

### Testing

- 6 new `TestRefinerIntegration` cases in
  `test_s2_frontalization.py`: refiner-disabled backward-compat pin,
  ΔH folding direction pin, reference-frame skip, rejection-fallback
  to baseline, frames-omitted pathway (TPM use case), direction
  pinning against a known 3-px canonical translate
- 5 new `TestPipelineConfig` cases: not-both stage refiners allowed,
  S2-refiner-alone valid, S2 checkpoint required, S2 bad max corner
  offset, S2 bad rejection threshold
- Updated `test_from_yaml_adv_parses_refiner_fields` to assert
  refiner config now lives in `frontalization:` and `revert.use_refiner`
  is off
- 438 total tests passing

### Rollout Plan

See `docs/s2_refiner_migration_plan.md` §Rollout for the ablation
protocol — verify on `real_video15` + one harder video before trusting
S2 refinement as the default.

## 2026-04-18 — Hi-SAM Segmentation-Based Inpainter (feat/text_seg)

### New S4 Inpainter Backend
- Add `SegmentationBasedInpainter` — a second `BaseBackgroundInpainter` implementation that uses Hi-SAM for pixel-level text stroke segmentation, then fills the masked pixels with `cv2.inpaint` (Navier-Stokes by default, Fast Marching/Telea optional)
- Selected via `propagation.inpainter_backend: "hisam"` alongside the existing `"srnet"` and `"none"`
- Same BGR-uint8 in / BGR-uint8 out contract as `SRNetInpainter` — downstream LCM ratio-map code is unchanged
- Hi-SAM inference runs entirely inside the main `.venv` — zero new pip installs (torch, torchvision, einops, shapely, pyclipper, scikit-image, scipy, matplotlib, pillow, tqdm, opencv, numpy all already present)

### HiSAMSegmenter Wrapper
- `code/src/stages/s4_propagation/hisam_segmenter.py` (258 lines) wraps Hi-SAM's `SamPredictor` behind a clean `segment(bgr_roi) -> uint8 H×W {0, 255} mask` API
- Supports single-pass and sliding-window (`use_patch_mode`) inference via `_patchify_sliding` / `_unpatchify_sliding` helpers copied verbatim from upstream `demo_hisam.py` (avoids depending on that script as an importable module)
- `load_model()` uses `contextlib.chdir(third_party/Hi-SAM)` + `sys.path.insert` to work around Hi-SAM's `build.py` hardcoding a relative path to the SAM ViT encoder weights. The chdir is exception-safe (restores cwd on error) and scoped to construction only — cwd is unchanged by the time `segment()` is called
- Thread-safety caveat documented in the `load_model()` docstring: single-threaded only, since `chdir` mutates process-global state

### S3 Adaptive-Mask Inpainter Wiring
- `TextEditingStage._get_inpainter()` now dispatches `"hisam"` to `SegmentationBasedInpainter` in addition to the existing `"srnet"` branch. Before this fix, setting `inpainter_backend: "hisam"` silently disabled AnyText2's adaptive-mask flow (S3 only knew about `"srnet"`) and produced the long-to-short gibberish-fill regression
- Graceful fallback semantics match the `"srnet"` branch: missing checkpoint → warn + return None → AnyText2 falls back to non-adaptive mask

### S5 Pre-Composite Inpainter
- `RevertStage._get_pre_inpainter()` now dispatches on `revert.pre_inpaint_backend` (new field) accepting `"srnet"` or `"hisam"`
- Four dedicated `revert.pre_inpaint_hisam_*` fields (model_type, mask_dilation_px, inpaint_method, use_patch_mode) — kept independent from `propagation.hisam_*` so S4 and S5 can be tuned separately (e.g. aggressive dilation in S5 to scrub boundary leakage without widening S4's LCM-ratio inpaint). Mirrors the existing separation of `pre_inpaint_checkpoint` / `pre_inpaint_device` from propagation's equivalents
- Unknown-backend raises `ValueError` (vs S3's warn-and-skip): S5's pre-inpaint is an explicit opt-in via `pre_inpaint=true`, so misconfiguration should fail loudly

### Configuration
- Add four knobs to `PropagationConfig` (only used when `inpainter_backend: "hisam"`): `hisam_model_type` (default `"vit_l"`), `hisam_mask_dilation_px` (3), `hisam_inpaint_method` (`"ns"` or `"telea"`), `hisam_use_patch_mode` (false)
- Add five knobs to `RevertConfig`: `pre_inpaint_backend` (default `"srnet"` for back-compat), plus four `pre_inpaint_hisam_*` fields matching the propagation shape
- `adv.yaml` documents both backends with commented-out swap lines and example Hi-SAM checkpoint paths — flip one line to try Hi-SAM at either stage

### Installation
- Add `third_party/install_hisam.sh` mirroring `install_srnet.sh`: idempotent clone of the project's forked Hi-SAM repo plus idempotent downloads of the SAM ViT-L encoder weights (`wget` from Meta) and the SAM-TS-L TextSeg head (`gdown` from Google Drive)
- Both checkpoints land in `third_party/Hi-SAM/pretrained_checkpoint/` (Hi-SAM's expected directory)

### Review Fixes
- Converted 3 production-path `assert` statements to explicit `ValueError` / `RuntimeError` with actionable messages (survive `-O`, match SRNetInpainter's guard style)
- Added a NOTE comment in the smoke script clarifying that Hi-SAM inference runs twice per ROI (once for mask viz, once inside `inpaint()`) — don't time throughput from the script

### Testing
- 15 unit tests for `SegmentationBasedInpainter`: shape/dtype contract, dilation behavior, inpaint-method flag dispatch, input validation, lazy segmenter construction, config-field forwarding
- 7 unit tests for `HiSAMSegmenter`: lazy vs eager load, `load_model()` idempotency, cwd-restored guarantee, `segment()` precondition, binary-mask shape/dtype/values (both single-pass and patch modes) — CPU-safe via `monkeypatch.setitem(sys.modules, ...)` injection of fake `hi_sam.modeling.*` modules
- 5 wiring tests for `PropagationStage._get_inpainter()`: construction with all Hi-SAM kwargs, no-checkpoint graceful fallback, lazy init, caching across calls, unknown-backend raises with all three valid values listed
- 2 S3 regression tests for the AnyText2 adaptive-mask wiring: hisam-backend forwarding, hisam-missing-checkpoint graceful fallback
- 4 S5 pre-inpaint dispatch tests: srnet dispatch, hisam dispatch with full kwarg forwarding, unknown-backend `ValueError`, caching across calls
- 427 total tests passing (baseline 401 + 15 seg inpainter + 7 hisam segmenter + 5 S4 wiring + 2 S3 + 4 S5 pre-inpaint - ℘overlaps counted once)

### Misc
- `code/scripts/smoke_test_hisam_inpainter.py` for visual validation — runs Hi-SAM + cv2.inpaint on a set of canonical ROIs and writes 4-panel `(original | mask | inpainted | diff×3)` PNGs
- Live GPU smoke on 5 real ROIs from `test_output/roi_extraction_2/` (SAMSUNG, NORMAL, PERM PRESS, HEAVY DUTY, SHIRTS tracks) — inpainted output is flat, halo-free at default `mask_dilation_px=3`
- Update `docs/architecture.md`: expanded `s4_propagation/` module-map entry to list both inpainter backends; added Hi-SAM paragraph to Cross-Cutting Concerns covering vendored location, zero-new-pip-install story, and the chdir workaround

## 2026-04-10 — AnyText2 Adaptive Mask Sizing (fix/anytext2-adaptive-mask)

### Adaptive Mask for Long→Short Translations
- When translated text is much shorter than source (e.g. "WARDEN" → "典狱长", "Birthday" → "生日"), shrink the AnyText2 edit mask to match the target's natural width instead of using the full source-width mask
- Prevents AnyText2 from filling excess mask space with gibberish characters — a documented limitation across all mask-based scene text editing models
- Character-class width heuristic estimates target text width without font dependencies (CJK=1.0, Latin upper=0.60, Latin lower=0.50, digit=0.55, space=0.30)
- Configurable aspect tolerance (`anytext2_mask_aspect_tolerance`, default 0.15) skips the adaptive flow for close-enough translations (e.g. STOP→ALTO)

### Adaptive Canvas Crop
- After shrinking the mask, crop the canvas sent to AnyText2 to be centered on the mask with mask-proportional expansion (reuses `roi_context_expansion` ratio)
- Improves mask-to-canvas ratio from ~21% to ~62%, giving AnyText2 a tighter, better-proportioned canvas
- m1 (font mimic) input unchanged — still uses the full pre-adaptive ROI for complete font style extraction
- Fix latent bug: mimic mask array now uses mimic-prepared dimensions instead of main canvas dimensions

### SRNet Inpaint Artifact Cleanup
- Apply bilateral filter (`d=9, sigmaColor=75, sigmaSpace=75`) to SRNet-inpainted background before compositing, removing colored noise artifacts that polluted AnyText2's style
- Skip middle-strip restore: send fully clean (text-free) background to AnyText2 instead of restoring original text in the mask area. AnyText2 generates text from scratch guided by m1 font style reference

### Font Mimic Decoupling
- Separate m1 (font mimic) input from the main edit canvas: m1 uses the pre-adaptive ROI with full-width mask so AnyText2's font encoder sees complete source glyphs
- Main ref_img uses the adaptive hybrid canvas with shortened mask

### Configuration
- Add `text_editor.anytext2_adaptive_mask` (default true), `anytext2_mask_aspect_tolerance` (default 0.15) to `TextEditorConfig`
- Reuses `propagation.inpainter_backend` for the SRNet inpainter (separate instance from S4, lazy-loaded)
- Add `ANYTEXT2_DEBUG_DIR` env var to save AnyText2 server inputs for inspection

### Testing
- 9 unit tests for `compute_adaptive_crop_box` (centering, edge clamping, containment)
- 13 unit tests for `compute_adaptive_mask_rect` (tolerance, centering, min-ratio, skip cases)
- 17 unit tests for `estimate_target_width` (character classification)
- 12 unit tests for `restore_middle_strip` (feathering, edge clamping)
- 8 editor integration tests (adaptive trigger, tolerance skip, flag-off, no-inpainter fallback, exception fallback, crop canvas size, caller non-mutation)
- 3 S3 wiring tests (inpainter forwarded/skipped)
- 243 total tests passing (4 pre-existing S5 failures unchanged)

## 2026-04-08 — Expanded ROI with Scene Context (feat/expanded-roi)

### ROI Context Expansion
- Expand the region sent to AnyText2 with real scene pixels from the video frame, giving the model visual context for better style-matching text generation
- Compute expanded warp via translation matrix `T @ H_to_frontal` — reuses existing homography, no S2 changes needed
- AnyText2 mask targets only the text area within the expanded ROI — scene margins provide context but are not edited
- S3 crops the result back to canonical size — S4/S5 see no change

### Configuration
- Add `text_editor.roi_context_expansion` (default 0.0, recommended 0.3) to both `default.yaml` and `adv.yaml`
- Expansion ratio is automatically capped to avoid exceeding AnyText2's 1024px max dimension

### Code Quality
- `BaseTextEditor.edit_text()` ABC gains optional `edit_region` param (backward compatible)
- `PlaceholderTextEditor` respects `edit_region` — renders text only within the specified sub-area
- `AnyText2Editor._prepare_roi()` now returns scale factor for accurate mask coordinate mapping
- `mask_rect` bounds clamped to image dimensions for safety
- AnyText2 dimension constants (`MAX_DIM`, `MIN_DIM`, `ALIGN`) exported as public module-level names — S3 imports instead of duplicating

### Testing
- 15 new tests: `_clamp_expansion_ratio` (6), `_expanded_warp` (3), S3 expansion integration (2), `_prepare_roi` scale return (3), edit_region mask targeting (1)
- 186 tests passing (4 pre-existing failures: missing `wordfreq`)

## 2026-04-08 — BPN Training and TPM Integration into S4 (feat/bpn_model)

### BPN Training Framework (`code/src/models/bpn/`)
- Add Background/Blur Prediction Network from STRIVE: ResNet18 backbone with modified 12-channel first conv (ref + 3 neighbors), GAP + 2 FC head producing per-neighbor `(sigma_x, sigma_y, rho, w)` blur parameters
- Add `DifferentiableBlur` module implementing the oriented anisotropic Gaussian blur kernel from the paper, applied via grouped conv2d for batched per-sample kernels
- Add `BPNDataset` reading aligned ROI sequences from `tpm_dataset/`, with sliding-window sampling, random track subsetting, and a contiguous-array RAM cache that decodes once into a single uint8 ndarray to avoid copy-on-write blowup with DataLoader workers
- Use canonical reference frames from `s1_tracks.json`'s `reference_frame_idx` so every sample in a track shares the same (sharpest) reference instead of arbitrary first-frame-of-window
- Add two-stage training script: Stage 1 supervised on synthetic blur with known parameters, Stage 2 self-supervised reconstruction + temporal consistency on real video tracks
- Combined loss: per-parameter normalized L_psi (Stage 1 only), MSE reconstruction L_R via differentiable blur, and L_T temporal consistency penalty across consecutive neighbors
- Add evaluation script with reconstruction MSE / parameter statistics, training curve plotting, and visualization of one randomly-sampled sample per distinct track with multiple non-consecutive target frames per sample
- Linear warmup + cosine annealing LR scheduler, gradient clipping, periodic and best-val checkpointing, resume from checkpoint
- Initialization fix: small-weight init on the final FC layer so tanh/softplus start in their linear regime instead of saturating into identity output

### S4 Propagation: STRIVE TPM Integration (`code/src/stages/s4_propagation/`)
- Convert `s4_propagation.py` to a package layout (`s4_propagation/stage.py`) to host the multi-file TPM implementation
- Add `LightingCorrectionModule` (paper's LCM): per-pixel multiplicative ratio map between reference and target inpainted backgrounds, applied to the edited reference ROI. Supports log-domain computation, Gaussian smoothing, ratio clipping, distance-weighted neighbor averaging, and EMA temporal smoothing across consecutive frames
- Add `BaseBackgroundInpainter` ABC mirroring `BaseTextEditor` so future inpainters (LaMa, MAT) plug in without changing S4
- Add `SRNetInpainter` concrete backend wrapping `lksshw/SRNet`'s `Generator._bin` subnetwork only, handling the legacy checkpoint's gotchas: `weights_only=False`, lazy `sys.path` injection, RGB color order, `[-1, 1]` normalization, and resize to H=64 with W as a multiple of 8 to match the trained input shape
- Add `BPNPredictor` wrapping the trained BPN + DifferentiableBlur with `predict_params` (one batched forward pass per `n_neighbors` chunk, sigma rescaling from training resolution to inference pixel units) and `apply_blur` (single-image differential blur application)
- New `inpainted_background` field on `TextDetection` carrying the canonical-frontal text-removed ROI between the inpainter and LCM
- Restructure `PropagationStage.run()` into a two-pass loop: first pass collects per-detection lit ROIs (LCM if backgrounds available, else legacy YCrCb histogram matching as fallback), second pass optionally applies BPN differential blur per detection
- Lazy-load both inpainter and BPN on first run when their respective `use_*` flags are set, sharing the loaded models across all tracks/detections
- New `PropagationConfig` knobs: `use_lcm`, `lcm_*` (eps, ratio clip range, smoothing kernel, log-domain, EMA, neighbor self-weight), `inpainter_backend`/`inpainter_checkpoint_path`/`inpainter_device`, `use_bpn`, `bpn_checkpoint_path`/`bpn_device`/`bpn_n_neighbors`/`bpn_image_size`/`bpn_kernel_size`
- `adv.yaml` turns LCM and BPN on by default with the SRNet and BPN-Stage2-final checkpoints

### Smoke Tests
- `test_srnet_inpainter.py` runs SRNet on extracted ROIs and writes side-by-side `(original | inpainted | diff×3)` visualizations
- `test_s4_lcm_e2e.py` plants real ROIs at known quads with multiplicative brightness changes, runs both the histogram baseline and LCM+SRNet paths, and writes per-frame side-by-side comparisons
- `test_s4_bpn_e2e.py` plants ROIs with `cv2.GaussianBlur` at three different sigmas, runs both LCM-only and LCM+BPN, verifying the BPN second pass actually adds visible blur to the heavily-blurred targets

## 2026-04-08 — Test Reorganization & Real E2E Test (feat/anytext2-integration)

### Test Structure
- Reorganize flat 14-file `tests/` directory into tiered layout: `unit/` (6), `stages/` (6), `models/` (1), `integration/` (1)
- Add `pytest.ini` with `--ignore=tests/e2e` default — e2e never runs unless explicitly requested
- Register custom pytest markers: `slow`, `gpu`, `network` for selective test execution

### E2E Tests
- Add `tests/e2e/` with 4 real end-to-end tests exercising full pipeline (PaddleOCR + CoTracker + AnyText2 server) on GPU
- Auto-skip fixtures in `e2e/conftest.py`: gracefully skip on machines without GPU, AnyText2 server, or test video
- Assertions: output video integrity, track detection, non-degenerate AnyText2 ROI, output differs from input
- Run with: `cd code && python -m pytest tests/e2e/ -v`

### Test Counts
- 171 unit/stage/model/integration tests (1.3s)
- 4 real e2e tests (225s on V100)

## 2026-04-07 — AnyText2 ROI Quality Fix (feat/anytext2-integration)

### ROI Resolution & Mask
- Upscale small ROIs so `max(h,w) >= 512` (AnyText2's training resolution) — previously sent at native size (often 256×256 or smaller)
- Pad all dimensions to multiples of 64, matching AnyText2's SD VAE+U-Net architecture — prevents server-side silent pixel cropping via `resize_image()`
- Localize edit mask to the actual text content rectangle (`alpha=255`), padding regions are now anchored (`alpha=0`) — fixes black corner artifacts caused by the model regenerating replicated-border padding
- Crop result to content region before downscaling back to original ROI dimensions

### Configuration
- Add `text_editor.anytext2_min_gen_size` (default 512, range 256–1024) to control the upscale quality floor

### Testing
- Replace 4 old `TestClampDimensions` tests with 12 `TestPrepareRoi` tests covering: upscale, 64-alignment, content rect integrity, border replication, min/max clamping, extreme aspect ratios
- Add `test_localized_mask_written` verifying mask covers only content region
- 29 AnyText2 tests passing, lint clean

## 2026-04-07 — Replace googletrans with deep-translator (feat/anytext2-integration)

### Translation Backend
- Replace `googletrans-py` with `deep-translator` — fixes silent `NoneType` failures on certain inputs (e.g., "WARDEN")
- `GoogleTranslator` as primary backend with automatic `MyMemoryTranslator` fallback — both free, no API key needed
- `deep-translator` raises explicit exceptions (`TranslationNotFound`, `RequestError`) instead of crashing silently
- Update `TranslationConfig.backend` default and both YAML configs from `"googletrans"` to `"deep-translator"`
- 4 translation tests: success, blank-text short-circuit, Google→MyMemory fallback, both-fail-returns-source

## 2026-04-06 — AnyText2 Integration (feat/anytext2-integration)

### Stage A Model
- Integrate AnyText2 (ICLR 2025) as a real Stage A text editing backend, replacing the placeholder for cross-language scene text replacement
- `AnyText2Editor` subclass of `BaseTextEditor` communicates with an external AnyText2 Gradio server via `gradio_client`
- Supports style-preserving editing: uses "Mimic From Image" font extraction and auto-detected text color from the ROI
- Handles ROI dimension clamping (256-1024px range), auto-resize back to original dimensions

### Configuration
- Add `text_editor.server_url`, `server_timeout`, and AnyText2-specific params (`ddim_steps`, `cfg_scale`, `strength`, `img_count`) to `TextEditorConfig`
- `adv.yaml` defaults to `backend: "anytext2"` (server URL must be set per-environment)
- `default.yaml` keeps `backend: "placeholder"` for offline testing

### Testing
- 20 new unit tests for AnyText2Editor: color extraction, dimension clamping, edge cases, mocked Gradio calls, config validation, S3 integration
- All tests run without AnyText2 server (fully mocked)

### E2E Integration Fixes (2026-04-07)
- Fix RGBA mask format: AnyText2 reads edit region from alpha channel, not RGB
- Fix text_prompt quoting: AnyText2's `modify_prompt()` regex requires literal `"text"` wrapping
- Fix Gradio client API: use `submit()` + `job.result(timeout=...)` for gradio_client v2.4
- Fix gallery result parsing: handle image entry as string path (not nested dict)
- Fix CoTracker checkpoint paths: use `../third_party/...` for scripts running from `code/`
- Add connection timeout to Gradio `Client()` constructor via `httpx_kwargs`
- Send `m1` mimic image for proper font style extraction in "Mimic From Image" mode
- Fix typo in diffusion prompt (`"supper"` → `"super"`)

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
