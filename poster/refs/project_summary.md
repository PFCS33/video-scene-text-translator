# Content Brief — Cross-Language Scene Text Replacement in Video

> **Purpose.** Source of truth for every technical claim in the LaTeX
> report. Reconciled against `master` as of 2026-04-19. The final
> presentation slides (`_refs/ppt_final.pdf`) are partially stale —
> this document overrides them wherever they disagree. Built from a
> full read of `code/src/`, `code/config/`, `code/scripts/`,
> `docs/architecture.md`, `CHANGELOG.md`, and the git log since
> `pres/final-slides`.

---

## 0. Top-level framing

- **Project goal.** Replace scene text in video across languages
  (e.g., English "WARDEN" → Chinese "典狱长"), preserving font style,
  perspective, lighting, and motion/focus blur, while staying
  temporally stable.
- **Framing for the paper (engineering).** Existing systems fall into
  two buckets, neither of which solves video cross-language text
  replacement: (a) generative video models (Sora, Runway Gen-4 Aleph)
  have zero glyph-level control; (b) image scene-text-editing models
  (SRNet, MOSTEL, AnyText2, CLASTE) don't address temporal consistency
  or propagation across frames. The closest prior system, STRIVE, has
  never released code. Our response is a modular 5-stage pipeline that
  composes 7 ML models — 5 pretrained, **2 trained by us** (BPN and
  Alignment Refiner) — and shows via ablation that each module earns
  its place.
- **Team.** Hebin Yao (301624519), Yunshan Feng (301625263),
  Liliana Lopez (301653778). SFU, CMPT 743.

---

## 0.5. Tone and positioning (read before writing any section)

**Voice model — borrowed from the team poster abstract.** Prototype,
feasibility, *aiming to* preserve, modular, extends prior work. Not
"we propose a novel method that achieves state-of-the-art." The
professor has been explicit: this is a strong **course-project
baseline** in the glyph-controlled paradigm; the real-world frontier
is generative. The report should be honest about that positioning
while leading with what we built well.

**Positioning the pipeline against generative video models.** Frame
the design space as **two complementary paradigms**, never as a
competition where one wins:

| | Generative video edit (Sora, Gen-4 Aleph) | Ours (glyph-controlled pipeline) |
|---|---|---|
| What's easy for it | Naturalistic style, lighting, texture | Correct target glyphs; exact layout |
| What's hard for it | Glyph-level control; target-language accuracy | Matching the original's textural naturalism |
| Where it lives today | Frontier research / commercial previews | Modular, reproducible, open-pipeline |

This table is the mental model behind every section. We never
claim to "beat" Gen-4 Aleph; we solve a *different* sub-problem
(glyph correctness) with an *open* method, and we acknowledge the
generative paradigm's style superiority plainly.

**Where the honest "style vs correctness" point lands in the report.**
Distributed across four places so no single section carries the
full honesty load:

1. **Introduction §1** — introduce the two-paradigm landscape.
   Generative gets one clean paragraph acknowledging its
   style strength and glyph weakness. We position ourselves in the
   glyph-controlled lane as a *reproducible open baseline*.
2. **Related Work §2** — when Gen-4 Aleph / Sora are surveyed,
   their style quality is stated plainly. The glyph-control
   limitation is an *open problem* in that paradigm, not a flaw.
3. **Experiments §4 qualitative subsection** — side-by-side frame
   triplets with a neutral caption that names both directions of
   the tradeoff. Proposed caption:
   > *"Generative systems such as Gen-4 Aleph (center) excel at
   > photorealistic style preservation but produce incorrect glyphs
   > when prompted for a specific translation, yielding unreadable
   > or gibberish text. Our pipeline (right) produces correct target
   > glyphs through explicit compositing, at the cost of reduced
   > style-transfer quality. The two approaches currently address
   > different subproblems of video text localization."*
4. **Conclusion §5** — close with the honest course-project framing
   reshaped positively: our contribution is a **reproducible,
   modular, glyph-controlled baseline** that future hybrid systems
   can build on. The generative paradigm is likely to dominate as
   glyph control matures; until then, modular pipelines provide
   predictable correctness. Future direction: hybrid (generative
   for style, ours for glyph control).

**Soft-tone writing rules.**
- Prefer *aims to*, *approximates*, *prototype*, *feasibility
  study*, *strong baseline*, *reference implementation*.
- Avoid *novel*, *state-of-the-art*, *superior*, *outperforms*.
- Weaknesses become **future directions**, not bugs. Instead of
  "the BPN's rho collapses to zero," write "improving BPN's rotation
  signal is a promising next step."
- Limitations subsection stays short and matter-of-fact (one short
  paragraph, not a bulleted litany). Long failure-case content that
  used to live in §I becomes structured around "what we'd improve
  next" rather than "what's broken."
- Never compare ourselves favorably to Gen-4 Aleph on naturalism.
  Only on glyph correctness and openness.
- Avoid defensive phrasing ("admittedly," "we recognize that,"
  "of course"). State the tradeoff neutrally and move on.

**Abstract style reference** (from the team poster, not academic but
useful calibration): "prototype end-to-end system... combines... in a
five-stage modular pipeline... results on real video examples show
the feasibility of this approach... while also revealing remaining
challenges in text tracking, text generation, and frame-to-frame
stability." Mirror this tone — plural "we," humble claims, honest
about remaining challenges, focused on feasibility over performance.

---

## A. Current pipeline, stage by stage

### Stage 1 — Detection, Tracking, Translation, Reference Selection

**Purpose.** Find text in every Nth frame via OCR, group detections
into tracks, translate each track's text, pick one reference frame
per track, and propagate the quad to every frame via optical flow.

**Inputs / outputs.** Video frames → `list[TextTrack]`.

**Key components.**
- **OCR backend** (`s1_detection/detector.py`). EasyOCR (default in
  `default.yaml`) or PaddleOCR (default in `adv.yaml`, the ship
  config), selected via `detection.ocr_backend`. PaddleOCR is faster
  and more accurate on natural scenes.
- **Gibberish filter.** `wordfreq` Zipf-frequency threshold suppresses
  false OCR detections that look plausible but aren't words.
- **Optical flow** for gap-fill (`detection.optical_flow_method`):
  Farneback (CPU, classical, fallback) / Lucas-Kanade / **CoTracker3
  online** (default in `adv.yaml`, GPU, chunked 60-frame sliding
  windows, ~25× faster than Farneback, much smoother).
- **Tracking.** IoU-based greedy match with configurable break
  threshold + text similarity check (`s1_detection/tracker.py`).
  Tracks serialized as dict keyed by `frame_idx` for O(1) lookup.
- **Translation** (`s1_detection/selector.py`). `deep-translator` is
  the default backend. GoogleTranslator first, MyMemory fallback with
  short→locale mapping (`en`→`en-GB`, `zh-CN` preserved). Optional
  `google-cloud-translate` via API key. Lazy-loaded.
- **Reference selection.** STRIVE-aligned 4-filter funnel:
  1. Keep detections with max text length in the track.
  2. OCR confidence ≥ `ref_ocr_min_confidence` (0.7).
  3. Top-K by sharpness (Laplacian variance), K=10.
  4. Composite score = 0.7 × contrast (Otsu inter-class variance) +
     0.3 × frontality (quad area / bbox area). Max wins.
     Falls back to all detections if pre-filters eliminate everything.
     Honors `max_frame_offset` for CoTracker-online mode (reference must
     land in the first sliding window).

### Stage 2 — Frontalization + (NEW) Alignment Refinement

**Purpose.** For each detection, compute a homography to a
**canonical frontal rectangle** (axis-aligned, same size across the
track, derived from the reference quad's edge lengths). As of commit
`930eb97` (post-presentation), this stage **also applies a learned
residual homography correction** (`ΔH`) to fix CoTracker drift.

**Inputs / outputs.** TextTracks with quads → same TextTracks with
`H_to_frontal` / `H_from_frontal` / `canonical_size` stamped onto each
detection. Pure geometry — no pixels warped here.

**Baseline homography.** `findHomography(..., RANSAC)` with
reprojection threshold 5.0 px; optional multi-point fit using
CoTracker grid points for better conditioning on slightly non-planar
surfaces.

**Alignment refiner (new, replaces the S5 refiner).**
- Inference wrapper `RefinerInference` around `ROIRefiner` network
  (see Section B.2 for architecture + training).
- Extracts reference canonical ROI once per track, target canonical
  ROI per detection, concatenates along channel dim → network
  predicts 4-corner offsets → converts to `ΔH` via a differentiable
  DLT.
- **Fold into warp chain:**
  `H_to_frontal_refined = inv(ΔH) @ H_to_frontal_base`;
  `H_from_frontal_refined = H_from_frontal_base @ ΔH`.
  (Direction-pinned by `test_refiner_direction_pinning` in
  `code/tests/stages/test_s2_frontalization.py`.)
- **Gate.** Reject predictions with max corner offset > 16 px
  (`refiner_max_corner_offset_px`). Log at INFO if ≥ 10% of track
  predictions are rejected. Rejected detections silently fall back to
  baseline `H_to_frontal_base`.
- **Invariant:** `config.py` validator forbids
  `frontalization.use_refiner` and `revert.use_refiner` both true —
  prevents double-correction.

**Key config.** `frontalization.use_refiner`,
`refiner_checkpoint_path` (`checkpoints/refiner/refiner_v1.pt`),
`refiner_image_size` (`(64,128)` to match training),
`use_refiner_gate`, `refiner_score_margin`.

### Stage 3 — Text Editing (Stage A integration point)

**Purpose.** Extract the reference frame's canonical ROI; send it
through a text-editing model to get the translated version with
style preserved.

**Inputs / outputs.** TextTrack + frame dict → TextTrack with
`edited_roi` (BGR ndarray).

**ROI extraction.** Warp via `H_to_frontal` (preferred); bbox crop
fallback if homography invalid.

**Context expansion.** `roi_context_expansion` (default 0.3 = 30% per
side) grows the canonical crop with surrounding scene pixels before
sending to AnyText2, then crops the result back. Improves style
matching (lighting / texture continuity at edges).

**Backend.** `BaseTextEditor` ABC swappable via `text_editor.backend`.
- **`anytext2`** (default in `adv.yaml`) — diffusion-based
  (SD 1.5 backbone + WriteNet + AttnX), multilingual. Accessed over
  HTTP via a Gradio server (separate process, URL configurable as
  `text_editor.server_url`).
- **`placeholder`** — Pillow text rendering for unit / pipeline tests.
  Accented characters supported via PIL.
- **`stage_a`** — placeholder that raises `NotImplementedError`
  (reserved for RS-STE integration, not shipped).

**Adaptive mask for long→short translation** (`models/anytext2_mask.py`).
The known failure mode of mask-based STE models: when the target
string is significantly shorter than the source (e.g., 7 CJK chars →
3), AnyText2 fills the unused mask space with hallucinated gibberish.
Our fix:
1. Estimate target text's natural width using a
   **character-class-based heuristic** (CJK 1.0, Latin upper 0.60,
   Latin lower 0.50, digit 0.55, space 0.30 — widths relative to
   height).
2. If `|target_aspect - source_aspect| / source_aspect ≥ 0.15` **and**
   target is narrower (only long→short triggers), pre-inpaint the
   canonical ROI with the configured background inpainter (SRNet or
   Hi-SAM), warp it in, then restore a feathered centered middle
   strip of original pixels matching the target's natural width.
3. Send this hybrid ROI + a shrunk mask to AnyText2.
4. S3 lazy-loads a **separate** inpainter instance, independent from
   S4's — reuses `propagation.inpainter_backend` config but decoupled
   lifecycle.

**Key config.** `anytext2_adaptive_mask` (default true),
`anytext2_mask_aspect_tolerance` (0.15), `anytext2_ddim_steps`,
`anytext2_cfg_scale`, `anytext2_strength`, `anytext2_min_gen_size`
(512 px — small ROIs upscaled first), `server_url`, `server_timeout`.

**Engineering note.** Editor calls wrapped in try/except with region
context (track id, region index) and explicit entry/exit timing logs
— surfaces silent Gradio hangs without aborting the whole pipeline.

### Stage 4 — Propagation (LCM + BPN)

**Purpose.** Adapt the edited reference ROI to each frame's lighting
(Lighting Correction Module) and motion/focus blur (Blur Prediction
Network); build per-frame feathered alpha masks for S5. Implements
the first two-thirds of STRIVE's Text Propagation Module.

**Inputs / outputs.** TextTrack with `edited_roi` + frame dict →
`dict[frame_idx → list[PropagatedROI]]`.

**Pass 1 — Lighting correction** (`use_lcm: true` by default).
Requires inpainted backgrounds from both ref and target frames:
1. Inpaint text away from ref canonical and each target canonical
   using the configured inpainter (SRNet or Hi-SAM — see §D).
2. Compute per-pixel ratio map
   `r = (I_target_bg + ε) / (I_ref_bg + ε)` (log-domain arithmetic
   for stability), smooth with a Gaussian (kernel 9), optionally
   temporally EMA-smoothed.
3. Apply to edited ROI: `I_out = I_edited × r`.

**Fallback path — histogram matching.** When LCM is disabled or
backgrounds are unavailable, use YCrCb-Y histogram matching of edited
against target. Less accurate for spatially varying light but robust.

**Pass 2 — BPN** (`use_bpn: true` by default; requires `use_lcm`).
Predicts 4 blur parameters per target frame and applies a
differentiable oriented Gaussian. See §B.1 for architecture.
`I_out_blurred = (1 + w) · I_edited − w · (I_edited ⊛ G_{σx,σy,ρ})`.
Positive `w` blurs, negative sharpens (clamped at inference).

**Feathered alpha mask.** Center = 1.0, linear fade to 0.0 over the
outer ~10% of each dimension. Used by S5 for smooth compositing.

**Key config.** `use_lcm`, `use_bpn`, `inpainter_backend`
(srnet/hisam/none), `bpn_checkpoint_path`
(`checkpoints/bpn/bpn_v1.pt`), `bpn_n_neighbors` (3),
`bpn_image_size` (`(64,128)` — must match training),
`bpn_kernel_size` (41).

**Engineering notes.** Lazy loading of inpainter and BPN to avoid
startup cost when disabled. 30-second heartbeat log inside the
per-ROI inpaint loop to surface slow tracks. Two-pass: LCM for all
detections first, then BPN.

### Stage 5 — Revert (De-Frontalization + Compositing)

**Purpose.** Warp propagated ROIs back to each frame's perspective
and composite into the original frame with seamless blending.

**Inputs / outputs.** Frame dict + TextTracks + PropagatedROIs →
final frame ndarrays.

**Inverse warp.** `cv2.warpPerspective` using `H_from_frontal` from
S2, but only to the target quad's bbox (expanded 5%, clamped to
frame) — avoids warping a black full-frame image.

**Temporal smoothing** (optional, default off). Gaussian-weighted
smoothing of the projected quad corners across a 7-frame window,
sigma 10. Operates on corner coordinates, not raw 3×3 matrices, so
averaging is geometrically meaningful.

**Pre-inpaint** (S5-only, `pre_inpaint: true` by default). Before
compositing, inpaints the target frame's expanded bbox region
(`pre_inpaint_expansion: 0.15`) to erase residual source-text
boundaries. Uses an **independent inpainter instance** with its own
backend config — can differ from S4 (e.g. S4=SRNet, S5=Hi-SAM).

**Blending.** Default feathered alpha blend
(`frame × (1 − α) + roi × α`). Poisson seamless-clone path exists
(`cv2.seamlessClone`) but is off by default; requires strict interior
mask and is overkill when feathering is working.

**Refiner (fallback path).** S5 refiner code is still present but
**disabled by default** (`revert.use_refiner: false`). The refiner
was moved to S2 in `930eb97`; S5 refinement is kept as an alternative
deployment mode and guarded by the validator against concurrent S2
refinement. Diagnostic flag `_REFINER_DIAGNOSTIC_BLUE` paints where
unrefined would have landed (dev-only).

---

## B. The two models we trained

### B.1 BPN — Blur Prediction Network

**Role.** Per-frame differential blur parameter prediction for the
edited ROI.

**Architecture** (`models/bpn/network.py`).
- Input: `(B, 3·(N+1), H, W)` — reference canonical + N neighbor
  canonicals concatenated along channels. Default N=3, resolution
  (64, 128) → input channels = 12.
- Backbone: ResNet18 with its first conv replaced to accept 12
  channels (weights tiled from the ImageNet-pretrained 3-channel
  first conv across the expanded fan-in).
- Head: GAP → FC(512→256, ReLU, Dropout) → FC(256→4N).
  Activations: softplus on σ, tanh on ρ and w.
- Output: `{σ_x, σ_y, ρ, w}` each shape `(B, N)`.
- Params: ~1.24M trainable end-to-end.

**Differentiable blur** (`models/bpn/blur.py`).
`I_out = (1 + w) · I_ref − w · (I_ref ⊛ G(σ_x, σ_y, ρ))` where `G` is
a rotated 2D anisotropic Gaussian. Grouped convolution per sample, so
kernels don't leak across batch items.

**Post-presentation fix — padding mode.** Commit `7a6b1f1` switched
`F.pad(..., mode="reflect")` → `mode="replicate"`. Reflect requires
`pad < input_size` in each dim; with `kernel_size=41` (pad=20) and
real-world tracks as small as 10–15 px tall, reflect crashes.
Replicate has no such constraint and yields equivalent halo
suppression. Training inputs are 64×128 (above threshold) so training
is unaffected — this is an inference-time robustness fix.

**Training scheme (two-stage).**
- **Stage 1 — supervised on synthetic blur.** Take a clean canonical
  ROI from the training dataset, apply a known oriented Gaussian
  blur with random `(σ_x, σ_y, ρ, w)`, learn to regress the
  parameters. Loss: MSE on parameters. Prevents overfitting to noisy
  real data and teaches the parameter space.
- **Stage 2 — self-supervised on real frames.** Two frames from the
  **same track** (no time-gap constraint — deployment pairs are
  arbitrary frames, often hundreds apart, so training matches that).
  Loss: Charbonnier reconstruction between ref blurred by the
  predicted kernel and the target + temporal consistency term across
  adjacent frame pairs.

**Training data.** Pre-rendered canonical ROIs from the TPM data-gen
pipeline (`tpm_data_gen_pipeline.py`), laid out as
`{video}/{track}/frame_{idx:06d}.png`. 31 train / 2 val videos
configured in `models/bpn/config.yaml`. `cache_in_ram: true`
preloads the entire train set (~87 GB at 64×128) to eliminate disk
I/O during Stage 2.

**Post-presentation retraining.** Commits `156844a` + `81caa3d`:
dataset regenerated using **S2-refined homographies**
(`corrected_track_info.json` written alongside `s1_tracks.json`) so
canonical ROIs are pixel-aligned. Pre-refinement, real-pair training
was struggling to distinguish misalignment from blur — everything
looked like blur. Post-refinement, the signal is clean.

**Current state.** `bpn_v1.pt` shipping, `use_bpn: true` in
`adv.yaml`. Evaluation shows predicted `σ` distributions matching the
Stage-1 prior and plausible `w`.

**Known BPN limitations.** Weak rotation signal — `ρ` frequently
collapses to ~0 axis-aligned even with rotated blur ground truth.
Occasional `w < −1` prediction (sharpening instead of blur); clamped
at inference but indicates undertraining in that regime.

### B.2 Alignment Refiner — Residual Homography Corrector

**Role.** Predicts `ΔH` to correct CoTracker residual drift
(typically 4–8 px) in the 4 tracked corner points, so downstream
stages operate on pixel-aligned canonical ROIs. Moved from S5 to S2
post-presentation.

**Architecture** (`models/refiner/network.py`).
- Input: `(B, 6, H, W)` — reference canonical + target canonical
  concatenated along channels. Resolution (64, 128).
- Body: 4 Conv-BN-ReLU blocks, each stride 2 (channels 6→32→64→96→128,
  total 16× downsampling).
- Head: Flatten → FC(4096→256, ReLU, Dropout 0.2) → FC(256→8)
  (small-init σ=1e-3 so the initial residual is near zero).
- Output: `(B, 4, 2)` corner offsets `(dx_i, dy_i)`.
- Params: ~1.237M (most in the 4096→256 FC).

**Differentiable DLT** (`models/refiner/warp.py`). Converts the 8
offsets into a 3×3 `ΔH` via a linear system, so gradients flow
through the geometric parameterization.

**Training scheme (mixed-batch, two-stage).**
- **Type A — synthetic pairs.** Take a clean ROI, apply a random
  ±8 px corner perturbation to define the "target," learn to predict
  the corners back to identity. Loss: masked Charbonnier on RGB +
  small corner L2 regularization. Ground truth is known exactly.
- **Type B — real pairs.** Two random distinct frames from the
  **same track** (no time-gap constraint — deployment aligns a
  possibly-frame-0 reference against arbitrary target frames).
  Loss: masked NCC on luminance (shift/scale invariant) + masked
  Charbonnier on Sobel magnitude (preserves structure) + corner L2.
- **Curriculum.** `RefinerDataset.__getitem__` picks sample type per
  item based on `real_pair_fraction` — a mutable attribute updated
  by the training loop's schedule. Stage 1: 100% synthetic. Stage 2:
  70% real / 30% synthetic, resumed from Stage 1 best.
- Single optimizer, single dataloader, one checkpoint lineage.

**Training data.** Same TPM data-gen layout as BPN, 28 train / 6 val
videos. Canonical ROIs resized to (64, 128).

**Current state.** `refiner_v1.pt` shipping;
`frontalization.use_refiner: true` in `adv.yaml`.

**Known refiner limitations.** Per-frame prediction, no temporal
smoothing on `ΔH` — small inter-frame oscillation remains. Training
doesn't see cross-track mis-initializations (assumes residual
error), so large baseline errors are out of distribution.

---

## C. Inpainter backends

Three call sites, three **independent** inpainter instances (each
lazy-loaded, each with its own config subsection):
1. **S3** adaptive-mask pre-inpaint (long→short).
2. **S4** LCM background isolation.
3. **S5** pre-compositing edge cleanup.

**SRNet** — learned text-removal net, ~500 MB checkpoint. Original
choice; preserved in code. Slower, highest quality on trained style
distribution.

**Hi-SAM** (default in `adv.yaml`, added post-presentation in
`feat/text_seg`, merge `25a50cd`). Pipeline:
1. `HiSAMSegmenter.segment()` — SAM-based stroke-level text
   segmentation returns a binary mask.
2. Optional dilation (`hisam_mask_dilation_px`, default 3).
3. `cv2.inpaint` (Navier-Stokes or Telea) fills the masked region.

Vendored at `third_party/Hi-SAM/`. No extra `pip install` needed
beyond the main venv. Wrapper uses `contextlib.chdir` during
construction because upstream `build.py` hardcodes a relative path
to SAM ViT encoder weights; scoped to init, cwd restored afterward.
Optional 512×512 patch-mode for very large ROIs.

**Tradeoffs.**

| | SRNet | Hi-SAM |
|---|---|---|
| Needs learned weights beyond SAM? | Yes (~500 MB) | No |
| Speed | Slower | Faster |
| Style generalization | Training-distribution only | Generic (SAM) |
| Failure mode | Residual faint strokes | Misses thin strokes / shadows |
| Mitigation | — | Mask dilation (3 px) |

**Config:** `propagation.inpainter_backend` ∈ {`srnet`, `hisam`,
`none`}, plus backend-specific fields (`hisam_model_type` vit_b/l/h,
`hisam_inpaint_method` ns/telea, etc.).

---

## D. OCR & Tracking details

- **PaddleOCR** (PP-OCRv4) is the default detector. Returns quads +
  recognized strings + confidences. Wrapped behind the same interface
  as EasyOCR (`detector.py`) so the backend is a config switch.
- **CoTracker3 online** (`stages/s1_detection/cotracker_online.py`
  + `streaming_tracker.py`) chunks video into 60-frame sliding
  windows (`detection.cotracker_window_len`). Auto-falls-back to
  pairwise Farneback for tracks <120 frames. Requires GPU.

---

## E. Translation details

- Primary: `deep-translator.GoogleTranslator(source, target)`.
- Fallback: `deep-translator.MyMemoryTranslator` with the short→locale
  map `_MYMEMORY_LOCALE` (commits `1819531` + `865e74c` + `7a959cb`
  addressed zh-CN casing and the fact that some short codes must be
  expanded to `xx-YY` for MyMemory's API).
- Optional: `google-cloud-translate` (API-key path).
- Language codes: ISO-639 (`en`, `zh-CN`, `es`, ...).
- Lazy-initialized on first translate call.

---

## F. Post-presentation changes (the deltas from the slides)

The final presentation deck does not reflect these four changes.
Everything in the paper's methodology section should use the
post-change state as ground truth.

| # | Change | Commit(s) | Effect on paper |
|---|---|---|---|
| 1 | Alignment Refiner moved **S5 → S2** | `930eb97` (+ `feat/mv_refine_to_s2`) | Describe refiner under S2 (Methodology § S2). S5 mentions it only as a disabled fallback. |
| 2 | BPN **retrained on S2-aligned data**, enabled by default | `156844a`, `81caa3d`, `85365f2` | BPN training section explains why pre-alignment matters (sim-to-real signal). |
| 3 | BPN padding `reflect → replicate` | `7a6b1f1` | One-line engineering note in BPN subsection. |
| 4 | **Hi-SAM** inpainter added as alternative to SRNet; default in `adv.yaml` | `feat/text_seg` merge `25a50cd`, `c64e1e8` | Inpainter discussion should present both backends with the tradeoff table above. |

Out-of-scope deltas (do **not** appear in paper): web client
(`feat/web-client` merges), stage-liveness watchdog
(`stage-liveness-observability` session), TPM data-gen streaming
pipeline (mentioned once as supporting tool for BPN training).

---

## G. Scope exclusions (explicit)

These exist in the repo but are **not mentioned in the paper body**:
- `server/` FastAPI backend, `web/` React client — used only as the
  hero-figure screenshot.
- Stage-liveness watchdog (`server/app/_liveness.py`).
- TPM data-gen pipeline — mentioned only as the tool that produced
  BPN + Refiner training data.
- Checkpoint fetch scripts.

---

## H. Quantitative metrics — what we can claim

No end-to-end metric computation code exists in the repo. Planned
claims for the paper, with placeholders to fill in:

1. **OCR readback accuracy** — run PaddleOCR on the translated output
   video, compare to the target string. Primary metric. Needs
   external evaluation script (easy to write — ~50 LoC).
2. **SSIM / PSNR on non-text regions** — between source and output,
   masked by the inverse of the text alpha. Shows the pipeline
   doesn't damage surroundings. Secondary metric.
3. **Temporal Δquad jitter (px)** — per-frame corner displacement
   delta, with vs. without Alignment Refiner. Shows the refiner
   reduces flicker.
4. **Per-stage runtime** (seconds, 1× consumer GPU) — already logged
   by the pipeline. Engineering legitimacy.

Existing internal metrics (from training-side code) not reported in
paper:
- Refiner: corner endpoint error, photometric NCC/Charbonnier,
  visualization grids (`models/refiner/evaluate.py`).
- BPN: parameter distribution plots, kernel visualizations
  (`models/bpn/evaluate.py`).

---

## I. Current limitations & directions for future work

Per the tone guide in §0.5, this section in the paper is **short**
and framed as growth directions, not a bug list. Only three
limitations are promoted to the paper body; the rest are absorbed
into the Future Work subsection of the Conclusion or simply omitted.

**Paper-body limitations (one short paragraph, ~3-4 sentences):**

1. **Style-transfer gap vs generative baselines.** Our compositing
   pipeline produces correct target glyphs but does not match the
   naturalistic style preservation of end-to-end generative models
   such as Gen-4 Aleph. This is the fundamental tradeoff of the
   glyph-controlled paradigm and our most important acknowledged
   limitation.
2. **Temporal stability on long sequences.** Each frame's homography
   is estimated independently; the Alignment Refiner reduces but
   does not fully eliminate inter-frame micro-oscillation. Future
   work: temporal smoothing on predicted `ΔH` corners.
3. **Long-to-short translation residual.** The character-class-width
   heuristic covers most cases, but very stylized fonts can exceed
   the 15% aspect-mismatch tolerance and leave faint artifacts.

**Future-work directions (in Conclusion):**

- Hybrid generative + glyph-controlled compositing (use a generative
  model for the background/style layer, our pipeline for the text
  layer).
- Learned temporal smoothing on `ΔH` corners.
- Better blur-rotation signal on real data (BPN training on richer
  oriented-blur pairs).
- Streaming main-pipeline to handle long videos (>500 frames) —
  architecture already exists in the TPM data-gen tool.

**Internal-only (keep out of paper unless asked):**

- Gap-fill propagating to text-absent frames
- Small-ROI degenerate homographies
- Translation rate-limit on fallback backend
- Hi-SAM missing thin strokes / shadows (mitigated by 3 px mask
  dilation already — no paper-level action needed)

---

## J. Architectural decisions worth naming in the paper

These are "good taste" decisions worth one sentence each:
1. Central `TextTrack` dataclass flowing through all 5 stages.
2. Canonical frontalization as shared working space (vs frame-space
   per-stage).
3. Store homography matrices, warp on-demand (memory-efficient).
4. Pluggable Stage A via `BaseTextEditor` ABC.
5. Lazy initialization for every expensive resource.
6. Config-driven — two YAMLs (`default.yaml` classical,
   `adv.yaml` learned) let us A/B every module.
7. Feathered alpha blending over Poisson by default — simpler,
   good enough with proper feathering.
8. Three independent inpainter instances (S3/S4/S5) — different call
   sites, different optimal configs.

---

## K. Report-section mapping

| Paper section | Drawn from brief sections |
|---|---|
| Abstract | §0 + §0.5 tone guide + §F top row (post-pres improvements) |
| 1. Introduction | §0 + §0.5 two-paradigm framing (our most important intro move) |
| 2. Related Work | §A.3 (AnyText2), §C (SRNet, Hi-SAM), §D (PaddleOCR, CoTracker3), + generative video (Sora, Gen-4 Aleph) + video STE (STRIVE unreleased) |
| 3. Methodology | §A (5 stages, S2 now includes refiner, Hi-SAM as default inpainter) + §B (two trained models full detail) + §C (inpainter tradeoff table) |
| 4. Experiments & Results | §H metrics with placeholders + §0.5 qualitative comparison subsection (complementary, not competitive) + selected ablations (LCM on/off, BPN on/off, Refiner on/off) |
| 5. Conclusion & Future Work | §0.5 honest-but-positive closing (open reproducible baseline; generative paradigm likely dominant; hybrid direction) + §I future-work list + §J design decisions |

---

## L. Resolved decisions (formerly open questions)

| # | Decision | Resolution |
|---|---|---|
| 1 | Gen-4 Aleph comparison framing | **Complementary, not competitive.** See §0.5. Use the WARDEN→典狱长 example + the highway sign. Caption explicitly names the tradeoff. |
| 2 | Ablation data | Structure in §M.3 is ready with placeholders. Real frames to be provided by user when available — swap is mechanical. |
| 3 | Quantitative numbers | All four metrics (§H) are placeholders (`XX.X`) in the draft. User fills in real numbers after runs; unavailable metrics silently dropped. |
| 4 | Hero image | Web UI screenshot placeholder at `figures/hero_webui.png`. User provides real screenshot when ready. Body never mentions the web client itself. |
| 5 | Team contributions paragraph | Not included by default. Samples 1/3 don't have one; sample 2 has only author list. Can add in §0.5 polish pass if requested. |
| 6 | Team identifiers | Cover-page footer / author block lists names + SFU IDs (see §0). |

---

## M. Figure placeholder template

**Why a template.** The draft will reference ~8-10 figures. Every
figure slot uses the same LaTeX macro so swapping in real images
later is mechanical (no caption / label hunting).

### M.1 LaTeX macro

Define once in `main.tex`:

```latex
\newcommand{\figplaceholder}[4]{%
  % #1 = filename (without extension), in figures/
  % #2 = width (e.g., \linewidth, 0.7\linewidth)
  % #3 = caption
  % #4 = label (e.g., fig:pipeline)
  \begin{figure}[t]
    \centering
    \IfFileExists{figures/#1.pdf}%
      {\includegraphics[width=#2]{figures/#1.pdf}}%
      {\IfFileExists{figures/#1.png}%
        {\includegraphics[width=#2]{figures/#1.png}}%
        {\fbox{\parbox{#2}{\centering\vspace{1.2cm}%
          \textit{\textbf{PLACEHOLDER:} #1}\\[4pt]%
          \textit{Figure to be inserted — see caption below.}%
          \vspace{1.2cm}}}}}%
    \caption{#3}
    \label{#4}
  \end{figure}}
```

Usage example:
```latex
\figplaceholder{hero_webui}{\linewidth}%
  {Our prototype web interface showing the end-to-end video
   localization flow: input video, source/target language
   selection, and the translated output.}%
  {fig:hero}
```

Behavior: shows the real image if `.pdf` or `.png` exists; otherwise
renders a visible framed "PLACEHOLDER" box with the filename. No
compile errors. Easy visual audit of what's still missing.

### M.2 Required figures (planned slots)

| Slot | Filename | Section | Role | Recommended layout |
|---|---|---|---|---|
| **F1 — Hero** | `hero_webui.png` | Page 1 (after abstract) | Visual convey "this is a working system." Web UI screenshot showing input video + output video + stage progress. Body never refers to the web UI. | Full-column width, 16:9 aspect, brief caption naming input/output languages. |
| **F2 — Pipeline diagram** | `pipeline.pdf` | §3 opener | The 5-stage diagram. Copy of `_refs/pipeline-pic.png`; ideally regenerated as vector PDF for print quality. | Full-column width, tight bounding box. Caption lists S1–S5 with one phrase each. |
| **F3 — Adaptive mask (S3)** | `adaptive_mask_before_after.png` | §3 S3 subsection | Long→short example. Left: source canonical ROI with 7 CJK chars. Middle: hybrid canvas after pre-inpaint + middle-strip restore. Right: AnyText2 output. | 3-panel horizontal; each 1/3 col; annotated labels "source / hybrid / output". |
| **F4 — BPN effect (S4)** | `bpn_ablation.png` | §3 BPN subsection or §4 ablation | Two rows: (top) output without BPN — visibly over-sharp against blurry surroundings. (bottom) output with BPN — matches frame blur. 3-4 frames per row. | Full-column width, 2 rows × 4 thumbnails. Caption explicitly names the metric. |
| **F5 — Refiner effect (S2)** | `refiner_jitter.png` | §3 refiner subsection or §4 ablation | Corner-trajectory plot (x/y vs frame index) with vs without refiner. Shows jitter reduction. Alternatively: heatmap of residual alignment error. | Half-column figure; two-line plot. Y-axis: pixels. |
| **F6 — Gen-4 Aleph comparison** | `comparison_warden.png` | §4 qualitative | Source | Gen-4 Aleph output | Ours, 3 panels. The WARDEN→典狱长 example from the PPT. | Full-column width, 3-panel horizontal. Caption is the one in §0.5 item 3. |
| **F7 — Second qualitative example** | `comparison_highway.png` | §4 qualitative | Highway sign (the second PPT example). Same 3-panel format as F6. | Full-column width. |
| **F8 — LCM ablation** | `lcm_ablation.png` | §4 ablation | Output with LCM on vs off in a scene with clear lighting variation. | Half-column; 2 thumbnails side by side. |
| **F9 — Failure case** | `failure_long_to_short.png` | §4 failure subsection (optional) | Example of a stylized font where adaptive mask leaves a residual. Honest but one panel only — not dwelled on. | Half-column; single thumbnail. |
| **F10 — Model involvement table** | *not a figure — LaTeX table* | §5 | Per-model row: pretrained / trained by us. From PPT slide 11 updated per §F. | Full-column table. |

### M.3 Ablation presentation

Proposed compact **results table** combining the three ablations
into one place (populated with placeholder numbers):

| Config | OCR readback ↑ | Background SSIM ↑ | Δquad jitter (px) ↓ |
|---|---|---|---|
| Full pipeline | XX.X | X.XX | X.X |
| w/o Alignment Refiner | XX.X | X.XX | X.X |
| w/o BPN (LCM only) | XX.X | X.XX | X.X |
| w/o LCM (hist-match) | XX.X | X.XX | X.X |
| w/o adaptive mask | XX.X | X.XX | — |

Table caption: *"Ablation on [video set name]. Each row disables one
component. Numbers are averages over N clips."* — user fills N and
the actual numbers.

### M.4 Figure filenames are load-bearing

The macro matches on filename without extension. Commit-tracked
placeholder PNGs (solid-color gray 16:9 with the target filename
printed on them) go into `report/figures/` from the start so
`latexmk` builds cleanly before real data arrives. Real images
replace them with the same filename — no LaTeX edits needed.

### M.5 Image provenance (note for later)

When real figures arrive, record in `report/figures/SOURCES.md`:
source video file, frame range, config used, date produced. Keeps
the paper reproducible without bloating the LaTeX.
