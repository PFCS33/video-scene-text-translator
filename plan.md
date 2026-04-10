# Plan: ROI Alignment Refiner for S5

## Goal

Train a lightweight neural network that predicts a **residual homography `ΔH`** between two
almost-aligned canonical-frontal text ROIs, and integrate it into S5 so that the edited ROI
lands pixel-accurate on the target text instead of drifting with CoTracker's residual error.

**Training problem** (pure 2D alignment, no pipeline dependencies):

> Given two canonical-frontal ROIs `S` and `T` of the same text taken from two different
> frames, predict the 4-corner offsets `Δcorners` such that `warp(S, ΔH)` aligns pixel-wise
> with `T`.

**Inference problem** (once integrated):

> At S5, predict `ΔH` from `(ref_roi_canonical, target_roi_canonical)` — both *unedited*
> — then compose `ΔH` into the existing de-frontalization warp and apply it to the
> *edited* ROI before compositing.

The network is trained and evaluated entirely standalone — no S5 dependency — and only
integrated once it has a usable checkpoint.

---

# Part 1: Standalone Refiner Network

All code lives under [code/src/models/refiner/](code/src/models/refiner/), mirroring the
BPN module layout.

```
code/src/models/refiner/
├── __init__.py
├── config.yaml              # training config (paths, hparams)
├── dataset.py               # RefinerDataset + synthetic + real pair sampling
├── warp.py                  # differentiable corners→H, warp, validity mask
├── model.py                 # ROIRefiner nn.Module
├── losses.py                # masked NCC, masked Sobel-magnitude Charbonnier, corner L1
├── train.py                 # training loop (mirrors bpn/train.py)
└── evaluate.py              # val metrics + visualizations
```

Training/eval are invoked via `python -m src.models.refiner.train --config src/models/refiner/config.yaml`,
matching the BPN CLI pattern.

## 1.1 Dataset

**Data source:** `/workspace/tpm_dataset/{video_name}/{track_name}/frame_NNNNNN.png`
— same layout the BPN dataset already uses. Every image is a canonical-frontal ROI;
all frames within a single track share the same native resolution (but tracks differ
widely: observed H ∈ [18, 218], W ∈ [37, 716] in one video).

**Network input size:** fixed `(H=64, W=128)`. Every ROI is resized on load
(bilinear). This matches BPN's convention and means corner offsets are predicted in
network pixel coordinates, which we unscale at inference.

**Split strategy:** by `video_name`, not by track. Different tracks within the same
video share camera/lighting/font style — safer to keep whole videos on one side of the
split. Config lists train/val video names explicitly, same as BPN.

**Sample types** — dataset produces a mix:

### Type A: Synthetic self-pairs (supervised)
1. Sample one frame from any track → `I`.
2. Resize to `(64, 128)`.
3. Sample random 4-corner offsets `Δcorners_gt` with each coordinate drawn
   uniformly from `[-8, +8]` pixels (see §1.2 for details).
4. Compute `H_gt` via DLT from canonical corners → perturbed corners.
5. `S = I` (identity), `T = warp(I, H_gt)` with `grid_sample` (bilinear, zero padding).
6. Apply independent photometric augmentation to `S` and `T` (see §1.3).
7. Return `(S, T, Δcorners_gt, valid_mask_gt)` where `valid_mask_gt` marks pixels in
   `T` that sampled inside `I` (zero at the border triangles).

### Type B: Real in-track pairs (self-supervised)
1. Sample a track with ≥ 2 frames.
2. Sample **two distinct random frame indices** `i, j` uniformly from the track —
   **no temporal-gap constraint**. Tracks can be hundreds of frames long; at
   inference the refiner aligns the reference frame against an arbitrary target
   frame that may be the far end of the track, so training must cover the full
   intra-track distance distribution.
3. Load both, resize to `(64, 128)`.
4. Random ordering: `S = ROI[i], T = ROI[j]` with 50% probability, swapped otherwise.
   The geometric alignment signal is symmetric, so both orderings are valid and
   double the effective training data.
5. Light photometric augmentation (less than Type A — real pairs already have real
   appearance differences).
6. Return `(S, T, None, None)` — no ground truth.

**Sampling budget per track:** long tracks can contribute more pairs, but we cap
them via `pairs_per_track` (default = `min(track_length, 64)`) to prevent a handful
of very long tracks from dominating the epoch. Each dataset sample entry stores only
a track index; the two frame indices are drawn fresh on every `__getitem__` call so
every epoch sees a different subset of the `N choose 2` possible pairs.

Each `__getitem__` returns a dict with a `sample_type` field (`"syn"` or `"real"`) so
the training loop can route to the right loss. A config knob `real_pair_fraction`
controls the Type A : Type B ratio per epoch (ramped over training — see §1.6).

**RAM caching:** reuse BPN's contiguous-uint8-ndarray trick from
[dataset.py:172](code/src/models/bpn/dataset.py#L172) — one `(N_unique, 64, 128, 3)`
array, fork-safe. At 24 KB/image it's a few GB for the whole dataset.

## 1.2 Synthetic warp generation

**Parameterization:** 4-corner offsets (DeTone 2016). Canonical source corners:
```
[(0, 0), (W, 0), (W, H), (0, H)]  # W=128, H=64
```
Perturbed corners: canonical + per-corner offset sampled from `Uniform(-8, +8)` pixels
independently on x and y. `H_gt` derived via `kornia.geometry.transform.get_perspective_transform`
(differentiable DLT).

**Reject degenerate samples:** after sampling, verify the perturbed quad is convex and
non-self-intersecting; resample if not. In practice ±8 px on 128×64 produces degenerate
quads extremely rarely, but we guard anyway.

**Warp direction:** `T = warp(S, H_gt)` where `warp` uses `F.grid_sample` with the
inverse of `H_gt` to build the sampling grid (standard: to know where to *read* from
`S` for each pixel of `T`, we need `H_gt^{-1}`). All helper functions live in
`warp.py` so the direction convention is defined in exactly one place.

**Ground-truth label:** the corner offsets the network should predict are `Δcorners_gt`
as given (not `-Δcorners_gt`), because the network's job is "warp `S` by `ΔH` to land
on `T`" — the same direction as the synthetic generation.

## 1.3 Photometric augmentation

Applied independently to `S` and `T`, kornia-style:
- Brightness shift: `± 0.2`
- Contrast scale: `× [0.8, 1.2]`
- Gamma: `[0.7, 1.4]`
- Hue/saturation: small
- Gaussian blur: `σ ∈ [0, 1.5]` with 50% probability
- Gaussian noise: `σ ∈ [0, 0.02]` with 50% probability

**Type A vs Type B strength:** Type A uses the full range above (purely synthetic, needs
lots of appearance variance or network overfits to RGB equality). Type B uses roughly
half the strength — the real-pair appearance gap already carries realistic variance, too
much augmentation on top hurts signal.

## 1.4 Network architecture

First version: **HomographyNet-style concat + CNN + FC head**. No correlation volume,
no feature pyramid, no attention — just the minimum viable design. Only upgrade if it
plateaus in evaluation.

```python
class ROIRefiner(nn.Module):
    def __init__(self, in_channels=6, base=32):
        # in_channels = 6 for [source RGB, target RGB] concat
        self.backbone = nn.Sequential(
            ConvBNReLU(6,   base,   3, s=2),   # 64x128 -> 32x64
            ConvBNReLU(base,   base*2, 3, s=2),   # -> 16x32
            ConvBNReLU(base*2, base*3, 3, s=2),   # -> 8x16
            ConvBNReLU(base*3, base*4, 3, s=2),   # -> 4x8
        )
        self.head = nn.Sequential(
            nn.Flatten(),              # 4*8*128 = 4096
            nn.Linear(4096, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 8),         # 4 corners x (dx, dy)
        )

    def forward(self, source, target):
        x = torch.cat([source, target], dim=1)
        feat = self.backbone(x)
        delta = self.head(feat)
        return delta.view(-1, 4, 2)   # (B, 4, 2)
```

Approx params: ~1.5M. Easily trains on one GPU, fast inference.

**Initialization note:** final FC layer weights scaled down so initial output is near
zero → initial `ΔH ≈ I`. Same trick BPN uses to keep tanh/softplus in the linear regime
at init ([CHANGELOG.md:38](CHANGELOG.md#L38)).

## 1.5 Differentiable warp + loss utilities

### `warp.py`

```python
def corners_to_H(delta_corners, canonical_corners):  # differentiable DLT
def warp_image(image, H, out_shape):                 # F.grid_sample wrapper
def warp_validity_mask(H, out_shape):                # warp a ones-tensor
def compose_H(H_left, H_right):                      # matmul with normalization
```

All operate on batched `(B, ...)` tensors. `corners_to_H` uses
`kornia.geometry.transform.get_perspective_transform` under the hood if kornia is
already a dependency — otherwise a ~15-line DLT implementation.

### `losses.py`

**Corner loss (Type A only):**
```python
L_corner = F.smooth_l1_loss(pred_corners, gt_corners)
```

**Reconstruction loss — masked robust photometric:**
Used as a secondary loss in Type A and primary in Type B. Computes loss only on pixels
where the warped source is valid.

```python
def masked_charbonnier(x, y, weight, eps=1e-3):
    diff = torch.sqrt((x - y).pow(2) + eps**2)
    return (diff * weight).sum() / (weight.sum() + 1e-6)
```

**Illumination-robust losses (Type B emphasis):**

1. **Masked Sobel-magnitude Charbonnier** — compute Sobel magnitude (not signed
   gradients) on both `warp(S, pred_H)` and `T`, then masked Charbonnier. Edge
   structure is what we want to match; magnitude is illumination-invariant up to
   contrast scaling.

2. **Masked normalized cross-correlation on luminance** — for each batch item, convert
   both images to Y, compute per-patch NCC under the mask. NCC is shift+scale
   invariant so it survives brightness/contrast changes. Loss = `1 - NCC`.

**Residual regularization (Type B):**
```python
L_reg = (pred_corners ** 2).mean()
```
Keeps `ΔH` near identity so the network doesn't drift off into unlikely warps on
ambiguous pairs.

**Mask construction inside the loss module:**
```python
W = validity_mask * center_weight
```
- `validity_mask`: 1 where `warp(S, pred_H)` sampled inside `S`, 0 otherwise. Obtained
  by warping a ones-tensor through the same `pred_H`.
- `center_weight`: fixed soft radial feather, 1.0 in the middle, fades to ~0.1 over
  the outer 10% of each dimension. Biases the loss toward the text core and away from
  bbox-fringe content that may be scene-specific.

### Total loss

**Stage 1 (synthetic supervised):**
```
L = w_corner * L_corner
  + w_recon  * masked_charbonnier(warp(S, pred_H), T, W)
```
Defaults: `w_corner=1.0`, `w_recon=0.25`.

**Stage 2 (real self-supervised):**
```
L = w_ncc   * (1 - masked_ncc_luminance(warp(S, pred_H), T, W))
  + w_grad  * masked_charbonnier(sobel_mag(warp(S, pred_H)), sobel_mag(T), W)
  + w_reg   * L_reg
```
Defaults: `w_ncc=1.0`, `w_grad=1.0`, `w_reg=0.01`.

**Mixed batches** (late Stage 1 / early Stage 2): compute whichever losses apply to
each sample and average by type.

## 1.6 Training loop

File: `refiner/train.py`. Structure lifts directly from `bpn/train.py` — YAML config,
`set_seed`, `create_dataloaders`, AdamW, cosine schedule with warmup, periodic + best-val
checkpoints, `--resume` support.

**Config knobs** (`refiner/config.yaml`):
```yaml
data_root: /workspace/tpm_dataset
train_videos: [realworld_real_video0, realworld_real_video1, ...]   # list
val_videos:   [realworld_COFFEE, realworld_custom_video1, ...]       # list
image_size: [64, 128]         # (H, W)
pairs_per_track: 64           # cap on Type B samples contributed by each track per epoch
real_pair_fraction: 0.0       # ramped by schedule below
syn_perturbation_px: 8
center_weight_band: 0.1

model:
  base_channels: 32
  dropout: 0.2

training:
  batch_size: 128
  num_workers: 8
  epochs: 50
  lr: 1.0e-4
  weight_decay: 1.0e-4
  warmup_epochs: 2
  grad_clip: 1.0
  real_pair_schedule:         # epoch -> real_pair_fraction
    0: 0.0
    10: 0.2
    20: 0.5
    35: 0.8
  loss_weights_stage1: {corner: 1.0, recon: 0.25}
  loss_weights_stage2: {ncc: 1.0, grad: 1.0, reg: 0.01}

checkpoint:
  out_dir: checkpoints/refiner
  save_every_epochs: 5
  keep_last: 3
```

**Training schedule** — no hard "Stage 1 / Stage 2" split; we ramp `real_pair_fraction`
over epochs. Synthetic-supervised signal stays in the batch even late in training as a
regularizer. Loss weights switch smoothly: at each step, compute both loss variants and
blend by the same ratio as the batch mix.

**Resume & checkpointing:** standard. State dict + optimizer + scheduler + epoch +
best-val metric.

**Monitoring:** log to tensorboard (or just JSON per-epoch, matching BPN). Track per
epoch: train/val loss, train/val mean corner error (on synthetic), train/val masked NCC
(on real), gradient norm, LR.

## 1.7 Evaluation & visualization

File: `refiner/evaluate.py`.

**Synthetic metrics** (ground truth known):
- Mean corner error (pixels, at network resolution). This is the headline metric.
- 90th percentile corner error.
- IoU of warped-source support mask vs. ground-truth warped mask.

**Real metrics** (no ground truth):
- Masked NCC on luminance (pre- vs. post-refinement).
- Masked Sobel-magnitude L1 (pre- vs. post-refinement).
- Mean predicted corner displacement (sanity: should be small but non-zero).

**Visualizations** — dump a few dozen PNG strips to `checkpoints/refiner/vis/`:
1. `(S, T, warp(S, pred_H), |warp(S, pred_H) - T|)` side-by-side.
2. Edge overlay: Canny edges of `S` (red) vs. `T` (green), before and after refinement.
   Good alignment → edges turn yellow.
3. Blink GIF: alternate `warp(S, pred_H)` and `T` at 2 Hz (2-frame GIF) for 20 random
   real pairs — the eye catches sub-pixel misalignment that scalar metrics miss.

## 1.8 Sanity checks & Done-When (Part 1)

**Unit tests** in [code/tests/models/](code/tests/models/):
- `corners_to_H` matches `cv2.getPerspectiveTransform` numerically.
- `warp_image(I, I_corners→I_corners)` is identity (no spurious transform).
- `warp_image(warp_image(I, H), H^{-1})` recovers `I` under the valid mask.
- Dataset `__getitem__` returns tensors with expected shapes and dtypes.
- `masked_charbonnier` handles all-zero mask without NaN.
- Loss is finite for random inputs.

**Done When:**
- [ ] Refiner module scaffolded under `code/src/models/refiner/`
- [ ] Dataset produces both sample types correctly (verified visually on a handful of samples)
- [ ] Stage 1 synthetic-only training converges to < 1.0 px mean corner error on val
- [ ] Stage 2 (mixed) training improves real masked-NCC over the Stage 1 checkpoint
- [ ] Visualizations in `checkpoints/refiner/vis/` show qualitatively better alignment
      after refinement on real pairs
- [ ] Final checkpoint saved to `checkpoints/refiner/refiner_v0.pt`

---

# Part 2: Pipeline Integration (S5)

Only start after Part 1 has a validated checkpoint.

## 2.1 Data flow changes

Currently, S5 receives `PropagatedROI` objects (from S4) and per-frame `H_from_frontal`
(from S2 via `TextDetection`). To run the refiner we need **two additional canonical
frontal images per target frame**:

1. **`ref_roi_canonical`** — the unedited reference frame warped to canonical frontal.
   Shape `track.canonical_size`. One per track.
2. **`target_roi_canonical`** — the current target frame warped to canonical frontal.
   Shape `track.canonical_size`. One per target detection.

**(1)** is trivial: `cv2.warpPerspective(frames[ref_idx], H_to_frontal[ref], canonical_size)`.
Compute once per track at the top of S5's `run()`.

**(2)** is already computed inside S4 for histogram matching / LCM — it's the
`frame_roi` variable inside `PropagationStage.run`. We need to keep it alive. Two
options:

- **Option A (preferred):** add an optional `target_roi_canonical: np.ndarray | None`
  field to `PropagatedROI` in [data_types.py](code/src/data_types.py). S4 fills it when
  the refiner is enabled (cheap — the array already exists in S4's loop). None when
  disabled. No other change to S4.
- **Option B:** recompute in S5 from `H_to_frontal[target]` and `frames[target_idx]`.
  Costs a redundant `warpPerspective` per detection.

Pick **A** to avoid duplicate work. Field is optional so non-refiner runs are unaffected.

## 2.2 Refiner wrapper

New file: [code/src/stages/s5_revert/refiner.py](code/src/stages/s5_revert/refiner.py)
(we'll convert `s5_revert.py` into a package — same pattern as S4).

```python
class ROIRefiner:
    """Inference wrapper around the trained refiner checkpoint."""

    def __init__(self, checkpoint_path: str, device: str = "cuda",
                 image_size: tuple[int, int] = (64, 128),
                 max_corner_offset_px: float = 16.0):
        # lazy: defer torch + model load until first predict() call
        ...

    def predict_delta_H(
        self,
        source_canonical: np.ndarray,  # (H_can, W_can, 3) uint8
        target_canonical: np.ndarray,  # (H_can, W_can, 3) uint8
    ) -> np.ndarray | None:
        """Return a 3x3 ΔH in canonical-frontal pixel coordinates, or None if
        the prediction fails sanity checks (see §2.5)."""
        ...
```

**Scale handling (critical):** the network operates on `(64, 128)`. Canonical sizes
vary per track. So:
1. Resize `(source, target)` to `(64, 128)` for the forward pass.
2. Predict `Δcorners_net` in network pixels.
3. Unscale: `Δcorners_canonical = Δcorners_net * (W_can / 128, H_can / 64)` per axis.
4. Compute `ΔH_canonical = corners_to_H(canonical_corners, canonical_corners + Δcorners_canonical)`
   with canonical corners at the **actual track canonical size**.

The `ΔH` the caller gets is already in track canonical-frontal pixel units, ready to
compose.

**Lazy load:** match the BPN pattern — don't touch torch at module import time, load
model state on first `predict_delta_H` call.

## 2.3 S5 integration point

Inject between [s5_revert.py:207-213](code/src/stages/s5_revert.py#L207-L213) (where
`warp_roi_to_frame` is called) and [s5_revert.py:214](code/src/stages/s5_revert.py#L214)
(the composite call).

Pseudocode:
```python
# Existing — compute ref canonical once per track at start of run()
ref_roi_by_track = {}
for track in tracks:
    ref_det = track.detections[track.reference_frame_idx]
    ref_frame = frames[track.reference_frame_idx]
    ref_roi_by_track[track.track_id] = cv2.warpPerspective(
        ref_frame, ref_det.H_to_frontal, track.canonical_size
    )

# Inside the frame loop, for each prop_roi:
if self.refiner is not None and prop_roi.target_roi_canonical is not None:
    delta_H = self.refiner.predict_delta_H(
        ref_roi_by_track[prop_roi.track_id],
        prop_roi.target_roi_canonical,
    )
else:
    delta_H = None

result = self.warp_roi_to_frame(
    prop_roi,
    det.H_from_frontal,
    frame.shape[:2],
    delta_H=delta_H,         # NEW
)
```

**`warp_roi_to_frame` change:** accept optional `delta_H`. When present:
```python
H_adjusted = T @ H_from_frontal @ np.linalg.inv(delta_H)
```

**Why `inv(delta_H)`:** we trained the network to predict `ΔH` such that
`warp(ref_canonical, ΔH) ≈ target_canonical`. That is, `ΔH` maps from "canonical space
at the reference alignment" to "canonical space at the target alignment." The edited
ROI is in the reference alignment (produced in S3 from the reference frame). To move
the edited ROI into the target alignment *and then* through `H_from_frontal` into
target frame space, the composition in OpenCV's "where to read from" convention is:
`H_adjusted = T @ H_from_frontal @ inv(delta_H)`. We'll write a unit test to verify
the direction is correct (§2.6).

**No change to `composite_roi_into_frame_seamless`** — it just sees a (possibly more
accurate) warped ROI and composites as before.

## 2.4 Config

Add to [code/src/config.py](code/src/config.py) under `RevertConfig`:
```python
use_refiner: bool = False
refiner_checkpoint_path: str = "checkpoints/refiner/refiner_v0.pt"
refiner_device: str = "cuda"
refiner_image_size: tuple[int, int] = (64, 128)
refiner_max_corner_offset_px: float = 16.0   # sanity threshold
```

Wire into [code/config/adv.yaml](code/config/adv.yaml) under `revert:` with
`use_refiner: true`. Leave `default.yaml` off.

Also wire an S4-side flag `propagation.save_target_canonical_roi: bool` that causes
S4 to populate `PropagatedROI.target_roi_canonical`. Default false. `adv.yaml` sets it
to true alongside the refiner.

## 2.5 Runtime fallback

Sanity checks inside `ROIRefiner.predict_delta_H`. Return `None` if any fails, and
S5 falls back to the existing (refiner-disabled) warp path — **never crash the
pipeline over a bad refiner prediction.**

Checks:
1. `max(|Δcorners|) > max_corner_offset_px` (in canonical pixels). Default 16. Cuts off
   clearly insane predictions.
2. `det(ΔH)` outside `[0.5, 2.0]` → degenerate warp, reject.
3. Condition number of `ΔH` > 1e4 → numerically unstable, reject.
4. `np.isnan(ΔH).any() or np.isinf(ΔH).any()` → reject.

Log at DEBUG level when a prediction is rejected; bump to INFO if the rejection rate
across a video exceeds some threshold (e.g. 10%) so we notice.

## 2.6 Tests

Add to [code/tests/stages/test_s5_revert.py](code/tests/stages/test_s5_revert.py):
- `warp_roi_to_frame` with `delta_H=None` produces identical output to before (no
  regression).
- `warp_roi_to_frame` with `delta_H=I` is equivalent to `delta_H=None`.
- **Direction sanity test**: construct a pair `(S, T)` where `T` is `S` translated by
  known `(dx, dy)`; feed a hand-computed `ΔH` matching that translation; verify the
  warped edited ROI lands at the expected pixel location in a synthetic frame. This
  catches `delta_H` vs. `inv(delta_H)` bugs — the exact gotcha ChatGPT flagged in §10.7.

Add to `code/tests/models/test_refiner.py`:
- `ROIRefiner.predict_delta_H` returns `None` on injected bad inputs (NaN image, all
  zeros, etc.).
- On a pair where `target = cv2.warpPerspective(source, H_known, ...)` with a small
  known warp, the predicted `ΔH` composed gives back an image close to the source
  (within a few pixels at canonical resolution). Uses a *mock* model that returns
  `Δcorners` derived from `H_known` to isolate the scale/compose math from the actual
  learned weights.

## 2.7 Done When (Part 2)

- [x] `PropagatedROI` extended with optional `target_roi_canonical` field
- [x] S4 populates the field when `save_target_canonical_roi` is on
- [x] `s5_revert.py` converted to `s5_revert/` package with `stage.py`, `refiner.py`
- [x] `warp_roi_to_frame` accepts and composes `delta_H`
- [x] `RevertConfig` has refiner fields; `adv.yaml` wires them on
- [x] Runtime fallback verified (crafted NaN input → fallback path runs)
- [x] All existing S5 tests still pass
- [x] New tests for direction, scale, and fallback pass
- [x] End-to-end run on one test video completes without crashes
- [x] Visual diff on at least one test video shows the refiner is either neutral or an
      improvement over no refiner (side-by-side output videos saved to
      `saved_videos/refiner_ablation/`)

---

## Risks

- **Synthetic-only training won't transfer.** Real residual errors may have a different
  distribution than uniform ±8 px corner perturbations. Mitigation: the Stage 2 real-pair
  schedule is load-bearing, not optional. If we skip it the refiner may regress real
  videos vs. no refiner.
- **Long tracks → large intra-track appearance gaps.** By dropping `max_gap` and sampling
  any two frames from a track, some pairs will have significant appearance differences
  (motion blur, lighting drift, partial occlusion). This matches inference distribution
  but also includes pairs where the *content* genuinely differs (e.g. specular highlights
  on a different part of the text). The illumination-robust losses (Sobel magnitude, NCC)
  and the residual regularizer guard against the network being pulled off by these, but
  if Stage 2 validation loss is unstable we may need to either (a) downweight very-far
  pairs with a soft distance-dependent weight, or (b) add an outlier-robust loss wrapper
  that down-weights samples with unusually high residual after convergence.
- **Lighting-induced photometric minimum ≠ geometric minimum.** Sobel-magnitude and
  NCC losses mitigate but don't eliminate. If we see the refiner "chasing shadows"
  during Stage 2, add a Canny-edge IoU loss term.
- **Canonical size varies wildly** (H 18-218 observed in one video). Extreme aspect
  ratios could confuse a network trained at fixed 64×128. If evaluation shows the
  refiner helps square-ish ROIs but hurts very narrow or very wide ones, consider
  multi-aspect training (sample a few canonical aspect ratios during Type A generation).
- **Direction bugs.** `delta_H` vs. `inv(delta_H)`, source-to-target vs. target-to-source
  — classic homography pitfalls. The explicit direction-sanity unit test in §2.6 is
  meant to catch these before they ever run on real data.
- **Training data leakage.** We split by video — must not use the same video in both
  train and val lists in `config.yaml`. A sanity assertion at dataloader construction
  rejects any overlap.
- **Refiner cost at inference.** Per-detection CUDA forward pass. For short clips with
  dozens of detections this is negligible; for long videos with thousands it could add
  up. If needed, batch all detections of a frame (or all detections of a track) in one
  forward pass — the model architecture supports arbitrary batch sizes out of the box.

## Progress

### Part 1: Network
- [x] Step 1.1: Scaffold `code/src/models/refiner/` module (empty files + `__init__.py`)
- [x] Step 1.2: Implement `warp.py` (corners→H, warp, validity) + unit tests
- [x] Step 1.3: Implement `dataset.py` (Type A + Type B) + sanity visualization
- [x] Step 1.4: Implement `model.py` (ROIRefiner) + init test
- [x] Step 1.5: Implement `losses.py` (corner, masked Charbonnier, Sobel mag, masked NCC, reg)
- [x] Step 1.6: Implement `train.py` + `config.yaml` + run synthetic-only overfitting sanity check
- [x] Step 1.7: Full Stage 1 (100% synthetic) training run to convergence
- [x] Step 1.8: Stage 2 (70% real / 30% synthetic) fine-tune on top of Stage 1 checkpoint
- [x] Step 1.9: Implement `evaluate.py` and produce synthetic + real metrics + vis strips
- [x] Step 1.10: Save `refiner_v0.pt` checkpoint (Stage 2 best selected based on real-pair metrics + visual comparison)

### Part 2: Integration
- [x] Step 2.1: Add optional `target_roi_canonical` field to `PropagatedROI`
- [x] Step 2.2: Populate field in S4 under config flag
- [x] Step 2.3: Convert `s5_revert.py` to package layout
- [x] Step 2.4: Implement `refiner.py` wrapper (scale, compose, sanity checks)
- [x] Step 2.5: Extend `warp_roi_to_frame` to accept `delta_H`
- [x] Step 2.6: Wire refiner into `RevertStage.run()`
- [x] Step 2.7: Add `RevertConfig` fields + `adv.yaml` wiring
- [x] Step 2.8: Write direction, scale, and fallback unit tests
- [x] Step 2.9: End-to-end run on a test video with and without refiner, save comparison
