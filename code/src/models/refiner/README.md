# ROI Alignment Refiner

Residual homography estimator for S5. Given two almost-aligned canonical-frontal
text ROIs (reference frame ROI + target frame ROI), predicts a small 3x3
correction `ΔH` that snaps the reference ROI onto the target ROI's actual
text position. The correction is then composed into the existing S5 warp
chain and applied to the **edited** ROI (not the reference) at compositing
time — so cross-language text replacements land pixel-accurate on the target
text instead of drifting by a few pixels with CoTracker's residual tracking
error.

The network is trained standalone on canonical-frontal ROIs extracted by the
TPM data generation pipeline. No dependency on the rest of the pipeline until
S5 integration (Part 2 of the alignment refiner plan).

## Role in the pipeline

```
           reference frame          target frame
                 │                       │
        H_to_frontal[ref]        H_to_frontal[tgt]   (from S2)
                 │                       │
                 ▼                       ▼
       ref_canonical_ROI        target_canonical_ROI  (shared canonical frontal space)
                 │                       │
                 └──────────┬────────────┘
                            ▼
                    ROIRefiner (this module)
                            │
                            ▼
                     Δcorners (4, 2)
                            │
                    corners_to_H
                            │
                            ▼
                       ΔH (3, 3)
                            │
                            ▼
     S5: edited_ROI ──────► warp(edited, T @ H_from_frontal[tgt] @ inv(ΔH))
                            │
                            ▼
                   composite into target frame
```

At training time we never touch the edited ROI — the refiner only learns
pairwise alignment from two unedited canonical frontal frames of the same
track. At inference time the `ΔH` is applied to the edited ROI.

## Files

| file                | contents                                                              |
|---------------------|-----------------------------------------------------------------------|
| `warp.py`           | Differentiable DLT, forward-direction warp, validity mask, compose    |
| `dataset.py`        | `RefinerDataset` — synthetic (Type A) + real in-track (Type B) pairs  |
| `model.py`          | `ROIRefiner` network: concat + CNN + FC head, ~1.24M params           |
| `losses.py`         | `RefinerLoss` with per-sample Type A/Type B routing + primitives      |
| `train.py`          | Training loop with schedule ramp, resume, init-from                   |
| `evaluate.py`       | Metrics + visualizations for head-to-head checkpoint comparison       |
| `config.yaml`       | Default training config (ramped Stage 1 → Stage 2 in one run)         |
| `config_stage1.yaml`| 100% synthetic supervised pretraining                                 |
| `config_stage2.yaml`| 70% real / 30% synthetic fine-tune from Stage 1 best                  |

Training data lives in `/workspace/tpm_dataset/{video_name}/{track_name}/frame_*.png`
— the same layout used by the BPN training code. All frames within a track
share the same native canonical-frontal resolution; tracks are resized to
`(64, 128)` on load.

## Key design decisions

### Coordinate system: canonical frontal, not frame pixel space

The refiner operates entirely in canonical frontal space (the axis-aligned
rectangle S2 computes per-track). This decision is load-bearing:

1. **Training data is free.** The TPM data gen pipeline already extracts
   `warpPerspective(frame, H_to_frontal, canonical_size)` for every frame of
   every track — these are exactly the inputs the refiner needs. No extra
   data pipeline.
2. **`H0 = I`.** If CoTracker were perfect, two canonical ROIs from the same
   track would overlap pixel-perfectly. The residual misalignment between
   two canonical ROIs is precisely the error the refiner must learn to
   correct, and "residual around identity" is the best-conditioned training
   regime.
3. **S5 already computes the target side.** S4 already warps each target
   frame to canonical frontal for LCM/histogram matching. At inference
   S5 just keeps that tensor alive and hands it to the refiner.
4. **Composition stays clean.** The predicted `ΔH` composes into the
   existing warp chain as `H_adjusted = T @ H_from_frontal[tgt] @ inv(ΔH)`
   without requiring any other S5 geometry changes.

### 4-corner offset parameterization

Standard in deep homography (DeTone 2016, HomographyNet): predict 8 values
as four `(dx, dy)` corner displacements, convert to `H` via DLT. Safer
than regressing the 3x3 matrix directly — the corner space is compact,
easily regularized toward identity, and convex (small perturbations
produce valid homographies nearly always).

### Synthetic (Type A) + real (Type B) sample mix

- **Type A**: one random frame → apply random ±8 px 4-corner perturbation
  → target. Ground truth corners known exactly. Used for supervised corner
  regression.
- **Type B**: **two random distinct frames from the same track, no time-gap
  constraint**. This was deliberate: inference aligns the reference (often
  frame 0) against arbitrary target frames, including ones hundreds of
  frames away. Training on close-in-time pairs would systematically
  undersample the deployment distribution.
- Per-epoch ratio controlled by a mutable `real_pair_fraction` attribute
  the training loop updates from a schedule.

### Single-class mixed-batch dataset

Rather than two parallel dataloaders, `RefinerDataset.__getitem__` picks
sample type per-item based on `real_pair_fraction`. The training loop
updates that attribute between epochs and the change takes effect on the
very next batch. This keeps the ramp schedule trivial and avoids two
separate optimizer bookkeeping paths.

### Lighting-robust losses on the self-supervised side

RGB MSE or plain Charbonnier on real pairs would punish the network for
lighting drift (shadows, auto-exposure, specular highlights) that is not
geometric. Type B uses:

- **Masked NCC on luminance** — shift/scale invariant per patch, so
  proportional brightness changes produce zero error.
- **Masked Charbonnier on Sobel magnitude** — not signed gradients, because
  the sign flips under contrast inversion. Matches edge structure, which is
  what carries the alignment signal.
- **Corner magnitude L2 regularization** — keeps `ΔH` near identity on
  ambiguous pairs where the photometric minimum isn't the geometric one.

Type A keeps the simpler masked Charbonnier on RGB — synthetic pairs share
the same source image so there's no lighting change to worry about.

### Validity + center weighting on all masked losses

The loss weight per sample is `validity_mask * center_weight`:

- **Validity mask**: warp a ones-tensor through the predicted `H`; anything
  that sampled outside the source is 0. This excludes the black triangles
  at the boundary from the loss. Computed on the fly, not stored.
- **Center weight**: fixed soft radial feather, 1.0 in the interior fading
  to 0.1 at the outer ~10% of each dimension. Downweights the bbox-fringe
  region which often contains scene content, not text.

Both weights are built at loss-evaluation time, so the dataset never needs
to carry alpha masks — a deliberate simplification.

### Small-weight init on the final FC

The final Linear layer has weights initialized with `N(0, 1e-3)` so the
untrained model predicts `Δcorners ≈ 0`, giving `ΔH ≈ I`. This is
load-bearing for two reasons: the reconstruction loss is best-conditioned
when the initial warp is a no-op (no spurious border triangles), and the
residual regularizer is anchored at zero.

## Network architecture

HomographyNet-style concat-and-CNN:

```
Input:  (B, 6, 64, 128)   # source RGB + target RGB, concatenated
        │
        ▼
ConvBNReLU(6, 32, s=2)     # (B, 32, 32, 64)
ConvBNReLU(32, 64, s=2)    # (B, 64, 16, 32)
ConvBNReLU(64, 96, s=2)    # (B, 96, 8, 16)
ConvBNReLU(96, 128, s=2)   # (B, 128, 4, 8)
        │
        ▼
Flatten                    # (B, 4096)
Linear(4096, 256)          # dominates parameter budget
ReLU + Dropout(0.2)
Linear(256, 8)             # small-init -> initial output ≈ 0
        │
        ▼
view(-1, 4, 2)             # (B, 4, 2) corner offsets
```

**Total parameters**: 1,237,576 (~1.24M). Parameter budget is dominated by
`Linear(4096, 256)` at 1.05M — about 85% of the network. Small, fast on
CPU for tests, trivial on GPU for real training.

Architecturally this is the simplest sensible choice. Upgrade paths
considered but deliberately deferred until the minimum version plateaus:

- **Correlation volume** at the feature level (à la geometric matching
  networks). Better inductive bias for explicit matching but triples code
  complexity. Defer until small version plateaus.
- **Multi-scale prediction** (coarse homography → refine).
- **Residual flow head** after the homography for non-planar surfaces.

## Training protocol

### Stage 1: 100% synthetic supervised

- **Config**: `config_stage1.yaml`
- **28 train videos / 6 val videos** (split by video name, `/workspace/tpm_dataset`)
- **Losses**: `corner=1.0` (smooth L1) + `recon=0.25` (masked Charbonnier on warped source vs target). All Type B losses zeroed.
- **Optimizer**: AdamW, `lr=1e-4` peak, `weight_decay=1e-4`
- **Schedule**: 2-epoch linear warmup from 1% → 100%, then cosine annealing to `1e-6`
- **Epochs**: 30
- **Batch size**: 128
- **Augmentation**: independent per-branch photometric aug (brightness ±0.2, contrast [0.8, 1.2], gamma [0.65, 1.35], optional Gaussian noise σ ≤ 0.02, optional Gaussian blur σ ≤ 1.5). Applied to both source and target independently to prevent raw RGB equality shortcuts.

### Stage 2: 70% real / 30% synthetic fine-tune (Option B from plan)

- **Config**: `config_stage2.yaml`, `--init-from checkpoints/refiner/stage1/refiner_best.pt`
- **Same data split, same model, same aug.**
- **Losses active**: all five — corner + recon on the 30% synthetic anchor (same weights as Stage 1, keeps the corner prediction grounded), plus ncc + grad + reg on the 70% real pairs.
- **Loss weights**: `corner=1.0, recon=0.25, ncc=1.0, grad=1.0, reg=0.01`
- **Optimizer**: AdamW, `lr=3e-5` peak (3x lower than Stage 1), `weight_decay=1e-4`
- **Schedule**: 1-epoch warmup, cosine to `1e-6`
- **Epochs**: 15
- **`real_pair_fraction=0.7` from epoch 0** — no ramp. Stage 1 is already converged on synthetic, so the usual 0→0.8 warmup would waste compute re-learning Stage 1's solution. Jumping straight to 70% real points every gradient step at the real-pair distribution gap.
- **`--init-from` loads only model weights**; the optimizer, scheduler, and epoch counter all start fresh so the new config's LR and epoch count actually take effect. (The `--resume` path does a full resume including optimizer state — used only for mid-training restarts within one stage.)

## Results

### Stage 1 (synthetic supervised only)

Monotonic descent over 30 epochs.

| Epoch | Train corner loss | Val corner_err_px |
|-------|-------------------|-------------------|
| 1     | 3.14              | 3.21              |
| 5     | 0.35              | 0.60              |
| 10    | 0.29              | 0.52              |
| 15    | 0.27              | 0.48              |
| 20    | 0.25              | 0.46              |
| **28 (best)** | **0.246** | **0.434** |
| 30    | 0.245             | 0.450             |

- Corner error dropped **3.21 px → 0.434 px** (~7.4x improvement) on held-out videos.
- Train-val gap of ~0.2 px throughout — no overfitting.
- Recon loss hit its floor (~0.15, photometric-aug noise) by epoch 2 and stayed there while corner loss kept descending. Recon is a secondary signal here; corner loss carries the training signal.
- Best checkpoint is epoch 28; epoch 30's corner error is slightly higher but still very close — the cosine schedule converged naturally.

### Stage 2 (70% real / 30% synthetic fine-tune)

15 epochs starting from `stage1/refiner_best.pt` via `--init-from`.

| Epoch | train total | ncc | grad | val corner_err_px |
|-------|-------------|-----|------|-------------------|
| 1     | 0.660       | 0.174 | 0.114 | 0.474             |
| 5     | 0.596       | 0.145 | 0.108 | 0.468             |
| 10    | 0.588       | 0.144 | 0.108 | 0.468             |
| **11 (best)** | 0.587 | 0.144 | 0.108 | **0.447** |
| 15    | 0.583       | 0.143 | 0.108 | 0.458             |

- **NCC loss: -17.8% (0.174 → 0.143)** — real-pair alignment measurably improved on the illumination-invariant metric.
- **Sobel-magnitude loss: -5.4% (0.114 → 0.108)** — smaller but consistent edge-structure improvement.
- **Corner loss stayed anchored at ~0.25** on the synthetic minority. The 30% synthetic anchor did its job — no drift off Stage 1's known-good solution.
- **Val corner error: +3% vs Stage 1 (0.447 vs 0.434)**. Small regression on synthetic, expected and accepted as the cost of real-pair gain.
- Fine-tune converged quickly: most of the NCC drop happened in the first 5 epochs, the rest was incremental polish.

### Head-to-head evaluation on held-out videos

`evaluate.py --checkpoint stage1/best --checkpoint stage2/best` on the same
val loader (forced `real_pair_fraction=0.5`):

| metric                    | stage1   | stage2   | winner |
|---------------------------|----------|----------|--------|
| syn_corner_err_mean_px    | 0.4395   | 0.4637   | stage1 (−5.5%) |
| syn_corner_err_p90_px     | 0.8067   | 0.8327   | stage1 (−3.2%) |
| syn_mask_iou_mean         | 0.9947   | 0.9943   | tie    |
| real_ncc_pre (baseline)   | 0.7929   | 0.7886   | —      |
| real_ncc_post             | 0.8721   | 0.8818   | stage2 |
| **real_ncc_gain**         | **+0.079** | **+0.093** | **stage2 (+17.7%)** |
| real_grad_pre (baseline)  | 0.1905   | 0.1914   | —      |
| real_grad_post            | 0.1572   | 0.1488   | stage2 |
| **real_grad_gain**        | **+0.033** | **+0.043** | **stage2 (+27.8%)** |
| real_pred_disp_mean_px    | 1.3998   | 1.1879   | stage2 (smaller, more selective) |

**Both checkpoints actively improve real alignment** (`real_ncc_gain` and
`real_grad_gain` are positive for both — not just the loss dropping during
training, but pre-vs-post refinement on held-out videos). Stage 2 produces
a larger improvement while predicting **smaller** corner displacements on
average (1.19 px vs 1.40 px) — it learned to make more selective, targeted
corrections instead of larger warps.

### Selected checkpoint

**`refiner_v0.pt` = `stage2/refiner_best.pt` (epoch 11)**

The 5.5% synthetic corner regression is bought for a 17.7% NCC gain and
27.8% Sobel gain on the real-pair distribution — the distribution the
refiner will actually see at inference. Synthetic ±8 px perturbations are
never seen in deployment; real in-track residual errors are. Visual
inspection of side-by-side strips, edge overlays, and blink GIFs confirmed
the improvement without any visible "chasing shadows" failure modes on
low-contrast samples.

## Running training

```bash
cd code

# Stage 1: 100% synthetic, ~30 epochs from scratch
python -m src.models.refiner.train --config src/models/refiner/config_stage1.yaml

# Stage 2: 70/30 fine-tune from Stage 1 best. Use --init-from (NOT --resume)
# so the Stage 2 config's LR and epoch count take effect on a fresh scheduler.
python -m src.models.refiner.train \
    --config src/models/refiner/config_stage2.yaml \
    --init-from checkpoints/refiner/stage1/refiner_best.pt
```

### Flags

- `--resume <path>`: full resume — loads model + optimizer + scheduler +
  epoch counter. Use for mid-training restarts within a single stage.
- `--init-from <path>`: weights-only init — loads just the model
  parameters, starts with a fresh optimizer, scheduler, and epoch counter.
  **Always use this for stage transitions** or any run where the new
  config's LR schedule and epoch count need to take effect cleanly.
- `--epochs`, `--batch-size`, `--device`, `--no-progress`: standard config
  overrides.

### Outputs

Checkpoints + log written to `checkpoint.out_dir` from the config:
```
checkpoints/refiner/stage{1,2}/
├── refiner_best.pt         # best by val corner_err_px
├── refiner_last.pt         # last epoch
├── refiner_epoch{5,10,...}.pt
└── train_log.json          # incremental per-epoch losses + val metrics
```

## Running evaluation

```bash
cd code

python -m src.models.refiner.evaluate \
    --config src/models/refiner/config_stage1.yaml \
    --checkpoint checkpoints/refiner/stage1/refiner_best.pt \
    --checkpoint checkpoints/refiner/stage2/refiner_best.pt \
    --out-dir checkpoints/refiner/eval \
    --n-vis 32
```

Produces per-checkpoint `strips.png`, `edge_overlays.png`, `blink/*.gif`,
and a combined `comparison.json` + side-by-side summary table printed to
stdout.

- `config.data` is read for the data root and val split.
- `real_pair_fraction` is forced to `0.5` for metrics and `1.0` for
  visualizations (overriding whatever the config says) so every checkpoint
  is scored on the same sample mix.
- Each checkpoint reconstructs its model architecture from the
  `config` dict embedded in the checkpoint itself, so evaluation is
  robust to architecture drift between checkpoints.

## Testing

```bash
cd code
python -m pytest tests/models/test_refiner_*.py -q
```

105 unit tests across 6 files, ~40 seconds on CPU. Coverage highlights:

- **`warp.py`**: DLT output matches `cv2.getPerspectiveTransform` numerically;
  `warp_image` matches `cv2.warpPerspective` on the interior; round-trip
  `warp → warp^{-1}` recovers the original; concrete direction test — a
  +10 px corner translation moves image content +10 px to the right (this
  is the guard against future sign flips); differentiability pinned.
- **`dataset.py`**: split by video respected; sample cap math correct;
  synthetic delta within perturbation range; synthetic target bit-exact
  equals `warp(source, H_gt)` with aug disabled; cache vs disk load
  equivalence; DataLoader collation smoke test.
- **`model.py`**: construction, parameter count (0.9M < n < 2M), shape
  validation, initial output near zero, differentiability through the
  warp chain.
- **`losses.py`**: every primitive pinned (luminance, Sobel magnitude,
  masked Charbonnier, masked NCC), Type A / Type B routing correct on
  mixed and pure batches, `RefinerLoss.total` is linear in weights, hand-
  computed references for corner and reg losses, gradient flow through
  the full pipeline.
- **`train.py`**: schedule interpolation, seeding determinism, end-to-end
  smoke test, **overfitting sanity check** (train corner loss must drop
  from ~2.6 → <1.0 px on 8 fake samples in 150 epochs — pins every link
  in the data → model → loss → optimizer chain), resume path, init-from
  path, mutual exclusion.
- **`evaluate.py`**: checkpoint round-trip, model architecture
  reconstruction from embedded config, metrics return all expected keys,
  untrained-model displacement stays small (init-scale sanity), vis
  output files exist.

## Upgrade paths (for the future)

If Stage 2 eventually plateaus on harder tracks (curved surfaces, rolling
shutter, extreme motion blur), try in this order:

1. **Widen the synthetic perturbation range** past ±8 px and see if the
   network handles it. Easy one-line change in config.
2. **Add small cutout / random erase augmentation** for partial occlusion
   robustness.
3. **Multi-scale prediction**: coarse homography at lower resolution,
   refine at higher.
4. **Correlation volume** at the feature level (geometric matching
   network). Explicit matching inductive bias.
5. **Residual flow head** after the homography, for non-planar surfaces
   where a single global `H` can't fit. Do not start here — most text
   surfaces in real videos are planar.
