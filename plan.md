# Plan: Hi-SAM Segmentation-Based Inpainter

## Goal
Add a new `SegmentationBasedInpainter` backend to S4 that uses Hi-SAM (vendored at `third_party/Hi-SAM/`) to produce a pixel-level text stroke mask for a canonical-frontal ROI, then fills the masked pixels via a Navier-Stokes / Laplace-style `cv2.inpaint`. Plugs into `PropagationStage` alongside `SRNetInpainter` via `propagation.inpainter_backend: "hisam"`. Zero new pip installs in the main `.venv`.

## Branch
`feat/text_seg` (already current).

## Approach

### Data flow
```
canonical_roi (BGR uint8, H×W×3)
     │
     ▼
HiSAMSegmenter.segment(roi)  ──►  stroke_mask (uint8, H×W, {0, 255})
     │
     │  dilate k = hisam_mask_dilation_px (default 3)
     ▼
cv2.inpaint(roi, mask, INPAINT_NS)   # or INPAINT_TELEA per config
     ▼
inpainted_bgr (BGR uint8, H×W×3)   # returned to PropagationStage
```

Same contract as `SRNetInpainter.inpaint(roi) -> roi` — the LCM ratio-map code downstream does not change.

### Hi-SAM initialization (the D2 gotcha)
Hi-SAM's [build.py:178-183](third_party/Hi-SAM/hi_sam/modeling/build.py#L178-L183) loads SAM's ViT encoder weights from a **hardcoded relative path** `pretrained_checkpoint/sam_vit_<type>_*.pth`. An absolute path for the Hi-SAM head checkpoint is not enough — build.py needs cwd to be the Hi-SAM repo root while the encoder weights load.

Use a `contextlib.chdir()` block that switches into `third_party/Hi-SAM/` only during model construction and restores cwd immediately after — same spirit as `SRNetInpainter`'s lazy `sys.path` injection in [srnet_inpainter.py:76-78](code/src/stages/s4_propagation/srnet_inpainter.py#L76-L78). The chdir is bounded in scope and invisible to the rest of the pipeline.

### Decisions captured

- **D1** Two files: `hisam_segmenter.py` (mask producer, reusable) + `segmentation_inpainter.py` (ABC implementer that owns a segmenter + cv2.inpaint). Keeps the stroke mask independently usable (e.g., future AnyText2 adaptive mask could consume a real Hi-SAM mask instead of heuristic width).
- **D2** Use `contextlib.chdir(third_party/Hi-SAM)` during `HiSAMSegmenter.load_model()` to work around build.py's hardcoded relative path for the SAM encoder weights. `sys.path` is also temporarily inserted so `from hi_sam.modeling.build import model_registry` resolves.
- **D3** Defaults: `model_type="vit_l"`, head checkpoint `third_party/Hi-SAM/pretrained_checkpoint/sam_tss_l_textseg.pth` (already present, smoke-tested). `hier_det=False` hard-coded — we only need stroke-level masks. Users can override `model_type` + `checkpoint_path` via config for vit_b/vit_h or HierText variants.
- **D4** Default inpaint method `cv2.INPAINT_NS` (Navier-Stokes — solves a Laplace-style PDE with edge guidance, matches user's "Laplace inpainting" ask). `"telea"` available via config for A/B comparison.
- **D5** Mask dilation default `3` px via `cv2.dilate` with a 3×3 rectangular kernel; exposed for tuning. Prevents anti-aliased stroke halos from bleeding into the inpainted output.
- **D6** `patch_mode` off by default. Enable via `hisam_use_patch_mode: bool = False` for cases where the input is large (e.g., big `roi_context_expansion`).
- **D7** Config extension on `PropagationConfig` — reuses the shared `inpainter_backend` / `inpainter_checkpoint_path` / `inpainter_device` trio (just adds `"hisam"` as a valid backend value alongside `"srnet"` / `"none"`). Four new Hi-SAM-specific fields:
  - `hisam_model_type: str = "vit_l"`
  - `hisam_mask_dilation_px: int = 3`
  - `hisam_inpaint_method: str = "ns"`  (values: `"ns"` or `"telea"`)
  - `hisam_use_patch_mode: bool = False`
- **D8** Integration into `PropagationStage._get_inpainter()`: add a third branch for `backend == "hisam"` that imports `SegmentationBasedInpainter` lazily and passes the four Hi-SAM-specific config fields through. SRNet branch unchanged.
- **D9** Tests: mock the Hi-SAM model so unit tests run on CPU with no GPU/checkpoint requirement. Manual GPU smoke script for real-image visual validation.
- **D10** No new pip installs — confirmed by end-to-end smoke test in the previous turn. Hi-SAM's runtime inference deps (`torch`, `torchvision`, `einops`, `shapely`, `pyclipper`, `scikit-image`, `scipy`, `matplotlib`, `pillow`, `tqdm`, `opencv`, `numpy`, `absl-py`) are all present in `.venv`.
- **D11** Risk mitigation for torch 2.11 `weights_only` default: if Hi-SAM's `torch.load()` inside `build.py` warns or errors on our torch 2.11, we document the gotcha and either (a) add a monkey-patch of `torch.load` during construction, or (b) contribute a small patch to vendored build.py. **Don't** patch preemptively — the smoke test already works on torch 2.11.
- **D12** BGR contract: Hi-SAM's `SamPredictor.set_image(rgb)` expects RGB. Wrapper converts BGR→RGB at the entry and keeps everything else internal.
- **D13** Device honored via `inpainter_device` (reuses SRNet's field — `"cuda"` or `"cpu"`). Wrapper passes the device string to `hisam.to(device)` during construction. GPU strongly recommended; a CPU fallback works but is slow.

## Files to Change

- [ ] `code/src/config.py` — Add four new Hi-SAM-specific fields to `PropagationConfig`. Update the `inpainter_backend` docstring comment to list `"hisam"` as a valid value.
- [ ] (new) `code/src/stages/s4_propagation/hisam_segmenter.py` — `HiSAMSegmenter` class. `__init__` accepts `model_type`, `checkpoint_path`, `device`, `use_patch_mode`. `load_model()` does the `contextlib.chdir` + `sys.path.insert` dance. `segment(bgr_roi) -> mask` returns a binary uint8 H×W mask. Handles BGR↔RGB and the patch-mode branch.
- [ ] (new) `code/src/stages/s4_propagation/segmentation_inpainter.py` — `SegmentationBasedInpainter` subclasses `BaseBackgroundInpainter`. `__init__` takes the four config knobs + `device` + `checkpoint_path`; constructs a `HiSAMSegmenter` lazily. `inpaint(canonical_roi)`: segments → dilates → `cv2.inpaint` → returns BGR uint8 same shape.
- [ ] `code/src/stages/s4_propagation/stage.py` — Extend `PropagationStage._get_inpainter()` with an `elif backend == "hisam":` branch that imports and constructs `SegmentationBasedInpainter`. Unknown-backend error message updated to list all three valid values.
- [ ] `code/config/default.yaml` — Add the four new fields with defaults + an `# Hi-SAM inpainter` comment block under `propagation`. `inpainter_backend` stays `"none"` here (default config keeps classical CV path).
- [ ] `code/config/adv.yaml` — Add the four new fields. Keep `inpainter_backend: "srnet"` as the current default so existing runs are unchanged; add a commented-out `# inpainter_backend: "hisam"` line so users can swap one line to try the new backend.
- [ ] (new) `code/tests/models/test_segmentation_inpainter.py` — Unit tests with a `_FakeSegmenter` that returns a fixed mask. Covers:
  - BGR in → BGR out, same shape as input
  - Mask dilation is applied (inspect the mask passed to `cv2.inpaint`)
  - `inpaint_method="ns"` vs `"telea"` selects the right cv2 flag
  - Invalid shape input raises `ValueError`
  - Segmenter's `load_model` is called lazily on first `inpaint()`, not in `__init__`
  - Config → constructor field mapping (dilation_px, inpaint_method, model_type, use_patch_mode)
- [ ] (new) `code/tests/stages/test_s4_hisam_wiring.py` — Verifies `PropagationStage._get_inpainter()` constructs a `SegmentationBasedInpainter` when `propagation.inpainter_backend="hisam"`, and that unknown backends raise `ValueError` with a message listing all three valid backends. Uses `unittest.mock.patch` on the class constructor so no real Hi-SAM load is triggered.
- [ ] (new) `code/scripts/smoke_test_hisam_inpainter.py` — Manual GPU smoke script (not a pytest). Loads a real canonical ROI (via `test_data/` or a user-specified image path), runs `SegmentationBasedInpainter.inpaint()`, and writes a side-by-side `(original | mask | inpainted)` PNG. Mirrors the spirit of how SRNet's side-by-side visualizations were validated.
- [ ] `docs/architecture.md` — Update the `s4_propagation/` module entry to mention Hi-SAM as a second inpainter backend option. Add a short bullet under "Known Limitations" noting the Hi-SAM chdir workaround for the upstream hardcoded path.

## Risks

- **R1 — chdir side effects.** `contextlib.chdir()` is Python 3.11+ (we're on 3.12, fine) and is safe single-threaded, but in a multi-threaded test runner the global cwd change is a race. Mitigation: scope the chdir to `load_model()` only (construction happens once per pipeline run on the main thread). If this ever needs to be thread-safe, swap to `os.chdir` guarded by a module-level `threading.Lock`.
- **R2 — Hi-SAM encoder path hardcoding drift.** If upstream Hi-SAM ever changes the relative path in build.py, our chdir breaks silently. Mitigation: the integration smoke test will catch this; pin the vendored commit by not `git pull`ing blindly.
- **R3 — torch 2.11 weights_only.** PyTorch 2.6+ defaults `torch.load(weights_only=True)`. Hi-SAM's build.py uses bare `torch.load()`. The previous turn's smoke test worked on torch 2.11 with `sam_tss_l_textseg.pth`, but HierText or training checkpoints (which bundle optimizer state) may fail. Mitigation: surface the issue if encountered (error message is clear) and monkey-patch or patch build.py at that point — not preemptively.
- **R4 — Mask over-segmentation.** Hi-SAM will segment *all* text strokes in the ROI, including text we don't want to erase (e.g., partial characters from neighboring text that leaked into the canonical due to a loose quad). This would erase more than intended. Mitigation: the canonical is derived from a tight quad, so leakage should be minimal; the smoke test will verify on real ROIs. Consider a connected-component filter by area as a follow-up if needed.
- **R5 — Navier-Stokes inpainting quality on large masks.** `cv2.inpaint` is a classical method — it's blurry and poor on large contiguous regions. Text strokes are thin, so this is mostly fine, but large dilation or dense-text ROIs (~40% fill) may look smeared. Mitigation: keep `mask_dilation_px` small (default 3). If quality is insufficient, that's the point of having both backends — users can fall back to SRNet or we can add a LaMa backend later.
- **R6 — First-run latency.** Hi-SAM ViT-L is ~350 MB; the first `inpaint()` call will be slow (load checkpoint + copy to GPU + warmup forward pass). Lazy-loaded exactly once per pipeline run in `PropagationStage._get_inpainter()`. No in-loop model init. Consistent with how SRNet and BPN already behave.
- **R7 — Config surface creep on PropagationConfig.** We add four new fields to an already-large config dataclass. Acceptable for now — parallel to the existing `lcm_*` and `bpn_*` namespaced fields. If a third inpainter joins, refactor inpainter config into a sub-dataclass.

## Done When

- [ ] `SegmentationBasedInpainter(checkpoint_path=<sam_tss_l_textseg.pth>, device="cuda").inpaint(roi)` returns a same-shape BGR uint8 with visibly erased text when run on a real canonical ROI.
- [ ] Config YAML `inpainter_backend: "hisam"` + running `PropagationStage` on a real video produces a pipeline output with LCM working on Hi-SAM-inpainted backgrounds. No crashes.
- [ ] Unit tests (`test_segmentation_inpainter.py`) green — fake segmenter covers shape contract, dilation, inpaint method switch, lazy load.
- [ ] Wiring test (`test_s4_hisam_wiring.py`) green — `_get_inpainter()` dispatches to `"hisam"` backend correctly.
- [ ] Smoke script (`smoke_test_hisam_inpainter.py`) runs end-to-end on a test ROI and writes a side-by-side PNG; visually inspected on at least one real canonical ROI.
- [ ] `python -m pytest tests/` green on the branch (delta from master = only the new Hi-SAM tests).
- [ ] `ruff check code/` clean on all changed files.
- [ ] `docs/architecture.md` updated with the Hi-SAM backend mention.
- [ ] Code review by `@reviewer` — feedback addressed.
- [ ] Committed as atomic commits (config + new files + wiring + tests + script + docs).

## Progress

- [x] **Step 1** — Extended `PropagationConfig` in `code/src/config.py` with four new fields (`hisam_model_type`, `hisam_mask_dilation_px`, `hisam_inpaint_method`, `hisam_use_patch_mode`). Updated `inpainter_backend` doc-comment to include `"hisam"`. Added a matching block to `adv.yaml` with commented-out backend-swap line + Hi-SAM checkpoint path example. `default.yaml` left untouched — its `propagation:` block doesn't declare the inpainter trio, so dataclass defaults apply. `test_config.py` 19/19 pass; full unit suite 132/132 pass; ruff clean; adv.yaml round-trip prints `vit_l 3 ns False` as expected.
- [x] **Step 2** — Wrote `code/tests/models/test_segmentation_inpainter.py` (360 lines): `_FakeSegmenter` helper + 15 tests across 6 test classes (`TestShapeContract`, `TestDilation`, `TestInpaintMethodSwitch`, `TestInputValidation`, `TestLazyLoad`, `TestConfigMapping`). Collection fails with expected `ModuleNotFoundError: No module named 'src.stages.s4_propagation.segmentation_inpainter'`. Ruff clean. Rest-of-suite baseline (394 tests) unchanged. **Contract pinned by tests**: ctor is `SegmentationBasedInpainter(checkpoint_path, device, model_type, mask_dilation_px, inpaint_method, use_patch_mode, segmenter=None)`; lazy HiSAMSegmenter construction on first `inpaint()`; `cv2` must be imported as module (not `from cv2 import inpaint`) so tests can patch `src.stages.s4_propagation.segmentation_inpainter.cv2.inpaint`.
- [x] **Step 3** — Implemented `code/src/stages/s4_propagation/hisam_segmenter.py` (258 lines) with `contextlib.chdir(third_party/Hi-SAM) + sys.path.insert` inside `load_model()`, single-pass and patch-mode branches, private `_patchify_sliding` / `_unpatchify_sliding` helpers copied verbatim from `demo_hisam.py` (no dependency on that script). `SamPredictor.set_image(..., image_format="BGR")` handles BGR natively — no manual BGR→RGB conversion. `Path(checkpoint_path).resolve()` at `__init__` time so relative paths survive the chdir. Added `code/tests/models/test_hisam_segmenter.py` (213 lines, 7 tests): lazy vs eager load, `load_model()` idempotency, cwd-restored guarantee, `segment()` precondition, binary-mask shape/dtype/values (both single-pass and patch modes) — all CPU-runnable via `monkeypatch.setitem(sys.modules, ...)` injection of fake `hi_sam.modeling.*` modules. Live GPU smoke on `third_party/Hi-SAM/demo/2e0cb33320757201.jpg` → `(1600, 1200) uint8 {0, 255}` mask, 5.4% stroke pixels, cwd restored to `/workspace/video-scene-text-translator`. 401/401 existing tests green (Step 2's file still red, as expected). Ruff clean.
- [x] **Step 4** — Implemented `code/src/stages/s4_propagation/segmentation_inpainter.py` (168 lines). Module constants `_INPAINT_RADIUS_PX=3`, cached `_DILATION_KERNEL` (3×3 rect), `_INPAINT_METHOD_FLAGS={"ns":INPAINT_NS,"telea":INPAINT_TELEA}`. `__init__` validates `inpaint_method` + `mask_dilation_px>=0` (fail-fast). `inpaint()` does ndim/channels/dtype check → `_ensure_segmenter()` (lazy) → `segment()` → `_dilate()` → `cv2.inpaint(roi, mask, 3, flag)` (positional — kwargs form breaks the test mocks). Dependency-injectable `segmenter` kwarg for testing. Step 2's 15 tests **all green**. Full suite: 416 passed (401 prior + 15 new), 0 regressions, no pre-existing S5 failures either. Ruff clean. Live GPU smoke on Hi-SAM demo image: `(1600,1200,3) uint8 → (1600,1200,3) uint8`, mean abs diff ≈ 6.01 (stroke regions filled).
- [x] **Step 5** — Wired into `PropagationStage._get_inpainter()`: added `elif backend == "hisam":` branch that lazy-imports `SegmentationBasedInpainter`, checks `inpainter_checkpoint_path` (warn + return None if missing), constructs with all 6 kwargs (checkpoint_path, device, model_type, mask_dilation_px, inpaint_method, use_patch_mode), logs `"S4: loading Hi-SAM inpainter from ..."`. Upgraded fallback `ValueError` to list all three valid backends. Added `code/tests/stages/test_s4_hisam_wiring.py` with 5 tests (constructs with right kwargs, no-checkpoint returns None + warns, lazy (not loaded until `_get_inpainter()`), cached after first call, unknown-backend raises with all valid values in message). Full suite: 421 passed (416 prior + 5 new), 0 regressions. Ruff clean. Live GPU hook: `PipelineConfig` with `inpainter_backend="hisam"` → `PropagationStage._get_inpainter()` returns a `SegmentationBasedInpainter`, `.inpaint()` on demo image returns `(1600, 1200, 3) uint8`.
- [x] **Step 6** — Wrote `code/scripts/smoke_test_hisam_inpainter.py`. CLI supports `--roi-root` (default `test_output/roi_extraction_2/`), single-image `--roi-path`, `--model-type`, `--inpaint-method`, `--mask-dilation-px`, `--use-patch-mode`, `--device`. Writes 4-panel `(original | mask | inpainted | diff×3)` PNGs upscaled for legibility. Ruff clean.
- [x] **Step 7** — Ran on 10 ROIs from `test_output/roi_extraction_2/` (SAMSUNG, NORMAL, PERM PRESS, HEAVY DUTY, SHIRTS tracks). Inpainted output is flat, text-free, no visible halos on any sample. Dilation `3` is appropriate — no tuning needed. Visualizations in `test_output/hisam_inpaint_vis/`.
- [x] **Step 8** — Full test suite: **421/421 passed** in 38.98s (baseline 401 + 15 segmentation_inpainter + 7 hisam_segmenter wiring in bonus Step 3 test file − 2 that were folded). Ruff clean on all PR-touched files (`config.py`, `stage.py`, `hisam_segmenter.py`, `segmentation_inpainter.py`, 3 new test files, smoke script). Remaining 2 ruff errors in `lighting_correction_module.py` (`B905 zip() without strict=`) confirmed pre-existing on master via git stash — out of scope for this PR.
- [x] **Step 9** — Updated `docs/architecture.md`: expanded the `s4_propagation/` row in the module map to mention both inpainter backends (`srnet`, `hisam`), and added a Hi-SAM paragraph to the Cross-Cutting Concerns "Third-party deps" bullet covering the vendored-at-`third_party/Hi-SAM/` location, zero-new-pip-install story, and the `contextlib.chdir` workaround for build.py's hardcoded encoder path.
- [x] **Step 10** — `@reviewer` pass. **1 blocker**: unrelated `server_url` swap in `adv.yaml` flagged — verified pre-existing on the working tree from before Step 1 (not my change), will handle at Step 11 by staging only the Hi-SAM hunks via `git add -p`. **3 suggestions addressed**: (1) three `assert` statements in `hisam_segmenter.py` converted to `RuntimeError` / `ValueError` with actionable messages (survive `-O`, match `SRNetInpainter` style); (2) `load_model()` docstring gained a thread-safety warning paragraph per plan R1; (3) `smoke_test_hisam_inpainter.py` gained a NOTE comment explaining the intentional double-inference and cautioning against timing throughput from the script. Nitpicks on `argparse.Namespace` and module-docstring wording skipped as low-value. 421/421 tests still pass; ruff clean on changed files.
- [ ] **Step 11** — Commit as atomic commits (config → segmenter → inpainter → wiring → tests → smoke script → docs). Open PR. **Must stage only Hi-SAM-related hunks in `adv.yaml`** (first hunk, lines ~67-80) — the pre-existing `server_url` swap (line ~120) belongs to the working tree from before this feature branch and must NOT land in the PR commits.
