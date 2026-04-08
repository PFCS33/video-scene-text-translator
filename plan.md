# Plan: Add LaMa Background Inpainter Backend for S4

## Context
The S4 propagation stage uses a background inpainter to produce text-free backgrounds for LCM (Lighting Correction Module) ratio computation. Currently only SRNet is supported. SRNet's inpainting quality on textured backgrounds (e.g., burlap, tiles) produces artifacts that degrade the LCM lighting ratio, causing washed-out results. LaMa (Large Mask Inpainting, WACV 2022) is a resolution-robust general-purpose inpainter with better texture synthesis via Fourier convolutions.

## Goal
Add LaMa as a second `BaseBackgroundInpainter` backend, selectable via `inpainter_backend: "lama"` in config. Extend the inpainter ABC to accept an optional text mask (LaMa needs one; SRNet doesn't).

## Approach

### Key Design Decisions
1. **Extend `BaseBackgroundInpainter.inpaint()` with optional `text_mask` kwarg** — S4 generates the mask via Otsu thresholding + dilation, passes it to whichever inpainter is configured. SRNet ignores it; LaMa uses it. A `uses_text_mask` class-level flag controls whether S4 bothers generating the mask.
2. **Upscale small ROIs** to min 256px (LaMa's training resolution) before inference, downscale after.
3. **Direct TorchScript loading** — load `big-lama.pt` via `torch.jit.load()`, no pip package. Install script downloads the 206MB model.

### Text Mask Generation (in `stage.py`)
- Convert canonical ROI to grayscale → Otsu threshold → auto-invert if majority region (ensure text = minority) → dilate (3×3 kernel, 2 iterations)
- Only computed when `inpainter.uses_text_mask is True`

```python
def _generate_text_mask(self, canonical_roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(canonical_roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:  # auto-invert: text should be minority
        binary = 255 - binary
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(binary, kernel, iterations=2)
```

### LaMa Inference Pipeline (in `lama_inpainter.py`)
1. Upscale if min(H,W) < 256
2. BGR→RGB, normalize to [0,1] float32 tensor
3. Mask: uint8 [0,255] → float32 [0,1] tensor
4. Pad both to mod-8 (reflect padding on right/bottom)
5. Concatenate to (1, 4, H, W), forward through TorchScript model
6. Crop padding, downscale if upscaled, RGB→BGR uint8

### LaMa Model Details
- Model: `big-lama.pt` TorchScript (~206MB), Apache 2.0 license
- Download: `https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt`
- Input: (1, 4, H, W) float32 — 3ch RGB image + 1ch mask, all [0,1], dims divisible by 8
- Output: (1, 3, H, W) float32 [0,1]
- Load: `torch.jit.load(path, map_location=device)` — compatible with PyTorch 1.13+

## Files to Change
- [ ] `code/src/stages/s4_propagation/base_inpainter.py` — Add `text_mask: np.ndarray | None = None` kwarg (keyword-only) + `uses_text_mask: bool = False` class attr
- [ ] `code/src/stages/s4_propagation/srnet_inpainter.py` — Accept and ignore `text_mask`
- [ ] (new) `code/src/stages/s4_propagation/lama_inpainter.py` — LaMa backend (~80 lines)
- [ ] `code/src/stages/s4_propagation/stage.py` — Add `_generate_text_mask()`, pass mask at inpaint call sites (lines 192, 211), add `"lama"` dispatch in `_get_inpainter()` (after line 76)
- [ ] `code/src/config.py` — Update comment on `inpainter_backend` to list "lama"
- [ ] `code/config/adv.yaml` — Add commented-out LaMa config alternative
- [ ] (new) `third_party/install_lama.sh` — Download `big-lama.pt` (~206MB) to `third_party/lama/`
- [ ] `code/tests/stages/test_s4_propagation.py` — Tests for mask generation, LaMa dispatch, mask-passing flow
- [ ] (new) `code/tests/stages/test_lama_inpainter.py` — Unit tests for LaMa wrapper (mocked TorchScript model)
- [ ] `CLAUDE.md` — Add LaMa to gotchas/stack sections

## Risks
- **Otsu mask quality**: May misclassify on low-contrast or multi-color backgrounds. Mitigated by auto-invert heuristic (text = minority region).
- **Small ROI quality**: ROIs at 64px are below LaMa's 256×256 training resolution. Mitigated by upscaling before inference.
- **TorchScript deprecation**: Deprecated in PyTorch 2.9 but `torch.jit.load()` still works. Not a concern for course project timeline.
- **Breaking ABC change**: `inpaint()` signature changes. Only 2 subclasses (SRNet + LaMa), both updated in this plan.

## Done When
- [ ] `inpainter_backend: "lama"` loads and runs LaMa on a canonical ROI
- [ ] `inpainter_backend: "srnet"` still works unchanged (no regression)
- [ ] Text mask generation produces reasonable binary masks on synthetic ROIs
- [ ] Small ROIs (<256px) are upscaled before LaMa inference
- [ ] `third_party/install_lama.sh` downloads checkpoint successfully
- [ ] All tests pass: `cd code && python -m pytest tests/ -v`
- [ ] Lint passes: `ruff check code/`

## Progress
- [ ] Step 1: Extend `BaseBackgroundInpainter` ABC — add `text_mask` kwarg and `uses_text_mask`
- [ ] Step 2: Update `SRNetInpainter` — accept and ignore `text_mask`
- [ ] Step 3: Create `LaMaInpainter` backend
- [ ] Step 4: Update `stage.py` — mask generation, mask passing, "lama" dispatch
- [ ] Step 5: Create `third_party/install_lama.sh`
- [ ] Step 6: Update config files (`config.py` comment, `adv.yaml` example)
- [ ] Step 7: Write tests (mask generation, LaMa wrapper, dispatch, integration)
- [ ] Step 8: Run full test suite + lint, update CLAUDE.md
