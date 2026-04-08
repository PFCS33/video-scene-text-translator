# Cross-Language Scene Text Replacement in Video

CMPT 743 Visual Computing Lab II final project (SFU). Replace scene text in video frames across languages, preserving font style, perspective, and lighting consistency.

## Pipeline Overview

The pipeline follows STRIVE's frontalization-first design: all ROIs are warped to a canonical frontal rectangle before editing and propagation, then warped back to the original perspective for compositing.

```
Video Frames
     |
     v
S1: Detection + Tracking + Selection
     |  list[TextTrack]  (dense detections, reference selected)
     v
S2: Frontalization
     |  list[TextTrack]  (H_to_frontal / H_from_frontal on each detection)
     v
S3: Text Editing
     |  list[TextTrack]  (edited_roi in canonical frontal space)
     v
S4: Propagation
     |  dict[frame_idx, list[PropagatedROI]]  (lighting-adapted, alpha-masked)
     v
S5: Revert
     |  list[np.ndarray]  (final output frames)
     v
Output Video
```

## Key Data Types

These dataclasses flow through the pipeline. Each stage reads and enriches them. All defined in [`code/src/data_types.py`](code/src/data_types.py).

### BBox
Axis-aligned bounding box. Fields: `x`, `y`, `width`, `height`. Provides `to_slice()` for numpy array indexing and `area()`. Derived from `Quad` via `quad.to_bbox()` — used for fast IoU matching and array cropping where perspective accuracy isn't needed.

### Quad
Four corner points defining a text region polygon. `points: np.ndarray` of shape `(4, 2)` in `[TL, TR, BR, BL]` order. This is the "real" geometry — the OCR backend (EasyOCR or PaddleOCR) produces quads, and homographies are computed from quad corners. Can be perspective-distorted, rotated, or skewed.

### TextDetection
Everything known about a text region in a single frame. Geometry (`quad`, `bbox`), OCR data (`text`, `ocr_confidence`), quality metrics (`sharpness_score`, `contrast_score`, `frontality_score`, `composite_score`), and homography fields (`H_to_frontal`, `H_from_frontal`, `homography_valid`). Geometry and OCR are set by S1; homography fields are set by S2.

### TextTrack
The central data structure — a tracked text region across multiple frames. Groups `TextDetection` objects that refer to the same physical text instance. Key fields:
- `detections: dict[int, TextDetection]` — frame_idx -> detection, dense after S1 gap-filling
- `reference_frame_idx: int` — best frame for editing (highest quality)
- `reference_quad: Quad` — read-only property, returns `detections[reference_frame_idx].quad`
- `canonical_size: tuple[int, int]` — (width, height) of the canonical frontal rectangle, set by S2
- `edited_roi: np.ndarray` — edited text image in canonical frontal space, set by S3
- `source_text` / `target_text` — original and translated text strings

### PropagatedROI
A lighting-adapted edited ROI ready for compositing into a specific frame. Contains `roi_image` (canonical frontal, color-corrected), `alpha_mask` (feathered blending mask), and `target_quad` (where to place it in the original frame). Created by S4, consumed by S5.

### PipelineResult
Final output: `tracks`, `output_frames`, `fps`, `frame_size`.

## Stage Details

### S1: Detection + Tracking + Selection

Detects text in video frames, tracks detections across frames, selects the best reference frame per track, and fills gaps via optical flow. Split into four submodules: `detector.py`, `tracker.py`, `selector.py`, `stage.py`.

**Steps:**

1. **Detect** — Runs the configured OCR backend (EasyOCR or PaddleOCR, set via `detection.ocr_backend`) on every Nth frame (`frame_sample_rate`). Filters detections by OCR confidence and minimum text area. Computes quality metrics per detection: sharpness (Laplacian variance), contrast (Otsu interclass variance), frontality (quad-to-bbox area ratio), and a weighted composite score.

2. **Track** — Groups detections across frames into `TextTrack` objects via greedy IoU matching (threshold 0.3). Unmatched detections start new tracks. Translates source text via deep-translator (GoogleTranslator with MyMemory fallback) on track creation.

3. **Select reference** — Picks the best frame per track for editing. Hard pre-filters: OCR confidence >= 0.7, top-K by sharpness. Then scores remaining candidates: `0.7 * contrast + 0.3 * frontality`. After selection, updates `source_text` and `target_text` from the reference frame's OCR (more reliable than the first detection's OCR).

4. **Fill gaps** — Optical flow (Farneback or Lucas-Kanade, configurable) propagates quad corners bidirectionally from the reference frame to all frames without OCR detections. Creates synthetic `TextDetection` entries with `ocr_confidence=0.0` for gap-filled frames.

**I/O:**

```
Input:
  frames: list[tuple[int, np.ndarray]]       # (frame_idx, BGR H*W*3 uint8)

Output:
  list[TextTrack]                            # complete — every frame has a TextDetection
    track.track_id: int
    track.source_text: str                   # OCR text from reference frame, e.g. "DANGER"
    track.target_text: str                   # translated text, e.g. "PELIGRO"
    track.source_lang / target_lang: str
    track.reference_frame_idx: int           # best frame for editing
    track.reference_quad: Quad               # property — returns detections[reference_frame_idx].quad
    track.canonical_size: None               # not yet computed (set by S2)
    track.edited_roi: None                   # not yet computed (set by S3)
    track.detections: dict[int, TextDetection]   # DENSE — all frames
      det.frame_idx: int
      det.quad: Quad                         # points: np.ndarray (4x2, float32)
      det.bbox: BBox                         # x, y, width, height (axis-aligned)
      det.text: str
      det.ocr_confidence: float              # 0.0 for gap-filled frames
      det.sharpness_score: float             # 0.0 for gap-filled frames
      det.contrast_score: float              # 0.0 for gap-filled frames
      det.frontality_score: float            # 0.0 for gap-filled frames
      det.composite_score: float             # 0.0 for gap-filled frames
      det.H_to_frontal: None                 # not yet computed (set by S2)
      det.H_from_frontal: None               # not yet computed (set by S2)
      det.homography_valid: False            # not yet computed (set by S2)
```

---

### S2: Frontalization (Pure Geometry)

Computes a homography from each frame's text quad to a canonical frontal rectangle. No pixels are warped — only 3x3 matrices are stored for downstream use.

**Steps:**

1. For each track, derives a **canonical frontal rectangle** from the reference quad's average edge lengths: `[[0,0], [w,0], [w,h], [0,h]]`. Sets `track.canonical_size = (w, h)`.

2. For each detection in the track, computes `cv2.findHomography(quad.points, canonical_rect)` via RANSAC. Each frame's homography is computed independently — directly from its own quad to the canonical rectangle (the reference frame is not a waypoint).

3. Stores `H_to_frontal` (frame -> canonical) and `H_from_frontal` (canonical -> frame) directly on each `TextDetection`.

**I/O:**

```
Input:
  tracks: list[TextTrack]                    # from S1, with dense detections

Mutates:
  track.canonical_size: tuple[int, int]      # (width, height) of canonical rect
  det.H_to_frontal: np.ndarray              # (3x3, float64) frame -> canonical frontal
  det.H_from_frontal: np.ndarray            # (3x3, float64) canonical frontal -> frame
  det.homography_valid: bool                 # True if RANSAC succeeded

Output:
  list[TextTrack]                            # same objects, homography fields populated
```

---

### S3: Text Editing

Warps the reference frame's text region to canonical frontal space, passes it through the text editor, and stores the result.

**Steps:**

1. For each track, retrieves the reference frame and reference detection.

2. If `H_to_frontal` is valid and `canonical_size` is set: warps the **entire reference frame** to canonical frontal space via `cv2.warpPerspective(frame, H_to_frontal, (w, h))`. This produces a clean, upright ROI regardless of the original perspective.

3. Fallback (no homography): crops raw bbox region from the reference frame.

4. Passes the ROI to `BaseTextEditor.edit_text(roi, target_text)`, which returns an edited ROI with target text rendered. Backends: `placeholder` (cv2.putText), `anytext2` (style-preserving via Gradio API). Configured via `text_editor.backend` in YAML.

5. Stores result in `track.edited_roi`.

**I/O:**

```
Input:
  tracks: list[TextTrack]
  frames: dict[int, np.ndarray]

Mutates:
  track.edited_roi: np.ndarray               # canonical frontal space, shape (h, w, 3) uint8
                                              # where (w, h) = track.canonical_size

Output:
  list[TextTrack]                            # same objects, edited_roi populated
```

---

### S4: Propagation

Adapts the edited reference ROI to each frame's lighting conditions using histogram matching on pixel-aligned frontalized ROIs.

**Steps:**

1. For each track, for each frame with a detection:
   - If `H_to_frontal` is valid: warps the full frame to canonical frontal space via `cv2.warpPerspective(frame, H_to_frontal, canonical_size)`. The result is pixel-aligned with the edited ROI.
   - Fallback: crops bbox from the frame.

2. **Histogram matches** the edited ROI's luminance to the target frame's luminance. Uses CDF-based matching on the Y channel in YCrCb color space. Because both images are in the same canonical space, pixels correspond spatially — the matching is more accurate than comparing misaligned perspectives.

3. Creates a **feathered alpha mask**: center = 1.0, edges linearly fade to 0.0 over a border of 10% of the smallest dimension. This ensures smooth blending when composited back into the frame.

4. Packs the result into a `PropagatedROI`.

**I/O:**

```
Input:
  tracks: list[TextTrack]
  frames: dict[int, np.ndarray]

Output:
  dict[int, list[PropagatedROI]]             # frame_idx -> list of ROIs for that frame
    PropagatedROI:
      .frame_idx: int
      .track_id: int
      .roi_image: np.ndarray                 # canonical frontal, lighting-adapted (h*w*3, uint8)
      .alpha_mask: np.ndarray                # feathered mask (h*w, float32, 0.0-1.0)
      .target_quad: Quad                     # where to place in original frame
```

---

### S5: Revert (De-Frontalization + Compositing)

Warps each propagated ROI from canonical frontal space back to the original frame's perspective and alpha-blends it into the frame.

**Steps:**

1. Builds a `tracks_by_id` lookup for fast access to track data.

2. For each frame, for each `PropagatedROI` in that frame:
   - Looks up the detection via `track.detections[frame_idx]` to get `H_from_frontal`.
   - Computes **bounded target region**: `target_bbox = target_quad.to_bbox()`, clamped to frame bounds. This avoids warping to full frame size.
   - Creates a **translation matrix** `T` to offset into bbox-local coordinates.
   - Warps ROI and alpha mask via `cv2.warpPerspective(roi, T @ H_from_frontal, (bbox.w, bbox.h))`. The coordinate chain is: canonical space -> frame space (`H_from_frontal`) -> bbox-local coords (`T` shifts origin to bbox top-left).
   - **Alpha-blends** only within the bbox region: `frame[bbox_slice] = frame * (1 - alpha) + roi * alpha`.

3. Collects all frames in sorted frame index order.

**I/O:**

```
Input:
  frames: dict[int, np.ndarray]              # original video frames
  propagated_rois: dict[int, list[PropagatedROI]]
  tracks: list[TextTrack]                    # for H_from_frontal lookup

Output:
  list[np.ndarray]                           # final output frames, sorted by frame_idx
                                             # same shape as input frames, text replaced
```

## Environment Setup

### Prerequisites
- Conda (Miniconda, Miniforge, or Anaconda)
- NVIDIA GPU with CUDA 12.x driver (`nvidia-smi` to verify)
- AnyText2 Gradio server running (managed separately — see below)

### 1. Create conda environment

```bash
conda create -n vc_final python=3.11 -y
conda activate vc_final
```

### 2. Install base dependencies

```bash
cd code
pip install -r requirements/base.txt
```

### 3. Install PyTorch (GPU)

Match the CUDA index to your driver version. Check with `nvidia-smi`:

| Driver CUDA | PyTorch index |
|-------------|---------------|
| 12.4–12.8   | `cu124`       |
| 13.0+       | `cu130`       |

```bash
# Example for CUDA 12.4–12.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verify: `python -c "import torch; print(torch.cuda.is_available())"`

### 4. Install EasyOCR

```bash
pip install easyocr
```

### 5. Install PaddlePaddle + PaddleOCR

PaddlePaddle GPU must also match your CUDA driver:

| Driver CUDA | PaddlePaddle index |
|-------------|-------------------|
| 12.4–12.8   | `cu126`           |
| 13.0+       | `cu130`           |

```bash
# Example for CUDA 12.4–12.8:
pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install paddleocr
```

> **Gotcha:** PaddlePaddle GPU built for CUDA 13.0 will silently fall back to CPU on a 12.x driver, then crash with a OneDNN `NotImplementedError`. Always match the CUDA index to your driver.

### 6. Install CoTracker

```bash
cd third_party
git clone https://github.com/facebookresearch/co-tracker.git
cd co-tracker
pip install -e .

# Download checkpoints
mkdir -p checkpoints && cd checkpoints
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth
cd ../../..
```

### 7. Install gradio_client (for AnyText2)

```bash
pip install "gradio_client>=1.5.0"
```

### 8. AnyText2 server

AnyText2 runs in a **separate** conda env (Python 3.10) to avoid dependency conflicts.
See [`third_party/install_anytext2.sh`](third_party/install_anytext2.sh) for setup instructions.

Once the server is running, set the URL in your config YAML:

```yaml
text_editor:
  backend: "anytext2"
  server_url: "http://<host>:<port>/"
```

### Quick verification

```bash
conda activate vc_final
cd code
python -m pytest tests/ -v          # All tests should pass
ruff check .                        # Lint check
```

## Usage

```bash
# Activate conda environment
conda activate vc_final
cd code

# Run the pipeline (advanced config: CoTracker + PaddleOCR + AnyText2)
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True python scripts/run_pipeline.py \
  --config config/adv.yaml \
  --input <video> --output <out> \
  --source-lang en --target-lang es

# Run with default config (Farneback + EasyOCR + placeholder editor)
python scripts/run_pipeline.py --input <video> --output <out> --source-lang en --target-lang es

# Run tests
python -m pytest tests/ -v

# Lint
ruff check .
```

## Project Structure

```
code/
  src/
    pipeline.py                     # Orchestrator: wires S1-S5
    data_types.py                   # Core dataclasses (TextTrack, TextDetection, etc.)
    config.py                       # YAML config loading + validation
    video_io.py                     # VideoReader / VideoWriter
    stages/
      s1_detection/                 # S1: Detection + Tracking + Selection
        detector.py                 #   OCR backends (EasyOCR / PaddleOCR), quality metrics
        tracker.py                  #   IoU matching, optical flow gap-filling
        selector.py                 #   Reference frame selection, translation
        stage.py                    #   Orchestrator
      s2_frontalization.py          # S2: Canonical rect homography
      s3_text_editing.py            # S3: Warp to frontal, edit text
      s4_propagation.py             # S4: Frontalize + histogram match + alpha mask
      s5_revert.py                  # S5: De-frontalize + bounded warp + composite
    models/
      base_text_editor.py           # ABC for Stage A models
      placeholder_editor.py         # cv2.putText placeholder
      anytext2_editor.py            # AnyText2 via Gradio API
    utils/
      geometry.py                   # Homography, quad metrics, canonical rect
      image_processing.py           # Sharpness, contrast, histogram matching
      optical_flow.py               # Farneback + Lucas-Kanade wrappers
  config/
    default.yaml                    # All configurable parameters
  tests/                            # 162 unit + integration tests
  scripts/
    run_pipeline.py                 # CLI entry point
```
