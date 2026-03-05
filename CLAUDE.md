# Cross-Language Scene Text Replacement in Video

CMPT 743 Visual Computing Lab II final project (SFU). Team: Hebin Yao, Yunshan Feng, Liliana Lopez.

## Project Goal

Replace scene text in video frames across languages (e.g., English "DANGER" → Spanish "PELIGRO") automatically, preserving font style, perspective, and lighting consistency across frames.

## Project Structure

```
vc_final/
├── _refs/
│   ├── pipeline.png          # Pipeline architecture diagram (5 stages)
│   └── report.pdf            # Milestone presentation (11 pages)
├── code/                     # Stage B pipeline implementation
│   ├── config/default.yaml
│   ├── src/
│   │   ├── pipeline.py       # Orchestrator: wires S1→S5
│   │   ├── data_types.py     # Core dataclasses (BBox, Quad, TextTrack, etc.)
│   │   ├── config.py         # YAML config loading + validation
│   │   ├── video_io.py       # VideoReader / VideoWriter
│   │   ├── stages/
│   │   │   ├── s1_detection.py      # EasyOCR + translation + reference selection
│   │   │   ├── s2_frontalization.py # Optical flow + homography
│   │   │   ├── s3_text_editing.py   # Stage A model wrapper
│   │   │   ├── s4_propagation.py    # Histogram matching
│   │   │   └── s5_revert.py        # Inverse homography + alpha compositing
│   │   ├── models/
│   │   │   ├── base_text_editor.py  # ABC for Stage A models
│   │   │   └── placeholder_editor.py
│   │   └── utils/
│   │       ├── geometry.py          # Homography, quad metrics
│   │       ├── image_processing.py  # Sharpness, contrast, histogram matching
│   │       └── optical_flow.py      # Farneback + Lucas-Kanade
│   ├── tests/                # 84 tests, all passing
│   ├── scripts/run_pipeline.py
│   └── requirements.txt
└── CLAUDE.md                 # This file
```

## Implementation Stages

### Stage A — Cross-Language Text Editing Model (separate work, not in code/)
- RS-STE: cross-language fine-tuning (main focus, training loop re-implemented)
- AnyText2: diffusion-based, imported for pipeline integration
- Stage A models are consumed via `BaseTextEditor` interface in `src/models/`

### Stage B — Basic Video Pipeline (IMPLEMENTED in code/)
Uses classical CV methods. 5 stages:
1. **S1 Detection**: EasyOCR → detect text → IoU-based tracking → Google Translate → score frames → pick reference
2. **S2 Frontalization**: Optical flow (Farneback default) → track quads → `cv2.findHomography` per frame
3. **S3 Text Editing**: Call Stage A model via `BaseTextEditor.edit_text(roi, target_text)` → returns edited ROI
4. **S4 Propagation**: YCrCb luminance histogram matching → adapt edited ROI to each frame's lighting
5. **S5 Revert**: Inverse homography warp → alpha blending with feathered edges → composite into frame

### Stage C — Full STRIVE Pipeline (NOT YET IMPLEMENTED)
- Replace homography with STTN (Spatial-Temporal Transformer Network)
- Replace histogram matching with TPM (Temporal Propagation Module)

## Key Architecture Decisions

- **Central data structure**: `TextTrack` flows through all 5 stages. S1 creates it, S2-S5 enrich it.
- **Stage A abstraction**: `BaseTextEditor` ABC in `src/models/base_text_editor.py`. To integrate a real model, subclass it and change `text_editor.backend` in config. No pipeline code changes needed.
- **Config-driven**: All parameters in `config/default.yaml`. CLI overrides via `scripts/run_pipeline.py`.
- **All frames in memory**: Works for short clips. Needs sliding-window for long videos.
- **Detections keyed by frame_idx** (dict, not list) for O(1) lookup.

## Environment

- **Conda env**: `vc_final` (Python 3.11)
- **Activate**: `conda activate vc_final`
- **Core deps installed**: numpy, opencv-python, PyYAML, Pillow, pytest, pytest-cov
- **NOT yet installed** (install when needed): `easyocr`, `googletrans==4.0.0-rc1`
- **Run tests**: `cd code && python -m pytest tests/ -v`
- **Run pipeline**: `python scripts/run_pipeline.py --input video.mp4 --output out.mp4 --source-lang en --target-lang es`

## Known Limitations (Stage B)

- Track grouping is IoU-based — breaks with large camera motion
- Optical flow drifts over long sequences
- Histogram matching is global per ROI (no spatially varying lighting)
- Placeholder editor uses OpenCV putText (crude, no style matching)
- `googletrans` is unofficial/unreliable — config supports swapping to `google-cloud-translate`

## What's Next

- Integrate real Stage A model (RS-STE or AnyText2) into `src/models/`
- Install easyocr + googletrans and test end-to-end on real video
- Stage C: STTN for frontalization, TPM for propagation
- Evaluation metrics and cross-model comparison (due Apr 3)
