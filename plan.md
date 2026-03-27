# Plan: Pipeline Architecture Refactor (S1–S5)

## Goal
Refactor the Stage B pipeline to align with STRIVE's frontalization-first design: all ROIs warped to a canonical frontal rectangle before editing and propagation. Also clean up data flow (fold FrameHomography into TextDetection, split S1 monolith, move gap-filling to where it belongs).

## Approach

### Key design decisions (settled in discussion)
1. **Canonical rectangle, not reference perspective** — S2 homographies map every frame's quad to an axis-aligned rectangle (derived from quad dimensions), not to the reference frame's quad. This gives Stage A models clean frontal input and enables pixel-aligned comparison in S4.
2. **Matrices only, warp on-the-fly** — S2 stores 3×3 matrices in TextDetection. Downstream stages warp when they need pixels. Keeps S2 as pure geometry, avoids memory bloat.
3. **Gap-filling is tracking** — Optical flow propagation moves from S2 to S1. S1 outputs complete TextTracks (every frame has a detection). S2 no longer mutates TextTrack.
4. **Single source of truth** — FrameHomography eliminated. H_to_frontal / H_from_frontal live on TextDetection. No more parallel data structures to manually join.

### Stage I/O (after refactor)

```
S1: Detection + Tracking + Selection
  Input:  frames: dict[int, np.ndarray]              # frame_idx → BGR image (H×W×3, uint8)
  Output: list[TextTrack]                             # complete — every frame has a TextDetection
          │
          ├── track.detections: dict[int, TextDetection]   # dense (all frames)
          │     ├── .quad: Quad (points: np.ndarray 4×2 float32)
          │     ├── .bbox: BBox (x, y, width, height)
          │     ├── .text: str
          │     ├── .ocr_confidence: float                 # 0.0 for optical-flow-filled frames
          │     └── .H_to_frontal / .H_from_frontal: None          # not yet computed
          ├── track.reference_frame_idx: int
          ├── track.reference_quad: Quad
          ├── track.source_text / target_text: str
          └── track.canonical_size: None                    # not yet computed

S2: Frontalization (pure geometry)
  Input:  tracks: list[TextTrack], frames: dict[int, np.ndarray]
  Output: list[TextTrack]                             # same tracks, with homography fields populated
          │
          ├── track.canonical_size: tuple[int, int]         # (width, height) of canonical rectangle
          └── track.detections[i]:
                ├── .H_to_frontal: np.ndarray (3×3, float64)   # frame → canonical frontal
                ├── .H_from_frontal: np.ndarray (3×3, float64) # canonical frontal → frame
                └── .homography_valid: bool

S3: Text Editing
  Input:  tracks: list[TextTrack], frames: dict[int, np.ndarray]
  Output: list[TextTrack]                             # same tracks, with edited_roi populated
          │
          └── track.edited_roi: np.ndarray                  # canonical frontal space (W×H×3, uint8)
              (warped via H_to_frontal before editing,
               dimensions = track.canonical_size)

S4: Propagation
  Input:  tracks: list[TextTrack], frames: dict[int, np.ndarray]
  Output: dict[int, list[PropagatedROI]]              # frame_idx → list of ROIs for that frame
          │
          └── PropagatedROI:
                ├── .frame_idx: int
                ├── .track_id: int
                ├── .roi_image: np.ndarray                  # canonical frontal, lighting-adapted (W×H×3)
                ├── .alpha_mask: np.ndarray                  # feathered mask (W×H, float32)
                └── .target_quad: Quad                      # where to place in original frame
          (internally: warp each frame's ROI to canonical via H_to_frontal,
           histogram-match against edited_roi, both now pixel-aligned)

S5: Revert (de-frontalization + compositing)
  Input:  frames: dict[int, np.ndarray]
          propagated_rois: dict[int, list[PropagatedROI]]
          tracks: list[TextTrack]                      # for H_from_frontal lookup
  Output: list[np.ndarray]                             # final output frames, sorted by frame_idx
          (warp each PropagatedROI via H_from_frontal to bounded region,
           alpha-blend into frame)
```

### Patterns to follow
- Existing `compute_homography()` in `utils/geometry.py` — reuse, just change dst_points
- Existing `frontalize_roi()` in S2 — already does `cv2.warpPerspective` with `H_to_frontal`, currently unused
- TDD: write failing tests first, then implement
- One logical change per commit

## Files to Change

### Data layer
- [ ] `code/src/data_types.py` — Add `H_to_frontal`, `H_from_frontal`, `homography_valid` to TextDetection. Add `canonical_size: tuple[int, int] | None` to TextTrack. Remove `FrameHomography` dataclass.

### S1: Split + gap-filling
- [ ] (new) `code/src/stages/s1_detection/__init__.py` — Re-export `DetectionStage`
- [ ] (new) `code/src/stages/s1_detection/detector.py` — EasyOCR wrapper, per-frame detection, quality metrics. Extracted from current `s1_detection.py`
- [ ] (new) `code/src/stages/s1_detection/tracker.py` — IoU matching + optical flow gap-filling. IoU matching from current S1, optical flow from current S2
- [ ] (new) `code/src/stages/s1_detection/selector.py` — Reference frame selection + translation. Extracted from current S1
- [ ] `code/src/stages/s1_detection.py` — Delete after splitting into submodules

### S2: Pure frontalization
- [ ] `code/src/stages/s2_frontalization.py` — Remove `track_quad_across_frames()`, `_propagate_quads()`, gap-fill in `run()`. Change homography dst_points to canonical rectangle. Write H into TextDetection. New return type: `list[TextTrack]`
- [ ] `code/src/utils/geometry.py` — Add `canonical_rect_from_quad(quad: Quad) -> np.ndarray`

### S3: Frontal editing
- [ ] `code/src/stages/s3_text_editing.py` — Warp reference ROI to canonical frontal via `H_to_frontal` before passing to editor. Use `track.canonical_size` for output dimensions.

### S4: Aligned propagation
- [ ] `code/src/stages/s4_propagation.py` — Warp each frame's ROI to canonical frontal via `H_to_frontal` before histogram matching. Compare aligned images.

### S5: Cleaner revert
- [ ] `code/src/stages/s5_revert.py` — Remove `all_homographies` param (read H from TextDetection via tracks). Warp to bounded region (target quad bbox) instead of full frame.

### Wiring
- [ ] `code/src/pipeline.py` — Update stage connections: S2 no longer returns separate dict, S5 no longer takes `all_homographies`
- [ ] `code/src/config.py` — Move `optical_flow_method` and flow params from `FrontalizationConfig` to `DetectionConfig`
- [ ] `code/config/default.yaml` — Mirror config structure change

### Tests
- [ ] `code/tests/test_data_types.py` — Tests for new TextDetection fields, remove FrameHomography tests
- [ ] `code/tests/test_s1_detection.py` — Tests for gap-filling, update for submodule imports
- [ ] `code/tests/test_s2_frontalization.py` — Tests for canonical rect homography, remove tracking tests
- [ ] `code/tests/test_s3_text_editing.py` — Test frontalized ROI input
- [ ] `code/tests/test_s4_propagation.py` — Test with aligned frontalized ROIs
- [ ] `code/tests/test_s5_revert.py` — Test bounded warp, update for no homography dict
- [ ] `code/tests/test_geometry.py` — Tests for `canonical_rect_from_quad()`
- [ ] `code/tests/test_pipeline_integration.py` — Update for new stage signatures
- [ ] `code/tests/conftest.py` — Update fixtures if needed

## Risks
- **S1 submodule split** may break import paths in tests and other stages — need to update all `from src.stages.s1_detection import ...` references
- **Canonical rectangle aspect ratio** — need to handle edge cases (near-zero width/height quads, degenerate quads)
- **S5 bounded warp** — translation matrix adjustment (`H_adjusted = T @ H_from_frontal`) must be correct or compositing misaligns. Need careful test with known coordinates.
- **Config migration** — moving optical flow params from `frontalization` to `detection` section is a breaking change to `default.yaml`. Existing custom configs will break.
- **Test count** — currently 101 tests. Goal: no regressions, likely +10-15 new tests for canonical rect, gap-filling, bounded warp.

## Done When
- [x] FrameHomography fully removed — `grep -r "FrameHomography" code/` returns nothing
- [x] S2 does NOT mutate track.detections — gap-filling is S1's job
- [x] S5 does NOT take `all_homographies` param
- [x] S3 receives frontalized (canonical rect) ROI, not raw bbox crop
- [x] S4 compares aligned ROIs (both in canonical frontal space)
- [x] S5 warps to bounded region, not full frame
- [x] All tests pass — 129 passed (`python -m pytest tests/ -v`)
- [x] Lint clean (`ruff check code/`)
- [x] Code review approved (@reviewer) — 7 findings, all fixed
- [x] Changes committed as 8 atomic commits

## Progress
- [x] Step 1: Data types — modify TextDetection, add canonical_size to TextTrack, remove FrameHomography
- [x] Step 2: Geometry — add `canonical_rect_from_quad()` to utils/geometry.py
- [x] Step 3: S1 split — extract detector.py, tracker.py, selector.py + move optical flow gap-filling
- [x] Step 4: S2 refactor — canonical rect homography, write H into TextDetection, new return type
- [x] Step 5: S3 update — warp reference ROI to frontal before editing
- [x] Step 6: S4 update — frontalize frame ROIs before histogram matching
- [x] Step 7: S5 update — remove homography param, bounded-region warp
- [x] Step 8: Pipeline wiring + config migration + FrameHomography removal
- [x] Step 9: Code review — all 7 findings fixed, 129 tests passing, lint clean

## Future (not in this refactor)
- S4: Per-pixel lighting ratio (classical LCM equivalent) — enabled by aligned ROIs. Needs inpainting to isolate background before computing ratio, otherwise artifacts at text edges.
- S2: Temporal smoothing of homography parameters — moving average to reduce jitter. Requires decomposing homography into translation/rotation/scale before smoothing (can't average raw 3×3 matrices).
- S1: Optical flow gap-filling propagates quads to ALL frames, including frames where text is genuinely absent (occluded, out of view, camera cut). This causes false replacements. Fix options: (1) validate tracked quads — reject if area changes >50% between frames or quad becomes degenerate/self-intersecting, (2) appearance verification — check contrast/sharpness in tracked region to confirm text is still present. Option 1 is simpler; option 2 is more robust.
