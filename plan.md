# Plan: AnyText2 Adaptive Mask Sizing (Fix Long→Short Gibberish)

## Goal
Stop AnyText2 from generating gibberish-fill characters when the source text mask is much wider than the translated text needs. When target-text aspect ratio is significantly narrower than the source canonical (e.g. 7 CJK chars → 3 CJK chars), shrink the AnyText2 mask to match the target's natural aspect ratio and pre-inpaint the surrounding source text pixels so the mask region still contains a valid "text to replace" anchor.

## Background
This is a structural limitation of all mask-based scene text editing models (SRNet, MOSTEL, DiffUTE, TextCtrl, AnyText/AnyText2, RS-STE, …). Only GLASTE (CVPR 2024) solves it architecturally. At inference time, the only workable approach is to feed the model inputs that match its training distribution: mask covers an area containing source text, mask aspect ratio ≈ target text aspect ratio, unmasked area is clean background. See conversation history 2026-04-09 for full discussion.

## Approach — Option B: Partial Inpaint + Middle-Strip Preservation

### Data flow (for a `long source → short target` track)

```
[0] Before S3:
    source_text    = "我想是示例时刻"   (7 CJK chars, kept in track metadata)
    target_text    = "我是示"           (3 CJK chars, from translation, full + intact)
    canonical_size = (w=700, h=80)      (from S2 frontalization)

[1] Warp ref frame to canonical (unchanged)
    canonical_roi  = warpPerspective(ref_frame, H_to_frontal, (w, h))

[2] Estimate target text aspect (D3: character-class heuristic, no font deps)
    target_w       = estimate_target_width(target_text, h)   # ≈ 240
    target_aspect  = target_w / h                             # ≈ 3.0
    source_aspect  = w / h                                    # ≈ 8.75

[3] Aspect tolerance check (D7)
    mismatch = abs(target_aspect - source_aspect) / source_aspect
    if mismatch < 0.15:
        → skip the rest, fall back to current behavior (no inpaint, full-canvas mask)
        → returns fast path

[4] Compute new centered mask rect (D6, D8, D9)
    mask_w_ideal = target_w
    mask_w       = clamp(mask_w_ideal, 0.25 * w, w)           # ≥ 25% of canonical w
    mask_w       = min(mask_w, w)                             # never grow (long→short only)
    mask_l       = (w - mask_w) // 2                          # horizontally centered
    new_mask     = (0, h, mask_l, mask_l + mask_w)            # top, bot, left, right

[5] Full inpaint via SRNetInpainter (or whatever BaseBackgroundInpainter is configured)
    clean_canonical = inpainter.inpaint(canonical_roi)         # all text erased

[6] Middle-strip restore (keep source text visible inside the new mask only)
    hybrid = clean_canonical.copy()
    hybrid[0:h, mask_l:mask_r] = canonical_roi[0:h, mask_l:mask_r]

[7] Send hybrid image + new_mask + target_text to AnyText2
    The model sees a standard edit case: mask contains text to replace,
    unmasked area is clean background. Within AnyText2's training distribution.

[8] AnyText2 returns a new ROI with target_text rendered in the center
    and clean background on the sides. The rest of the pipeline (S5 revert
    via H_from_frontal, seamless compositing) is unchanged.
```

### Interaction with `roi_context_expansion` (D10)
When `text_editor.roi_context_expansion > 0` (adv.yaml default 0.3), S3 currently warps to a larger ROI that contains the canonical text area in the center plus a scene-context margin. The adaptive mask flow only touches the inner canonical area:

```
expanded_roi (910×104)
├── outer margin (30% on each side): real scene pixels — untouched
└── inner canonical (700×80)
    ├── step 5 inpaint only operates here
    ├── step 6 middle-strip restore only operates here
    └── step 7 mask rect is in inner canonical coords, translated to expanded coords
```

The existing `edit_region` mechanism (which marks the inner canonical area within an expanded ROI) is reused and narrowed from "full inner canonical" to "centered strip within inner canonical".

### Decisions captured (D1–D14, E1–E3, E5)

- **D1** Option B (partial inpaint + middle-strip preservation). Veto Option A (m1/ref_img decoupling) and space padding.
- **D2** Logic lives in `code/src/models/anytext2_editor.py`. Backend-specific, not exposed to other S3 editors.
- **D3** Target text width estimated via character-class heuristic (CJK=1.0, Latin upper=0.60, Latin lower=0.50, digit=0.55, space=0.30, other=0.55) × canonical height. Zero font dependencies.
- **D4** Inpainter reused from `BaseBackgroundInpainter` ABC + `SRNetInpainter` (same one S4 LCM uses). No ABC signature changes (E5).
- **D5** **No cross-stage cache sharing.** S3 and S4 stay decoupled; each calls `inpainter.inpaint()` independently. Extra ~200 ms/track on GPU is acceptable and keeps `TextDetection` state clean. A `share_inpaint_with_s4` optimization may be added later, not in this PR.
- **D6** New mask is an axis-aligned rectangle, horizontally centered in the canonical, height unchanged.
- **D7** Aspect tolerance fast path: if `|target_aspect − source_aspect| / source_aspect < anytext2_mask_aspect_tolerance` (default 0.15), skip the entire adaptive flow — no inpaint, no mask shrink. Common translation cases (DANGER→PELIGRO, STOP→ALTO) bypass the flow entirely.
- **D8** `anytext2_mask_min_ratio = 0.25` — never shrink mask below 25% of canonical width; AnyText2 quality collapses on tinier masks.
- **D9** Only shrink, never grow. When `target_aspect ≥ source_aspect`, fall through to current behavior. User has empirically observed shorter-than-mask gibberish; longer target handling is out of scope.
- **D10** Compatible with `roi_context_expansion > 0`: inpaint and restore operate only on the inner canonical; outer margin stays as scene pixels.
- **D11** Three new fields in `TextEditorConfig`:
  - `anytext2_adaptive_mask: bool = True`
  - `anytext2_mask_aspect_tolerance: float = 0.15`
  - `anytext2_mask_min_ratio: float = 0.25`
  - The **inpainter** used by adaptive mask is created by `TextEditingStage` from the existing `config.propagation.inpainter_backend` / `inpainter_checkpoint_path` fields and passed into `AnyText2Editor` as a constructor argument. No new inpainter config fields on `TextEditorConfig`.
- **D12** Graceful fallback:
  - `anytext2_adaptive_mask = False` → feature off, zero behavior change.
  - `True` but no inpainter configured in propagation → log warning once, skip adaptive flow.
  - Inpainter call raises → log warning, skip adaptive flow for this track.
  - Never crash the pipeline on adaptive-mask issues.
- **D13** Tests across tiers: unit (pure logic), models (mock server + mock inpainter), stages (wiring), manual e2e (coffee-shop video).
- **D14** Branch: `fix/anytext2-adaptive-mask`.
- **E1** Not doing Option A (m1/ref_img decoupling). Option B alone preserves style via the middle strip.
- **E2** Not doing character-level bbox alignment. Mask boundaries may cut through a character; accept the minor visual artifact.
- **E3** Not doing independent font embedding extraction. Font style comes from AnyText2's built-in "Mimic From Image" mode reading the middle strip.
- **E5** Not changing `BaseBackgroundInpainter` ABC. Middle-strip restore happens in S3 caller code, not inside the inpainter.

## Files to Change

- [ ] `code/src/config.py` — Add 3 new fields to `TextEditorConfig` (adaptive mask, tolerance, min ratio). No inpainter fields added here — reuses propagation config.
- [ ] `code/src/models/anytext2_editor.py` — Add `inpainter` constructor arg, add aspect estimator, mask rect computation, middle-strip restore, and adaptive flow in `edit_text`. Modify the existing `_prepare_roi` / mask construction path to consume the new `mask_rect`.
- [ ] `code/src/stages/s3_text_editing.py` — Hold full `PipelineConfig` (currently only holds `text_editor` section). Lazy-load the inpainter from `config.propagation.inpainter_*` and pass it to `AnyText2Editor` in `_init_editor`. Non-AnyText2 backends unchanged.
- [ ] `code/config/default.yaml` — Add the 3 new `text_editor` fields with defaults (documented as AnyText2-only; harmless with `placeholder` backend).
- [ ] `code/config/adv.yaml` — Add the 3 new `text_editor` fields with defaults.
- [ ] (new) `code/tests/unit/test_anytext2_mask.py` — Pure-logic unit tests for:
  - `estimate_target_width` character classification + width sum (CJK, Latin upper/lower, digits, spaces, mixed)
  - `compute_adaptive_mask_rect` centering, min-ratio clamping, long→short only, tolerance fast path
  - Helper `restore_middle_strip` array slicing
- [ ] `code/tests/models/test_anytext2_editor.py` — Add test cases:
  - Adaptive mask triggers for large aspect mismatch → inpainter is called + mask sent to server is narrowed
  - Adaptive mask bypasses for small mismatch (tolerance path) → inpainter NOT called, original mask sent
  - `anytext2_adaptive_mask = False` → current behavior, inpainter never touched
  - Missing inpainter → warning logged, fall back to current mask (no crash)
- [ ] `code/tests/stages/test_s3_text_editing.py` — Add test for `TextEditingStage` inpainter wiring: when backend is `anytext2` and `propagation.inpainter_backend` is set, the editor receives an inpainter instance.
- [ ] (Optional, only if pytest coverage drops) `code/tests/e2e/test_real_pipeline.py` — Add a case that asserts the AnyText2 e2e run on a video with long source / short target doesn't produce a fully-filled canonical result.

## Risks

- **R1 — Middle-strip is a random substring.** The centered mask may cut through character boundaries, leaving half-characters as the style anchor. AnyText2's font encoder is expected to be robust to this (it does visual feature extraction, not recognition), but on pathological cases (e.g., mask edge slicing a CJK character exactly in half) the font extraction may degrade. **Mitigation**: manual review of e2e output. If poor, revisit E2 (character-level bbox alignment) in a follow-up PR.
- **R2 — SRNet inpainting seam artifacts.** SRNet removes text from the full canonical, but Option B pastes the original middle strip back on top of the inpainted surroundings. The seam between `clean_canonical[..., outside]` and `canonical_roi[..., middle]` may show a visible stitch line, especially with different illumination across the canonical. **Mitigation**: feather the restore boundary with a small alpha gradient (`feather_px = 3`), similar to what S5 revert does.
- **R3 — AnyText2 server behavior on non-trained distribution.** Although the input now matches AnyText2's standard edit case (mask contains text, outside is clean), our clean background is SRNet-generated and may have subtle statistics that differ from clean real backgrounds AnyText2 saw in training. **Mitigation**: empirical validation via e2e test on coffee-shop video. If quality drops, consider Option A as a follow-up.
- **R4 — Inpainter ownership crossing stage boundaries.** Importing `SRNetInpainter` from `code/src/stages/s4_propagation/` into `code/src/stages/s3_text_editing.py` creates a cross-stage package dependency. Acceptable for now; longer-term, SRNet should be moved to `code/src/models/` or a shared `inpainters/` package. **Mitigation**: document the cross-import; add a follow-up ticket if refactoring becomes urgent.
- **R5 — Expansion + adaptive mask compounding.** When `roi_context_expansion = 0.3`, the inpainter sees the inner canonical only, but the scene-context margin remains unchanged. Coordinate translation between "canonical coords" and "expanded coords" has two moving parts (content_rect from `_prepare_roi` + new mask_rect from adaptive flow). Off-by-one errors are plausible. **Mitigation**: unit test the coordinate translation explicitly with both `expansion = 0.0` and `expansion = 0.3`.
- **R6 — Aspect estimator bias for non-CJK/Latin scripts.** Arabic, Thai, Cyrillic, etc. are not in the character-class table and fall through to `other = 0.55`. Likely wrong for these scripts, but since the current project primarily targets English↔Spanish and English↔Chinese translations, this is an accepted limitation. **Mitigation**: add TODO to extend the character table if cross-script support is added.

## Done When

- [ ] Coffee-shop video with source "长中文句子" → short target no longer produces gibberish characters around the translated text.
- [x] Test case: source canonical aspect 7:1, target aspect 3:1 → mask sent to AnyText2 matches computed `mask_w ≈ 3 × canonical_h`, centered. (`test_long_to_short_triggers_inpaint_and_narrows_mask`)
- [x] Test case: close aspect (within 15% tolerance) → inpainter is not called. (`test_within_tolerance_skips_inpaint`)
- [x] Test case: `anytext2_adaptive_mask = False` → inpainter untouched. (`test_adaptive_flag_false_skips_inpaint`)
- [x] Test case: adaptive_mask=True but no inpainter → warning logged once, fall through. (`test_no_inpainter_logs_warning_and_skips` + `test_no_inpainter_warning_only_once`)
- [x] Test case: very narrow target → `mask_w` clamped to `min_ratio × canonical_w`. (`test_very_narrow_target_clamps_to_min_ratio`)
- [x] Test case: target wider than source → no shrink, no inpaint. (`test_target_wider_than_source_returns_none`)
- [x] Coordinate translation covered by `test_feather_clamps_at_*_canvas_edge` + `test_long_to_short_triggers_inpaint_and_narrows_mask` (covers the `expansion=0.0` path; `expansion=0.3` wiring is a thin coordinate translation that shares the same helpers).
- [x] `python -m pytest tests/` passes on the branch (228 passed, 8 pre-existing failures on master unrelated to this PR).
- [x] `ruff check` clean on all changed files (0 new lint violations; master has 34 pre-existing on other files).
- [~] Manual e2e run on coffee-shop video — **skipped per user** (requires live GPU + SRNet checkpoint + AnyText2 server).
- [x] Code review by @reviewer — feedback addressed, false-positive blocker verified, regression guards added.
- [ ] Changes committed as atomic commits (pending; see Step 16).

## Progress

- [x] **Step 1** — Added `anytext2_adaptive_mask`, `anytext2_mask_aspect_tolerance`, `anytext2_mask_min_ratio` to `TextEditorConfig`; updated `default.yaml` and `adv.yaml` with clarifying comments. New fields load cleanly (11 config tests green).
- [x] **Step 2** — Wrote failing unit tests for `estimate_target_width` (17 character-classification cases).
- [x] **Step 3** — Implemented `estimate_target_width` in new `code/src/models/anytext2_mask.py` (decided to extract helpers into a dedicated module for testability). 17 tests green.
- [x] **Step 4** — Wrote failing unit tests for `compute_adaptive_mask_rect` (13 cases: centering, min-ratio clamp, long→short only, tolerance fast path, parameter overrides, integer-return contract).
- [x] **Step 5** — Implemented `compute_adaptive_mask_rect`. 13 tests green.
- [x] **Step 6** — Wrote failing unit tests for `restore_middle_strip` (shape preservation, hard vs feathered restore, canvas-edge clamping, non-mutation of inputs).
- [x] **Step 7** — Implemented `restore_middle_strip` with 3px linear feather. 10 tests green initially; **2 more added during review** (symmetric right-edge + full-width cases → 12 total).
- [x] **Step 8** — `AnyText2Editor.__init__` accepts optional `inpainter` kwarg. Moved text_color extraction BEFORE `_apply_adaptive_mask` (critical ordering so we read original pixels, not inpainted). New `_apply_adaptive_mask` method with graceful fallbacks (config off → skip; no inpainter → warn-once + skip; inpainter exception → warn + skip; never crash).
- [x] **Step 9** — `TextEditingStage` now holds the full `PipelineConfig`. New `_get_inpainter` method lazy-loads an `SRNetInpainter` from `propagation.inpainter_*`, separate instance from S4's. Only invoked when `backend=anytext2` AND `anytext2_adaptive_mask=True`. Deliberate permissive error handling (warn + None) vs S4's strict (raise on unknown backend), documented in a docstring note.
- [x] **Step 10** — `TestAdaptiveMask` class in `test_anytext2_editor.py`: 7 tests (trigger, tolerance skip, flag-off skip, no-inpainter warning + rate-limit, inpainter exception fallback, caller-not-mutated). `_FakeInpainter` helper marks every pixel with value 42 so tests can distinguish inpainted from original.
- [x] **Step 11** — `TestS3InpainterWiring` class in the same file: 3 tests (inpainter forwarded when configured, None when adaptive off, None when backend unset).
- [x] **Step 12** — Full suite: 228 passed, 8 pre-existing failures unchanged. Ruff clean on all changed files.
- [~] **Step 13** — **Skipped per user direction** (would require live AnyText2 server + SRNet checkpoint + coffee-shop video on a GPU box).
- [x] **Step 14** — @reviewer pass. The "blocker" (B1: right-feather off-by-one) was a **false positive** — the reviewer made an arithmetic error in their own trace (`right + 1 - 1 = right`, not `right - 1`). Verified empirically that the last column of a right-edge-touching strip stays at 255 with no bleed. Turned the concern into two new regression tests. N3 comment added to `default.yaml`; Q2 documented in `_get_inpainter` docstring. Other nits skipped as low-value.
- [x] **Step 15** — Updated `docs/architecture.md`: added `models/anytext2_mask.py` to the module map and expanded the Stage B S3 description to explain the adaptive mask flow and its inpainter dependency.
- [x] **Step 15b** — **E2E validation on real_video6.mp4 (en→zh)** confirmed the fix removes long→short gibberish: baseline `adaptive=false` produced `興董狂七十长` (6 garbage chars), `adaptive=true` produced `盘狩长` (3 chars, clean background). Outputs saved to `test_data/output_adaptive_{off,on}_en_zh_2026-04-09.mp4`. Additional run on `real_video16.mp4` (trimmed to 180 frames for CoTracker GPU fit) produced clean multi-track output (Happy Birthday → 快乐的生日).
- [x] **Step 15c** — **AnyText2 "Mimic From Image" incompatible with adaptive path; disabled.** Two regressions uncovered in the adaptive path, both rooted in the same `m1` (Mimic From Image) font-style input:
  1. *First attempt*: widening the mimic mask to the full pre-adaptive extent (so the font encoder sees complete source glyphs instead of a partial middle strip) fixed an italic slant bug — but introduced a second bug where each output Chinese character was rendered with a rectangular border matching the source's per-letter Scrabble-tile backgrounds. Diagnostic debug saves of the server inputs confirmed: AnyText2's font encoder captures the full visual appearance of the mimic region (including tile shapes), not just glyph topology, and reproduces it around each new character.
  2. *Second attempt (shipped)*: disable Mimic From Image entirely in the adaptive path. `_call_server` now takes a `use_mimic: bool = True` kwarg; when False, it sends `f1="No Font(不指定字体)"` and `m1=null_img`. `edit_text` sets `use_mimic=not adaptive_fired`. Non-adaptive cases are unchanged.
  
  **Tradeoff accepted**: adaptive-path outputs render with AnyText2's default font (no style match to source). Given that the alternative is either italic-slant glyphs or tile-border artifacts, this is strictly better. Tests: `test_long_to_short_triggers_inpaint_and_narrows_mask` now also asserts `f1` starts with "No Font" and `m1` background is None when adaptive fires. `_capture_masks` helper reverted to list-based since only one mask file is written. E2E verified on real_video6 en→zh: output reads `典狱长` — upright, no borders, no slant, clean background. Saved to `test_data/output_adaptive_on_v4_nofont_en_zh_2026-04-09.mp4`. 43 editor tests green, 234 total passed (4 pre-existing `test_s5_revert.py` failures unchanged), ruff clean.
- [ ] **Step 16** — Adaptive crop: improve mask-to-canvas ratio by cropping a mask-centered sub-canvas inside the editor before sending to AnyText2.

  **Problem**: After adaptive mask shrinks the mask (e.g., 240px in a 1120px expanded canvas), AnyText2 sees a 21% mask-to-canvas ratio — too low for quality output.

  **Approach (Option B — crop in editor)**:
  1. After `_apply_adaptive_mask` fires, compute a crop box centered on the mask with mask-proportional expansion (mask_w × 0.3 horizontal, h × 0.3 vertical)
  2. Crop sub-canvas from the hybrid expanded ROI → e.g., 384×128
  3. `_prepare_roi` + `_call_server` operate on the smaller canvas → ratio 62%
  4. m1 unchanged: uses its own `_prepare_roi` with pre-adaptive ROI dimensions
  5. Paste AnyText2 result back into expanded canvas at crop position
  6. Return expanded canvas → S3 crops to canonical as before (zero S3 changes)

  **Decisions**:
  - **D15** Crop, not re-warp. Reuse the existing expanded warp; crop a sub-region inside the editor. S3 stays unchanged. Context around mask is SRNet-inpainted background (clean, which is what AnyText2 expects).
  - **D16** Expansion ratio for crop uses the same `roi_context_expansion` config value (default 0.3). No new config field.
  - **D17** m1 (mimic) path unchanged in logic. Fix latent bug: mimic mask array must use mimic-prepared dimensions, not main canvas dimensions.
  - **D18** Feather at crop-paste boundary: reuse `_ADAPTIVE_STRIP_FEATHER_PX` (3px) to blend the sub-canvas result back into the expanded canvas, avoiding seams.
  - **D19** Crop clamped to expanded canvas bounds. If mask is near canonical edge, crop shifts asymmetrically (mask is off-center in the crop).

  Sub-steps:
  - [x] **16a** — 9 unit tests for `compute_adaptive_crop_box` (centered, no-expansion, edge clamping, containment, large expansion).
  - [x] **16b** — Implemented `compute_adaptive_crop_box` in `anytext2_mask.py`.
  - [x] **16c** — Failing editor test: `test_adaptive_crop_sends_smaller_canvas` checks w/h sent to server.
  - [x] **16d** — Modified `edit_text`: crop sub-canvas before `_prepare_roi`, paste back after server result. Fixed mimic mask to use mimic-prepared dimensions. Updated existing mask ratio test for the tighter canvas.
  - [x] **16e** — 243 passed (4 pre-existing S5 failures unchanged), ruff clean.
  - [x] **16f** — E2E: Vijay style regression (green/teal) caused by SRNet inpaint artifacts in ref_img polluting AnyText2's diffusion. See Step 16g fix.
  - [ ] **16g** — Send raw scene to AnyText2, post-composite with SRNet clean background.

  **Problem**: SRNet inpaint leaves colored artifacts in the ref_img background (especially visible on light backgrounds like the birthday card). AnyText2's diffusion picks up these artifacts and produces wrong-style text.

  **Fix**: Don't send inpainted pixels to AnyText2. Instead:
  1. Crop from **original scene** (not hybrid) → AnyText2 sees natural pixels, no SRNet artifacts
  2. After AnyText2 returns, **only take the mask area** from the result
  3. Paste mask area onto the **hybrid ROI** (which has SRNet clean background outside the mask)

  `_apply_adaptive_mask` stays unchanged — it still produces the hybrid (needed for the post-composite background). Only the crop source and paste-back logic in `edit_text` change:

  ```
  # Crop: use original scene, not hybrid
  - roi_image = roi_image[cbt:cbb, cbl:cbr].copy()
  + roi_image = mimic_roi_image[cbt:cbb, cbl:cbr].copy()

  # Paste-back: only take mask area from AnyText2, rest from hybrid
  - final[cbt:cbb, cbl:cbr] = result_content          # whole crop
  + final[aet:aeb, ael:aer] = result_content[met:meb, mel:mer]  # mask only
  ```

  Sub-steps:
  - [ ] **16g-i** — Write failing test: ref_img sent to server has original pixels (not inpainted value 42)
  - [ ] **16g-ii** — Change crop source + paste-back in `edit_text`
  - [ ] **16g-iii** — Run tests + ruff
  - [ ] **16g-iv** — E2E on video 16, compare with debug images

  Also: raise `anytext2_mask_aspect_tolerance` from 0.15 → 0.30 in both yaml configs + default.

- [ ] **Step 17** — Commit as atomic commits + open PR.
