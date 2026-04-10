# Session: AnyText2 Adaptive Mask — Canvas Crop + Artifact Fix — 2026-04-10

## Completed
- Implemented adaptive canvas crop: after mask shrinks, crop the canvas sent to AnyText2 centered on the mask with mask-proportional expansion (reuses `roi_context_expansion`). Improves mask-to-canvas ratio from ~21% to ~62%.
- Fixed mimic mask dimension bug: mimic mask array now uses mimic-prepared dimensions instead of main canvas dimensions.
- Discovered SRNet inpaint color artifacts (cyan/teal noise at text edges) polluting AnyText2's style output on light backgrounds. Root cause: colored speckle noise in SRNet's text-removal output.
- Fixed with bilateral filter (`d=9, sigmaColor=75, sigmaSpace=75`) on SRNet output before compositing.
- Tried and rejected "raw scene" approach (send original scene to AnyText2, post-composite with SRNet): eliminated cyan from ref_img but SRNet artifacts still leaked from the hybrid background outside the mask in the final composite.
- Skipped middle-strip restore: send fully clean (text-free) background to AnyText2 instead of restoring original text in mask. AnyText2 generates from scratch guided by m1 style. Cleaner results.
- Added `ANYTEXT2_DEBUG_DIR` env var for saving AnyText2 server inputs (ref_img, ref_mask, m1_img, m1_mask).
- Deleted unused `third_party/fonts/` directory.
- Added CHANGELOG entry for the full adaptive mask feature.
- Merged master into branch (alignment refinement, S5 temporal smoothing), then merged branch to master and pushed.

## Current State
- Branch `fix/anytext2-adaptive-mask` merged to `master` at `c64b397`
- Adaptive mask flow: SRNet inpaint → bilateral filter → skip middle-strip → crop to mask → send to AnyText2 → paste back
- 243 tests pass (4 pre-existing S5 failures unchanged), ruff clean
- E2E validated on real_video6 (WARDEN→典狱长) and real_video16 (7 tracks en→zh)
- Disable with `anytext2_adaptive_mask: false` in config

## Next Steps
1. Test on more videos to validate bilateral filter effectiveness across different backgrounds
2. Consider raising `anytext2_mask_aspect_tolerance` from 0.15 to 0.25 if borderline cases (like Vijay at 20%) cause issues
3. Birthday (生日) AnyText2 output quality is still poor — this is a model limitation for short (2-char) text in narrow masks, not our code

## Decisions Made
- **Crop in editor, not re-warp in S3**: reuse existing expanded warp, crop a sub-region inside the editor. S3 stays unchanged. Simpler, one warp, one inpaint.
- **Bilateral filter over raw scene approach**: tried sending raw scene pixels to AnyText2 (no SRNet artifacts in ref_img), but SRNet artifacts still leaked from the hybrid background in the final composite. Bilateral filter on SRNet output is simpler and fixes the root cause.
- **Skip middle-strip restore**: sending fully clean background to AnyText2 is better than restoring original text inside the mask. AnyText2 generates from scratch guided by m1 font style. Avoids any residual SRNet artifacts inside the mask.
- **Keep tolerance at 0.15**: user wants to keep it for now, may raise to 0.25 later if needed.
- **Character-class heuristic is accurate enough**: CJK=1.0 is precise for the en→zh use case. Latin estimation (0.50-0.60) has ±20% variance but only matters for target text, which is mainly CJK.

## Open Questions
- Should tolerance be raised to avoid borderline cases like Vijay (20% mismatch)?
- Birthday (2-char "生日") consistently produces poor AnyText2 output — is this a mask-size limitation or can the prompt/parameters be tuned?
