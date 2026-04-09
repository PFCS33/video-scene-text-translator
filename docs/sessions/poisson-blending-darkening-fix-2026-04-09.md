# Session: Poisson Blending Darkening Fix — 2026-04-09

## Completed
- Analyzed root cause of LCM darkening (~8-9 pt drop): neural inpainters (SRNet, LaMa) regress-to-mean → biased ratio map < 1.0
- Planned a log-domain mean-centering fix with separate global scale from raw canonicals (committed to `feat/lama-inpainter`)
- **Teammate shipped a different, better fix on master**: replaced alpha blending with `cv2.seamlessClone` (Poisson) in S5 revert + 5% bbox expansion for interior mask requirement (commit `a5cca7d`)
- E2E verified: Scrabble tile "GUARDIAN" result now has natural brightness matching surrounding burlap, no washed-out effect
- Analyzed interaction between Poisson blending and LCM/BPN — confirmed BPN fully preserved, LCM's spatial component preserved via gradients, LCM's global bias (the unreliable part) overridden by frame border

## Current State
- Branch `master` at `a5cca7d` includes Poisson compositing in S5 (`composite_roi_into_frame_seamless`) + bbox expansion in `warp_roi_to_frame`
- `feat/lama-inpainter` branch contains the now-obsolete log-domain LCM fix plan — can be closed or repurposed
- Darkening issue resolved without modifying LCM or inpainters

## Next Steps
1. Close/delete the log-domain LCM fix plan on `feat/lama-inpainter` (it's superseded)
2. Optional cleanup: disable LCM's global scaling explicitly since Poisson overrides it (not strictly necessary)
3. Test Poisson fix on more videos (different textures, lighting conditions, text sizes)
4. Check for Poisson edge cases: very small ROIs, high-contrast neon text, color-cast scenes

## Decisions Made
- **Poisson > LCM log-domain fix**: Content-agnostic, one-place change, handles any brightness/color mismatch (LCM + inpainting + AnyText2 style + SRNet output), not just inpainting bias
- **Fix at compositing layer, not at source**: Rejects STRIVE paper's "TPM handles all photometry end-to-end" philosophy — adds a safety net at the final compositing step instead
- **5% bbox expansion is mandatory**: `cv2.seamlessClone` requires strictly interior mask; expansion guarantees zero-alpha border
- **Fallback to alpha blending** when source+center doesn't fit in destination frame (edge case handled in `composite_roi_into_frame_seamless`)

## Open Questions
- Does Poisson blending attenuate text contrast noticeably on any real video? (Theoretically ~10% contrast reduction if LCM had k=0.9 global scale)
- Should LCM's global scaling be explicitly disabled now that Poisson overrides it, for cleaner code?
- Will Poisson's color-bleeding cause problems on high-saturation text (e.g., neon signs)?
