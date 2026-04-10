# Session: AnyText2 Adaptive Mask E2E + Italic/Border Fix — 2026-04-10

## Completed
- E2E testing on real_video6 (WARDEN→典狱长) and real_video16 (Birthday→生日, Happy→快乐的, etc.) with adaptive mask on/off
- Discovered and fixed italic-slant regression: AnyText2's "Mimic From Image" font encoder read partial letter fragments through the narrow adaptive mask as italic. Fix: decouple m1 (font mimic) from ref_img — m1 gets pre-adaptive ROI + wide mask, ref_img gets the hybrid + narrow mask (commit `94ef4b2`)
- Investigated rectangular tile-border issue when m1 sees full source: AnyText2's font encoder captures full visual appearance (including Scrabble-tile backgrounds) and reproduces them. Accepted as source-specific artifact — non-tile sources (e.g., birthday card on wood) render clean
- Removed `anytext2_mask_min_ratio=0.25` which was the actual cause of "Birthday"→"生日" gibberish (clamped 2-char mask to 3 chars wide). Heuristic estimate was correct all along (commit `067ab47`)
- Explored and rejected: Pillow font-ratio width estimation (cross-script ratio inflates CJK masks), ROI re-crop around shortened mask (black background + style loss from paste-back)
- Organized test_data into input/ and output/ subfolders, cleaned up old videos

## Current State
- Branch `fix/anytext2-adaptive-mask` at `067ab47`, 3 commits ahead of previous session
- Adaptive mask flow: SRNet inpaint + middle-strip restore + narrowed mask + decoupled m1 mimic
- No min_ratio floor — trusts the character-class heuristic directly
- 233 tests pass (4 pre-existing S5 failures unchanged), ruff clean
- Output videos in `test_data/output/`: 3 videos (off v6, on v6, on v16)
- Font file downloaded: `third_party/fonts/NotoSansSC-Regular.ttf` (not used in code yet)

## Next Steps
1. Push branch + open PR for the adaptive mask feature
2. Optionally: improve width estimation with Pillow absolute formula (`tw × canonical_h / th`) for same-script variable-width text — but heuristic works well enough for now
3. Test on more videos to validate tile-border artifact frequency

## Decisions Made
- **Wide mimic (fix 1) over No Font fallback**: keeps font-style matching. Tile-border artifact is source-specific (Scrabble tiles), acceptable tradeoff
- **Remove min_ratio entirely, no replacement floor**: the heuristic gives accurate widths for CJK (~1.0×h per char); any floor would interfere with correct estimates
- **Rejected Pillow ratio approach**: `(tw/sw) × canonical_w × (sh/th)` over-widens CJK masks because scene Latin font width doesn't predict AnyText2 CJK rendering width. Absolute approach (`tw × canonical_h / th`) is better for cross-script but the heuristic already matches it
- **Rejected ROI re-crop**: cropping ref_img around shortened mask to improve mask-to-canvas ratio caused black background artifacts and broke font-style matching. The mask-to-canvas ratio (28%) is an AnyText2 limitation, not fixable at our layer without side effects

## Open Questions
- Does the 28% mask-to-canvas ratio (from roi_context_expansion=0.3 + adaptive narrowing) cause consistent quality issues, or was the Birthday case an outlier?
- Should Pillow absolute estimation replace the heuristic for same-script cases where variable-width chars matter (e.g., "WILLING" vs "IIIIIII")?
