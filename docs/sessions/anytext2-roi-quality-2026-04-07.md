# Session: AnyText2 ROI Quality Fix — 2026-04-07

## Completed
- Diagnosed root cause of black corner artifacts and low-quality AnyText2 output
- Discovered AnyText2 requires 64-aligned dimensions (SD architecture), which our code didn't respect
- Replaced `_clamp_dimensions` with `_prepare_roi`: upscale to 512+, 64-align via padding, return content_rect
- Localized edit mask to content region only (padding = alpha=0, content = alpha=255)
- Added `anytext2_min_gen_size` config field (default 512)
- Updated YAML configs, tests (29 passing), lint clean
- Code review: fixed misleading mock sizes, added invariant comments

## Current State
- Branch `feat/anytext2-integration` — all plan steps complete, committed and pushed
- 167 tests passing (4 PaddleOCR failures are pre-existing `wordfreq` not installed)
- AnyText2 editor now sends properly sized, 64-aligned images with localized masks
- Needs real-world testing on GPU server to confirm quality improvement

## Next Steps
1. Test on remote GPU with real video to verify black corner artifacts are gone
2. Merge `feat/anytext2-integration` to master
3. Investigate CoTracker OOM on 1080p+ video
4. Consider Option C (expand ROI with real scene context from S2) if quality still needs improvement

## Decisions Made
- **Upscale to 512, not 768/1024**: 512 is AnyText2's training resolution — sweet spot for quality vs latency
- **Pad to 64-multiples (not crop)**: AnyText2 server crops via `w-(w%64)`, silently losing content pixels. Pre-aligning via padding avoids this.
- **Localized mask over full mask**: Full mask caused AnyText2 to regenerate padding regions → black corners. Localizing mask to content rect anchors the padding as background context.
- **All changes in anytext2_editor.py**: No S2 or pipeline changes needed. Option C (S2 context expansion) deferred as future improvement.
- **`_MIN_DIM % _ALIGN == 0` invariant**: Documented in code comment — if either constant changes, 64-alignment could silently break.

## Open Questions
- Does the quality improvement hold on real video at various ROI sizes? Need GPU testing.
- For extreme aspect ratios (e.g., 1000×30), padding is still a large fraction of the image — may need Option C for these cases.
