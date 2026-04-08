# Session: Expanded ROI with Scene Context — 2026-04-08

## Completed
- Planned and implemented Option C: expand AnyText2 ROI with real scene context pixels
- Added `roi_context_expansion` config field (default 0.0, adv.yaml 0.3)
- Updated `BaseTextEditor` ABC with optional `edit_region` param
- `AnyText2Editor`: targeted mask + color extraction using `edit_region`, bounds clamping
- `PlaceholderEditor`: respects `edit_region` by rendering only within sub-area
- S3: `_expanded_warp()` via translation matrix `T @ H_to_frontal`, auto-capped ratio
- Code review fixes: deduplicated `MAX_DIM` constant, added skip logging, bounds clamping
- 15 new tests (182 total passing), changelog updated

## Current State
- Branch `feat/expanded-roi` — all plan steps complete, ready to merge
- Expansion only touches S3 + editor layer; S2/S4/S5 unchanged
- `_prepare_roi` now returns scale factor (3-tuple); all callers updated
- AnyText2 dimension constants renamed to public (`MAX_DIM`, `MIN_DIM`, `ALIGN`)
- Both `default.yaml` (0.0) and `adv.yaml` (0.3) document the new field

## Next Steps
1. Test on GPU with AnyText2 server — compare quality: expansion=0 vs 0.3
2. Merge `feat/expanded-roi` to master after GPU validation
3. Investigate CoTracker OOM on 1080p+ video
4. Add more e2e test videos (different text counts, languages, resolutions)

## Decisions Made
- **Expand at S3 level, not S2**: S2's homography already maps the full frame — we just use a larger output window via translation matrix. Avoids changing `canonical_size` semantics and rippling into S4/S5.
- **Cap expansion to avoid downscaling**: `effective_ratio = min(ratio, (1024/max_dim - 1) / 2)` ensures expanded dims never trigger unnecessary downscaling in `_prepare_roi`.
- **Public constants over private duplicates**: Renamed `_MAX_DIM` → `MAX_DIM` so S3 can import it instead of maintaining a copy.

## Open Questions
- What expansion ratio works best? 0.3 is a starting point — needs GPU testing on multiple videos.
- For extreme aspect ratios (e.g., 1000x30), padding still dominates — may need per-axis expansion caps.
