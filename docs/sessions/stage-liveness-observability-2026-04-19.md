# Session: Stage Liveness Observability — 2026-04-19

## Completed
- Diagnosed systemic gap: S5 pipeline hang (>1000s, no UI error) is same class of bug as the S3 fix in commit `5fd4a51` but applied more generally.
- Wrote `plan.md` covering three-layer observability-only fix (no cancellation, no state-flip).
- Implemented three layers in parallel via @coder agents:
  - **Stage layer** — S4/S5 `try/except` + elapsed logging + 30s heartbeats; S5 refiner `torch.load` wrap. 446 pytest pass.
  - **Server layer** — `_LivenessWatchdog` daemon thread in `pipeline_runner.py`; env var `PIPELINE_LIVENESS_TIMEOUT_S` (default 180s); fires `src_logger.error(...)` on silence. 97 pytest pass (87 baseline + 10 new).
  - **Client layer** — `stalledMs: number` in `JobStreamState`; `<StageProgress>` stall badge. 158 web tests pass (129 baseline + 29 new).
- @reviewer review passed with 2 warnings + 3 suggestions; all 5 addressed (S5 first-seen log ordering, watchdog counter ordering, S4 exit log unification, `0.05` floor comment, STALL_THRESHOLD_MS sync-point JSDoc).
- All tests + lint + type-check clean after fixes.

## Current State
- Working tree on `feat/web-client` with uncommitted changes across `code/src/stages/`, `server/app/pipeline_runner.py`, `server/tests/test_pipeline_runner.py`, `web/src/lib/stages.ts`, `web/src/hooks/useJobStream.ts`, `web/src/components/StageProgress.tsx`, `web/src/App.tsx`, plus two web test files.
- Dev server running (uvicorn :8000 + Vite :5173); uvicorn `--reload` has picked up the latest code.
- `plan.md` Steps 1–5 marked `[x]`; Step 7 (commit + push) pending — user will handle manually. Step 6 (E2E) deferred.

## Next Steps
1. Commit the working-tree changes (suggested: three atomic commits — stage / server / client) and push `feat/web-client`.
2. E2E manual test whenever convenient — submit a real video and confirm healthy run produces no stall badge / no watchdog log; a forced-slow run (`PIPELINE_LIVENESS_TIMEOUT_S=30`) surfaces both the red "no progress" log line and the stall badge.
3. Out-of-scope follow-up: cancellation tokens (Option C in plan.md's Out of Scope section) — separate plan.

## Decisions Made
- **Pure observability, no cancellation.** Watchdog log-only; does NOT synthesize `ErrorEvent`, flip `record.status`, or kill the worker thread. Real exceptions still go through `JobManager._run_job`'s existing catch at `server/app/jobs.py:343`.
- **Decision 1C (hybrid logging):** per-track INFO entry/exit + 30s heartbeat + DEBUG per-ROI. Avoids `<LogPanel>` flood while keeping drill-down available.
- **Decision 3A (state-field for stall):** `stalledMs` as reducer state, not component-local. Single source of truth so future surfaces can reuse.
- **Flat 180s timeout + env override**, not per-stage thresholds. Heartbeats are the forcing function; per-stage only if observed need.
- **Preserved existing swallows** in S5 `predict_delta_H` and `_pre_inpaint_region` — upgraded log level only (DEBUG → WARNING), kept non-fatal behavior.
- **Branch:** stayed on `feat/web-client` rather than forking a new `fix/` branch (user preference).

## Open Questions
- If S3's AnyText2 single-region waits ever exceed 180s, a poll-loop heartbeat inside `anytext2_editor.py` is the follow-up — tracked only if observed.
- Recovery-from-stall: options laid out in plan.md's Out of Scope (A manual restart / B relax DELETE / C cancellation tokens / D external health probe). Recommended sequencing: A today, C when there's appetite.
