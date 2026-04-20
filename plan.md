# Plan: Stage Liveness Observability

## Goal
Make stage hangs visible across three layers (stage code, server, client) so
no stage in any future run can silently consume 1000+ seconds with zero
feedback. Pure observability — no cancellation, no synthetic errors, no
status flip. The user sees "stage 5 stalled at 3m12s, no progress for 180s"
instead of a frozen tile with no signal.

Context: the S3 `feat(s3): add per-region logging + try/except` fix
(commit `5fd4a51`) solved the same class of problem at the per-region level
for S3. This plan extends that pattern to S4/S5 and adds a server watchdog
+ client stall indicator as defense-in-depth.

## Approach

### Layer 1 — Stage layer (Python, additive timing + heartbeats)

Apply a **hybrid logging pattern** (decision 1C) to the unbounded inner
loops in S4 and S5:
- Per-track INFO entry/exit logs (coarse, matches S3's spirit).
- Periodic heartbeat INFO log every ~30s inside the inner loop:
  `"S5 composite: 120/681 ROIs, elapsed 45s"`.
- DEBUG per-ROI detail so drill-down is available without flooding `<LogPanel>`.
- `try/except` with `time.monotonic()` elapsed log around each long call;
  re-raise so the normal failure path still fires (matches S3 fix).

S1 already has stage-level elapsed logs. S2 is pure matrix math (fast,
skip). S3 has per-region wraps (commit 5fd4a51, skip). Only S4 and S5 +
the S5 refiner's `torch.load` need changes.

### Layer 2 — Server watchdog (log-only, no cancel)

Add a daemon thread in `pipeline_runner.run_pipeline_job` that tracks
`time.monotonic() - last_emit_ts`. If the gap exceeds
`PIPELINE_LIVENESS_TIMEOUT_S` (default 180s, env-overridable), emit
`src_logger.error("no progress for Xs, stage may be hung")` — which flows
through the existing `_PipelineLogHandler` → SSE `log` event →
`<LogPanel>` shows a red line. Watchdog resets on every `emit()` call, so
healthy stages with heartbeats never trigger it. Watchdog does NOT
synthesize `ErrorEvent`, does NOT flip `record.status`, does NOT cancel
the worker — real exceptions still use the existing
`JobManager._run_job` catch path (`server/app/jobs.py:343`).

### Layer 3 — Client stall indicator (state-field, decision 3A)

Add `stalledMs: number` to `JobStreamState`. Inside the existing
`setInterval` tick in `useJobStream`, compare `activeStageElapsedMs` to
`STALL_THRESHOLD_MS` (e.g. 180_000) and update `stalledMs` in the same
setState call. Clear on stage change / terminal (piggy-backs on the
existing `activeStageElapsedMs = 0` reset sites). `<StageProgress>`
reads `stalledMs` and renders a warning badge on the active tile when
> 0. Single source of truth so a future `<StatusBand>` chip / `<LogPanel>`
pill can reuse it without re-deriving the threshold.

## Files to Change

### Stage layer
- [x] `code/src/stages/s4_propagation/stage.py` — wrap `inpainter.inpaint()` calls at lines 231 + 251 with per-track INFO entry/exit + elapsed timing + `try/except`. Add periodic heartbeat (30s) inside the main for-loop. No behavior change on success.
- [x] `code/src/stages/s5_revert/stage.py` — wrap `predict_delta_H` (line 533), `_pre_inpaint_region` (line 681), `composite_roi_into_frame_seamless` (line 693). Upgrade existing DEBUG exception logs to WARNING with elapsed time. Add periodic heartbeat (30s or every N frames) inside Pass 2 composite loop (line 608+) reporting frame_idx / total + elapsed.
- [x] `code/src/stages/s5_revert/refiner.py` — wrap `torch.load` call at line 123 with INFO entry + elapsed exit log + `try/except`. This is a one-shot first-call blocker; wrapping surfaces slow checkpoint loads.

### Server layer
- [x] `server/app/pipeline_runner.py` — add `_LivenessWatchdog` helper class: daemon thread, resets on every `emit`, fires `src_logger.error(...)` on timeout. Wire into `run_pipeline_job`: instrument the emit closure to update timestamp, start the watchdog in the `try:` block, stop in `finally:`. Read `PIPELINE_LIVENESS_TIMEOUT_S` env var with 180s default. Document in module docstring.

### Client layer
- [x] `web/src/lib/stages.ts` — add `STALL_THRESHOLD_MS = 180_000` constant next to `STAGES`.
- [x] `web/src/hooks/useJobStream.ts` — add `stalledMs: number` to `JobStreamState` + `initialState()` + reset paths. Update `setInterval` tick to compute stall and setState. Reset on stage_start / stage_complete / done / error / status-sync-terminal / unmount / reset.
- [x] `web/src/components/StageProgress.tsx` — render stall badge on the active tile when `stalledMs > 0`. Use existing design tokens from `globals.css`; match `.warn-pill` styling if one exists.

### Tests
- [x] `server/tests/test_pipeline_runner.py` — watchdog tests: fires after silence, resets on emit, cleans up on exception / normal exit. Use a fake `PipelineRunner` that sleeps longer than the timeout.
- [x] `web/src/hooks/__tests__/useJobStream.test.ts` — `stalledMs` transitions: starts 0, increments past threshold, resets on stage change / terminal. Use `vi.useFakeTimers()`.
- [x] `web/src/components/__tests__/StageProgress.test.tsx` — renders stall badge when `stalledMs > 0`, hides when 0, positioned on the active tile.

## Risks

- **False positives on legitimate slow stages.** S3's AnyText2 call can take 8–15 min on multi-region clips; single-region waits within that can be >60s. Mitigation: rely on stage-layer heartbeats (new for S4/S5, existing for S3) to reset the watchdog clock. Flat 180s threshold + env var override. If S3 single-region waits trip it, add a poll-loop heartbeat to `anytext2_editor.py` as a follow-up (out of scope here).
- **Log-panel flood from heartbeats.** 30s heartbeat across 5 stages → ~10 extra INFO lines in a typical 3-minute run. Well under the 500-entry cap. Per-ROI detail stays at DEBUG so `<LogPanel>` doesn't see it.
- **Client `stalledMs` timer precision.** `Math.floor(elapsed / 1000) * 1000` already used for `activeStageElapsedMs`; reuse the same rounding so the threshold comparison is stable (no flicker at the boundary). Verify no off-by-one in test.
- **Dev-server churn during implementation.** Uvicorn `--reload` will trigger on every pipeline-code edit. Expect noisy reloads; not a correctness concern.

## Done When
- [ ] S4 and S5 emit per-track INFO entry/exit + periodic heartbeat logs during a full pipeline run (verified in `<LogPanel>`).
- [ ] Forcing a silent 200s sleep in S5 (test fixture) surfaces a red `"no progress for 180s"` log line in the browser within ~180s.
- [ ] `<StageProgress>` active tile shows a stall badge after 180s on a healthy-but-slow stage; badge clears when the stage completes.
- [ ] A healthy end-to-end pipeline run (on a short clip) produces zero stall warnings and zero watchdog fires.
- [ ] `PIPELINE_LIVENESS_TIMEOUT_S=60 ./server/scripts/dev.sh` tightens the threshold for manual stall testing.
- [ ] No regression in existing unit/integration tests (`cd code && python -m pytest tests/ -v`, `cd server && python -m pytest tests/ -v`, `cd web && npm run test`).
- [ ] Lint + type-check pass (`ruff check code/`, `cd web && npm run type-check`).
- [ ] Code review approved (`@reviewer`).
- [ ] Changes committed as atomic commits (stage layer, server layer, client layer, tests — one per commit if size permits).

## Progress
- [x] Step 1 — S4/S5 stage-layer wraps + heartbeats (code/src/stages/). 446 pytest pass, ruff clean on touched files.
- [x] Step 2 — S5 refiner `torch.load` wrap. Covered in Step 1 commit.
- [x] Step 3 — Server watchdog in `pipeline_runner.py` + tests. 97 pytest pass (87 baseline + 10 new), ruff clean.
- [x] Step 4 — Client `stalledMs` state + hook tests.
- [x] Step 5 — `<StageProgress>` stall badge + component tests. 158 web test pass (129 baseline + 29 new across hook + component), type-check clean.
- [ ] Step 6 — End-to-end manual test: force a stall with `PIPELINE_LIVENESS_TIMEOUT_S=30` + a sleep, verify UI surfaces it.
- [ ] Step 7 — Review + commit.

## Out of Scope (Follow-ups)

This plan is **pure observability** — user sees the stall, but cannot
recover from it without external action. Recovery-from-stall is a
separate concern, captured here so it isn't lost:

- **Why "no heartbeat for N seconds" ≠ "system is dead":** a stuck CUDA
  kernel, a hung AnyText2 socket, and a dead process all look identical
  from inside the Python worker. The watchdog thread only knows "emit()
  has not been called." That is probabilistic evidence, not proof. Any
  auto-restart built on this signal will eventually kill a legitimately
  slow run.

- **Option A — process-level restart.** `supervisorctl restart uvicorn`
  or kill+relaunch `dev.sh`. Guaranteed clean. Kills the whole server
  and interrupts other sessions. No code change. Viable short-term
  recovery path for the demo today.

- **Option B — relax `DELETE /api/jobs/{id}` to accept running jobs.**
  Flip `record.status = "failed"`, emit `ErrorEvent`, UI clears. Trap:
  the stuck worker thread still leaks until a real process restart
  (Python cannot force-terminate a thread blocked in a CUDA kernel).
  The UI would *appear* to recover while a zombie worker keeps holding
  the GPU. Do not pursue.

- **Option C — cancellation tokens through the pipeline** (previously
  "decision 2B"). Thread a `threading.Event` into every stage; stages
  check it at loop boundaries; watchdog or a user-triggered endpoint
  sets it on timeout; pipeline raises `PipelineCancelled` → existing
  `JobManager._run_job` error path fires → browser renders
  `<FailureCard>`. The architecturally clean answer. Scope: every stage
  gains a cancellation surface, `DELETE` on running job becomes the
  user-triggered path, watchdog can optionally trigger it after a hard
  ceiling (e.g. 20 min). Large enough to deserve its own plan.

- **Option D — external health probe + container restart policy.** Add
  a `/api/health` probe that verifies worker-thread responsiveness;
  Docker/systemd restarts the container on failure. Doesn't solve
  per-job recovery; restarts everything. Orthogonal to this work.

**Recommended follow-up sequencing:** Option A today (manual SSH
restart), Option C when there's appetite for the cross-stage change.
Skip B entirely.
