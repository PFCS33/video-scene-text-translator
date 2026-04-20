/**
 * useJobStream — owns the lifecycle state of a single job.
 *
 * Responsibilities
 *   - Seed initial state from `GET /status` (so we don't wait N seconds for
 *     the first stage_start to reveal which stage is running after a
 *     page-reload rejoin).
 *   - Subscribe to the SSE stream via `openEventStream` and fold each event
 *     into a local reducer.
 *   - Re-sync current stage on SSE reconnect via `onStatusSync` (D16).
 *   - Close the stream cleanly when `jobId` changes or the component
 *     unmounts, and when a terminal event arrives.
 *
 * State shape lives in `JobStreamState` and is exported so component tests
 * can mock this hook with a hand-rolled value. Consumers read `state` and
 * call `reset()` to return to the initial idle shape (used by the "submit
 * another" flow in `<App>`).
 *
 * Reducer contract:
 *   - stage_start(s) -> stage s active, preceding stages flipped to done,
 *     currentStage = s.
 *   - stage_complete(s) -> stage s flipped to done, duration recorded,
 *     currentStage cleared when it equals s (a terminal stage_complete may
 *     arrive without a matching stage_start if the stream reconnects in
 *     between, so we defensively also mark preceding stages done).
 *   - log -> append; capped at LOG_CAP entries.
 *   - done -> all five stages done, outputUrl set, status=succeeded, stream
 *     closed.
 *   - error -> status=failed, error captured, stream closed.
 *
 * Why not a formal useReducer? The state is small (~6 fields) and the
 * transitions are event-driven rather than action-driven; a functional
 * setState with a switch statement reads just as clearly and avoids a
 * separate action-type enum. If we grow a cancel/replay UX the tradeoff
 * flips — reach for useReducer then.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { getJobStatus, outputUrl } from "@/api/client";
import { openEventStream, type EventStream } from "@/api/sse";
import type {
  JobStatus,
  LogLevel,
  SSEEvent,
  Stage,
} from "@/api/schemas";
import { STAGES, STALL_THRESHOLD_MS } from "@/lib/stages";

const LOG_CAP = 500;

export type StageState = "pending" | "active" | "done";

export interface LogEntry {
  level: LogLevel;
  message: string;
  ts: number;
}

export interface JobStreamState {
  status: "connecting" | "running" | "succeeded" | "failed";
  stages: Record<Stage, StageState>;
  stageDurations: Partial<Record<Stage, number>>;
  logs: LogEntry[];
  error: { message: string; traceback?: string | null } | null;
  outputUrl: string | null;
  currentStage: Stage | null;
  /**
   * Stage that was active when the pipeline transitioned into `failed`.
   *
   * Captured in the same reducer tick as `status: "failed"` (from both the
   * SSE `error` event path and the `applyStatusSync` failed branch), because
   * `currentStage` is cleared in that same commit — by the time
   * <StageProgress> renders, `currentStage` is already null, so it cannot
   * carry the fail-tile signal. On seed-fetch against an already-failed job
   * the value comes from `status.current_stage` directly.
   *
   * `null` whenever the job is not in a terminal failed state.
   */
  failedStage: Stage | null;
  /**
   * Elapsed time spent in the currently-active stage, in milliseconds,
   * floored to whole seconds. 0 when no stage is active.
   *
   * Driven by a `setInterval(1000)` bound to the latest `stage_start.ts`
   * (or, on a page-reload rejoin, to the time the SPA learned a stage was
   * in-flight — the server doesn't hand us the original start timestamp in
   * `/status`, so rejoin elapsed is "since this page loaded", not "since
   * the stage started"). The interval is cleared on stage_complete, done,
   * error, and unmount. See plan.md D8.
   */
  activeStageElapsedMs: number;
  /**
   * How long the active stage has been stalled past
   * `STALL_THRESHOLD_MS`, in milliseconds. 0 while `activeStageElapsedMs`
   * is below the threshold, and while no stage is active.
   *
   * Derived in the same tick as `activeStageElapsedMs` (one render per
   * second), then reset alongside it on stage_start / stage_complete /
   * done / error / status-sync-terminal / reset / unmount. Kept as a
   * state field (not re-derived in components) so every surface that
   * wants to show a stall signal reads a single source of truth and
   * never redefines the threshold. See plan.md Layer 3.
   */
  stalledMs: number;
}

export interface UseJobStreamResult {
  state: JobStreamState;
  reset(): void;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function initialStages(): Record<Stage, StageState> {
  return {
    s1: "pending",
    s2: "pending",
    s3: "pending",
    s4: "pending",
    s5: "pending",
  };
}

function initialState(): JobStreamState {
  return {
    status: "connecting",
    stages: initialStages(),
    stageDurations: {},
    logs: [],
    error: null,
    outputUrl: null,
    currentStage: null,
    failedStage: null,
    activeStageElapsedMs: 0,
    stalledMs: 0,
  };
}

function allDoneStages(): Record<Stage, StageState> {
  return {
    s1: "done",
    s2: "done",
    s3: "done",
    s4: "done",
    s5: "done",
  };
}

/**
 * Given a stage, mark it `mark` and flip all preceding stages to `done`.
 * Later stages are left untouched (they might already be done if events
 * arrived out of order).
 */
function withStageMarked(
  stages: Record<Stage, StageState>,
  stage: Stage,
  mark: StageState,
): Record<Stage, StageState> {
  const next = { ...stages };
  const idx = STAGES.indexOf(stage);
  for (let i = 0; i < STAGES.length; i++) {
    const s = STAGES[i]!;
    if (i < idx && next[s] === "pending") {
      next[s] = "done";
    } else if (i === idx) {
      next[s] = mark;
    }
  }
  return next;
}

/** Seed the stages record from a /status snapshot. */
function stagesFromStatus(status: JobStatus): {
  stages: Record<Stage, StageState>;
  currentStage: Stage | null;
} {
  if (status.status === "succeeded") {
    return { stages: allDoneStages(), currentStage: null };
  }
  if (status.current_stage) {
    return {
      stages: withStageMarked(initialStages(), status.current_stage, "active"),
      currentStage: status.current_stage,
    };
  }
  return { stages: initialStages(), currentStage: null };
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useJobStream(jobId: string | null): UseJobStreamResult {
  const [state, setState] = useState<JobStreamState>(initialState);
  const streamRef = useRef<EventStream | null>(null);
  // Stops late setState calls after unmount or jobId change.
  const activeJobRef = useRef<string | null>(null);
  // ------------------------------------------------------------------
  // Active-stage ticker refs (D8).
  //
  //   `intervalRef` holds the handle returned by `window.setInterval`
  //   so we can clear it on stage change, terminal events, and unmount.
  //   `stageStartMsRef` holds the wall-clock ms at which the current
  //   stage began. Kept in a ref (not state) because the whole point of
  //   the ticker is to own the cheap per-second state update — we don't
  //   want an extra render every time a stage starts just to record the
  //   baseline.
  // ------------------------------------------------------------------
  const intervalRef = useRef<number | null>(null);
  const stageStartMsRef = useRef<number | null>(null);

  const clearTicker = useCallback(() => {
    if (intervalRef.current !== null) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    stageStartMsRef.current = null;
  }, []);

  const startTicker = useCallback(
    (startMs: number) => {
      // Always clear before starting so overlapping stage_start events
      // (or a status-sync racing a stage_start) never leak an interval.
      clearTicker();
      stageStartMsRef.current = startMs;
      // Kick off at 0 immediately — `startTicker` is called from event
      // handlers that also set `activeStageElapsedMs` in the same
      // setState, so we don't need a separate setState here. The
      // interval then takes over on the next 1s boundary.
      intervalRef.current = window.setInterval(() => {
        const baseline = stageStartMsRef.current;
        if (baseline === null) return;
        // Integer-second resolution: consumers render `Math.floor(ms/1000)`
        // anyway, so updating state at sub-second granularity would burn
        // renders without changing the output.
        const elapsed = Math.floor((Date.now() - baseline) / 1000) * 1000;
        // Stall readout is derived from the same baseline in the same
        // tick — one render per second covers both fields and keeps the
        // threshold comparison stable (no boundary flicker, since the
        // flooring matches `activeStageElapsedMs`).
        const stalled =
          elapsed > STALL_THRESHOLD_MS ? elapsed - STALL_THRESHOLD_MS : 0;
        setState((prev) =>
          prev.activeStageElapsedMs === elapsed && prev.stalledMs === stalled
            ? prev
            : { ...prev, activeStageElapsedMs: elapsed, stalledMs: stalled },
        );
      }, 1000);
    },
    [clearTicker],
  );

  const reset = useCallback(() => {
    clearTicker();
    setState(initialState());
  }, [clearTicker]);

  // -------------------------------------------------------------------------
  // Event handler — closed over setState, so we can declare once and reuse
  // across seed + stream callbacks without recreating per render.
  // -------------------------------------------------------------------------
  const applyEvent = useCallback(
    (ev: SSEEvent) => {
      setState((prev) => {
        switch (ev.type) {
          case "stage_start":
            return {
              ...prev,
              status: prev.status === "connecting" ? "running" : prev.status,
              stages: withStageMarked(prev.stages, ev.stage, "active"),
              currentStage: ev.stage,
              // Reset the elapsed readout in the same commit as the stage
              // flip so the UI never shows the previous stage's time on
              // the new stage's tile. `stalledMs` rides the same reset
              // so a stall carried over from the previous stage doesn't
              // flash on the new tile for a tick.
              activeStageElapsedMs: 0,
              stalledMs: 0,
            };
          case "stage_complete":
            return {
              ...prev,
              status: prev.status === "connecting" ? "running" : prev.status,
              stages: withStageMarked(prev.stages, ev.stage, "done"),
              stageDurations: {
                ...prev.stageDurations,
                [ev.stage]: ev.duration_ms,
              },
              currentStage:
                prev.currentStage === ev.stage ? null : prev.currentStage,
              activeStageElapsedMs:
                prev.currentStage === ev.stage ? 0 : prev.activeStageElapsedMs,
              stalledMs:
                prev.currentStage === ev.stage ? 0 : prev.stalledMs,
            };
          case "log": {
            const entry: LogEntry = {
              level: ev.level,
              message: ev.message,
              ts: ev.ts,
            };
            const nextLogs =
              prev.logs.length >= LOG_CAP
                ? [...prev.logs.slice(prev.logs.length - LOG_CAP + 1), entry]
                : [...prev.logs, entry];
            return { ...prev, logs: nextLogs };
          }
          case "done":
            return {
              ...prev,
              status: "succeeded",
              stages: allDoneStages(),
              outputUrl: ev.output_url,
              currentStage: null,
              activeStageElapsedMs: 0,
              stalledMs: 0,
            };
          case "error":
            return {
              ...prev,
              status: "failed",
              error: {
                message: ev.message,
                traceback: ev.traceback ?? null,
              },
              // Capture the stage that was running BEFORE we clear
              // currentStage — otherwise <StageProgress> sees null and
              // can't paint the fail tile. Fall back to an existing
              // failedStage if one is already recorded (defensive against
              // out-of-order events).
              failedStage: prev.currentStage ?? prev.failedStage,
              currentStage: null,
              activeStageElapsedMs: 0,
              stalledMs: 0,
            };
          default:
            return prev;
        }
      });

      // Ticker side-effects — run *after* setState so the render order
      // is: state-update first, then interval bookkeeping. We bind to
      // `Date.now()` rather than `ev.ts * 1000` to avoid server/client
      // clock-skew ever producing a negative elapsed value. The tiny
      // network + parse delay between `stage_start` being emitted and
      // received is acceptable (sub-second in practice).
      switch (ev.type) {
        case "stage_start":
          startTicker(Date.now());
          break;
        case "stage_complete":
        case "done":
        case "error":
          clearTicker();
          break;
        default:
          break;
      }

      // Close the stream on terminal events so we don't keep a dead socket
      // around. Matches the server, which finishes the SSE response after
      // emitting done/error.
      if (ev.type === "done" || ev.type === "error") {
        streamRef.current?.close();
        streamRef.current = null;
      }
    },
    [startTicker, clearTicker],
  );

  const applyStatusSync = useCallback(
    (status: JobStatus) => {
      setState((prev) => {
        if (status.status === "succeeded") {
          // If the `done` event arrived during the SSE reconnect gap, this is
          // the only place the client learns the job is finished — so we
          // must also populate `outputUrl` here, not just on the SSE `done`
          // event path. Without this the download button never appears.
          return {
            ...prev,
            status: "succeeded",
            stages: allDoneStages(),
            outputUrl: prev.outputUrl ?? outputUrl(status.job_id),
            currentStage: null,
            activeStageElapsedMs: 0,
            stalledMs: 0,
          };
        }
        if (status.status === "failed") {
          return {
            ...prev,
            status: "failed",
            error: prev.error ?? {
              message: status.error ?? "Pipeline failed",
              traceback: null,
            },
            // Prefer the locally-observed active stage; fall back to any
            // already-captured failedStage, then to the server snapshot's
            // current_stage (which is what /status returns for a failed
            // job on reconnect when we never saw the SSE error event).
            failedStage:
              prev.currentStage ??
              prev.failedStage ??
              status.current_stage ??
              null,
            currentStage: null,
            activeStageElapsedMs: 0,
            stalledMs: 0,
          };
        }
        if (status.current_stage) {
          return {
            ...prev,
            status: "running",
            stages: withStageMarked(prev.stages, status.current_stage, "active"),
            currentStage: status.current_stage,
          };
        }
        return prev;
      });

      // Ticker side-effects for status syncs.
      //
      // Terminal: clear — the stream is about to close anyway, and a
      // lingering interval would keep firing setState with stale elapsed.
      //
      // Active stage but no ticker yet: fallback-seed with `Date.now()`.
      // We don't have the original `stage_start.ts` (see D8 option A —
      // `/status` doesn't carry it), so a reload-rejoin shows "elapsed
      // since this page loaded" rather than "elapsed since the stage
      // started". Acceptable UX tradeoff for an edge case.
      //
      // Active stage with ticker already running: leave it alone — the
      // authoritative start time is the one captured from the SSE
      // `stage_start` event, and overwriting it here would reset the
      // displayed elapsed every time the SSE stream reconnects.
      if (status.status === "succeeded" || status.status === "failed") {
        clearTicker();
        // Server-side SSE generator has already finished, so keeping the
        // stream around produces a noisy reconnect loop (server closes,
        // browser reconnects, repeat). Close our end to break the cycle.
        streamRef.current?.close();
        streamRef.current = null;
      } else if (status.current_stage && intervalRef.current === null) {
        startTicker(Date.now());
      }
    },
    [startTicker, clearTicker],
  );

  useEffect(() => {
    if (!jobId) {
      return;
    }

    activeJobRef.current = jobId;
    // Fresh state on every jobId change. A caller can also call reset()
    // explicitly, but binding to the effect means a direct `key` swap on
    // the consumer also does the right thing.
    setState(initialState());
    // Any ticker from a previous jobId dies with the job. The new job
    // will start its own on the first stage_start or status-sync.
    clearTicker();

    // Seed from /status so a page-reload rejoin shows the current stage
    // immediately instead of waiting for the next stage_start.
    let cancelled = false;
    getJobStatus(jobId)
      .then((status) => {
        if (cancelled || activeJobRef.current !== jobId) return;
        setState((prev) => {
          const { stages, currentStage } = stagesFromStatus(status);
          if (status.status === "succeeded") {
            return {
              ...prev,
              status: "succeeded",
              stages: allDoneStages(),
              outputUrl: outputUrl(jobId),
              currentStage: null,
            };
          }
          if (status.status === "failed") {
            return {
              ...prev,
              status: "failed",
              error: prev.error ?? {
                message: status.error ?? "Pipeline failed",
                traceback: null,
              },
              stages,
              currentStage: null,
              // Seed path: we never observed an active stage locally, so
              // the only source of truth is the server snapshot.
              failedStage: status.current_stage ?? null,
            };
          }
          // "queued" on the server has no direct mapping in our local
          // JobStreamState — treat it as "connecting" so the UI doesn't
          // falsely claim the pipeline is running before S1 starts.
          return {
            ...prev,
            status: status.status === "queued" ? "connecting" : "running",
            stages,
            currentStage,
          };
        });
      })
      .catch(() => {
        // Non-fatal: if /status 404s or races with job creation the SSE
        // stream will still give us events once it connects. Don't flip
        // into "failed" on a seed miss.
      });

    const stream = openEventStream(jobId, {
      onEvent: applyEvent,
      onStatusSync: applyStatusSync,
      onError: () => {
        // Transport error surface intentionally left thin for MVP — the
        // browser will auto-reconnect and `onStatusSync` re-syncs.
      },
    });
    streamRef.current = stream;

    return () => {
      cancelled = true;
      activeJobRef.current = null;
      stream.close();
      streamRef.current = null;
      // Clear the ticker on unmount / jobId change so we never leave an
      // orphan `setInterval` calling setState on an unmounted hook.
      clearTicker();
    };
  }, [jobId, applyEvent, applyStatusSync, clearTicker]);

  return { state, reset };
}
