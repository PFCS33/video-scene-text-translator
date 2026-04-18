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

const STAGES: readonly Stage[] = ["s1", "s2", "s3", "s4", "s5"] as const;
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

  const reset = useCallback(() => {
    setState(initialState());
  }, []);

  // -------------------------------------------------------------------------
  // Event handler — closed over setState, so we can declare once and reuse
  // across seed + stream callbacks without recreating per render.
  // -------------------------------------------------------------------------
  const applyEvent = useCallback((ev: SSEEvent) => {
    setState((prev) => {
      switch (ev.type) {
        case "stage_start":
          return {
            ...prev,
            status: prev.status === "connecting" ? "running" : prev.status,
            stages: withStageMarked(prev.stages, ev.stage, "active"),
            currentStage: ev.stage,
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
          };
        case "error":
          return {
            ...prev,
            status: "failed",
            error: {
              message: ev.message,
              traceback: ev.traceback ?? null,
            },
            currentStage: null,
          };
        default:
          return prev;
      }
    });

    // Close the stream on terminal events so we don't keep a dead socket
    // around. Matches the server, which finishes the SSE response after
    // emitting done/error.
    if (ev.type === "done" || ev.type === "error") {
      streamRef.current?.close();
      streamRef.current = null;
    }
  }, []);

  const applyStatusSync = useCallback((status: JobStatus) => {
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
          currentStage: null,
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
    // If the resync landed on a terminal status, the server-side SSE
    // generator has already finished, so keep the stream around produces
    // a noisy reconnect loop (server closes, browser reconnects, repeat).
    // Close our end to break the cycle.
    if (status.status === "succeeded" || status.status === "failed") {
      streamRef.current?.close();
      streamRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!jobId) {
      return;
    }

    activeJobRef.current = jobId;
    // Fresh state on every jobId change. A caller can also call reset()
    // explicitly, but binding to the effect means a direct <JobView key>
    // swap also does the right thing.
    setState(initialState());

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
    };
  }, [jobId, applyEvent, applyStatusSync]);

  return { state, reset };
}
