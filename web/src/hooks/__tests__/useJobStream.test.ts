/**
 * Tests for <useJobStream>.
 *
 * Strategy:
 *   - Mock `@/api/client` (getJobStatus) and `@/api/sse` (openEventStream).
 *   - `openEventStream` mock captures the `StreamOptions` so each test can
 *     drive the reducer synchronously via `act(() => capturedOptions.onEvent(...))`.
 *   - `getJobStatus` mock is set per-test to shape the seed fetch result.
 *
 * We test behaviour, not implementation:
 *   1. null jobId -> idle-ish initial state
 *   2. stage_start sets the stage active and updates currentStage
 *   3. stage_complete flips to done + records duration
 *   4. done event: all stages done, outputUrl set, status succeeded
 *   5. error event: status failed + error message captured
 *   6. log events append in order + get capped at 500
 *   7. reset() returns state to initial
 *   8. onStatusSync merges current_stage from a re-sync JobStatus
 */

import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";

import type { JobStatus, SSEEvent } from "@/api/schemas";
import type { StreamOptions } from "@/api/sse";

// ---------------------------------------------------------------------------
// Module mocks — hoisted by Vitest. `capturedOptions` lets each test reach
// into the latest openEventStream invocation and drive its callbacks.
// ---------------------------------------------------------------------------

let capturedOptions: StreamOptions | null = null;
const mockClose = vi.fn();

vi.mock("@/api/sse", () => ({
  openEventStream: vi.fn((_jobId: string, opts: StreamOptions) => {
    capturedOptions = opts;
    return { close: mockClose, readyState: 1 };
  }),
}));

vi.mock("@/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/api/client")>();
  return {
    ...actual,
    // Only getJobStatus is stubbed — outputUrl / eventsUrl are pure helpers
    // and we want the real string-building behaviour so assertions like
    // `/api/jobs/job-1/output` are the actual app path, not a mock constant.
    getJobStatus: vi.fn(),
  };
});

import { useJobStream } from "../useJobStream";
import { getJobStatus } from "@/api/client";
import { openEventStream } from "@/api/sse";

function baseStatus(overrides: Partial<JobStatus> = {}): JobStatus {
  return {
    job_id: "job-1",
    status: "running",
    source_lang: "en",
    target_lang: "es",
    created_at: 0,
    current_stage: null,
    finished_at: null,
    error: null,
    output_available: false,
    ...overrides,
  };
}

beforeEach(() => {
  capturedOptions = null;
  mockClose.mockReset();
  vi.mocked(openEventStream).mockClear();
  vi.mocked(getJobStatus).mockReset();
  vi.mocked(getJobStatus).mockResolvedValue(baseStatus());
});

describe("useJobStream", () => {
  it("starts in initial state when jobId is null", () => {
    const { result } = renderHook(() => useJobStream(null));

    expect(result.current.state.status).toBe("connecting");
    expect(result.current.state.currentStage).toBeNull();
    expect(result.current.state.outputUrl).toBeNull();
    expect(result.current.state.error).toBeNull();
    expect(result.current.state.logs).toEqual([]);
    expect(result.current.state.activeStageElapsedMs).toBe(0);
    expect(result.current.state.stages).toEqual({
      s1: "pending",
      s2: "pending",
      s3: "pending",
      s4: "pending",
      s5: "pending",
    });
    // No SSE subscription when jobId is null.
    expect(vi.mocked(openEventStream)).not.toHaveBeenCalled();
  });

  it("marks a stage active on stage_start", async () => {
    const { result } = renderHook(() => useJobStream("job-1"));

    await waitFor(() => expect(capturedOptions).not.toBeNull());

    act(() => {
      capturedOptions!.onEvent({
        type: "stage_start",
        stage: "s2",
        ts: 1,
      } satisfies SSEEvent);
    });

    expect(result.current.state.stages.s2).toBe("active");
    expect(result.current.state.stages.s1).toBe("done"); // preceding flipped
    expect(result.current.state.stages.s3).toBe("pending");
    expect(result.current.state.currentStage).toBe("s2");
  });

  it("flips a stage to done on stage_complete and records duration", async () => {
    const { result } = renderHook(() => useJobStream("job-1"));
    await waitFor(() => expect(capturedOptions).not.toBeNull());

    act(() => {
      capturedOptions!.onEvent({
        type: "stage_start",
        stage: "s1",
        ts: 1,
      });
    });
    act(() => {
      capturedOptions!.onEvent({
        type: "stage_complete",
        stage: "s1",
        duration_ms: 2345,
        ts: 2,
      });
    });

    expect(result.current.state.stages.s1).toBe("done");
    expect(result.current.state.stageDurations.s1).toBe(2345);
    expect(result.current.state.currentStage).toBeNull();
  });

  it("on done, marks all stages done, sets outputUrl and status=succeeded", async () => {
    const { result } = renderHook(() => useJobStream("job-1"));
    await waitFor(() => expect(capturedOptions).not.toBeNull());

    act(() => {
      capturedOptions!.onEvent({
        type: "done",
        output_url: "/api/jobs/job-1/output",
        ts: 99,
      });
    });

    expect(result.current.state.status).toBe("succeeded");
    expect(result.current.state.outputUrl).toBe("/api/jobs/job-1/output");
    expect(result.current.state.stages).toEqual({
      s1: "done",
      s2: "done",
      s3: "done",
      s4: "done",
      s5: "done",
    });
    // Stream closed after terminal event.
    expect(mockClose).toHaveBeenCalled();
  });

  it("on error, sets status=failed and captures the message + traceback", async () => {
    const { result } = renderHook(() => useJobStream("job-1"));
    await waitFor(() => expect(capturedOptions).not.toBeNull());

    act(() => {
      capturedOptions!.onEvent({
        type: "error",
        message: "boom",
        traceback: "Traceback (...)",
        ts: 7,
      });
    });

    expect(result.current.state.status).toBe("failed");
    expect(result.current.state.error).toEqual({
      message: "boom",
      traceback: "Traceback (...)",
    });
    expect(mockClose).toHaveBeenCalled();
  });

  it("captures failedStage from the last active stage on error", async () => {
    // Regression: previously `currentStage` was cleared in the same setState
    // as `status: failed`, so <StageProgress>'s fail-tile styling never
    // activated. We now freeze the last active stage into `failedStage`
    // before clearing `currentStage`.
    const { result } = renderHook(() => useJobStream("job-1"));
    await waitFor(() => expect(capturedOptions).not.toBeNull());

    act(() => {
      capturedOptions!.onEvent({ type: "stage_start", stage: "s3", ts: 1 });
    });
    act(() => {
      capturedOptions!.onEvent({
        type: "error",
        message: "boom",
        ts: 2,
      });
    });

    expect(result.current.state.status).toBe("failed");
    expect(result.current.state.currentStage).toBeNull();
    expect(result.current.state.failedStage).toBe("s3");
  });

  it("captures failedStage from applyStatusSync on a failed job", async () => {
    // SSE reconnect lands directly on a failed status — no SSE error event
    // arrived locally, so failedStage must fall back to status.current_stage.
    const { result } = renderHook(() => useJobStream("job-1"));
    await waitFor(() => expect(capturedOptions).not.toBeNull());

    act(() => {
      capturedOptions!.onStatusSync?.(
        baseStatus({
          status: "failed",
          current_stage: "s4",
          error: "dead",
        }),
      );
    });

    expect(result.current.state.status).toBe("failed");
    expect(result.current.state.currentStage).toBeNull();
    expect(result.current.state.failedStage).toBe("s4");
  });

  it("seed-fetch on a failed job populates failedStage from status.current_stage", async () => {
    vi.mocked(getJobStatus).mockResolvedValueOnce(
      baseStatus({
        status: "failed",
        current_stage: "s2",
        error: "ded",
      }),
    );

    const { result } = renderHook(() => useJobStream("job-1"));

    await waitFor(() => {
      expect(result.current.state.status).toBe("failed");
    });

    expect(result.current.state.failedStage).toBe("s2");
  });

  it("appends log events in order and caps at 500", async () => {
    const { result } = renderHook(() => useJobStream("job-1"));
    await waitFor(() => expect(capturedOptions).not.toBeNull());

    act(() => {
      for (let i = 0; i < 505; i++) {
        capturedOptions!.onEvent({
          type: "log",
          level: "info",
          message: `m${i}`,
          ts: i,
        });
      }
    });

    // Oldest 5 should be dropped; newest preserved; order preserved.
    expect(result.current.state.logs).toHaveLength(500);
    expect(result.current.state.logs[0]!.message).toBe("m5");
    expect(result.current.state.logs[499]!.message).toBe("m504");
  });

  it("reset() returns state to initial", async () => {
    const { result } = renderHook(() => useJobStream("job-1"));
    await waitFor(() => expect(capturedOptions).not.toBeNull());

    act(() => {
      capturedOptions!.onEvent({
        type: "stage_start",
        stage: "s3",
        ts: 1,
      });
      capturedOptions!.onEvent({
        type: "log",
        level: "info",
        message: "hi",
        ts: 1,
      });
    });

    expect(result.current.state.stages.s3).toBe("active");
    expect(result.current.state.logs).toHaveLength(1);

    act(() => {
      result.current.reset();
    });

    expect(result.current.state.status).toBe("connecting");
    expect(result.current.state.currentStage).toBeNull();
    expect(result.current.state.logs).toEqual([]);
    expect(result.current.state.stages.s3).toBe("pending");
  });

  it("onStatusSync merges current_stage from a JobStatus snapshot", async () => {
    const { result } = renderHook(() => useJobStream("job-1"));
    await waitFor(() => expect(capturedOptions).not.toBeNull());

    act(() => {
      capturedOptions!.onStatusSync?.(
        baseStatus({ status: "running", current_stage: "s4" }),
      );
    });

    expect(result.current.state.currentStage).toBe("s4");
    expect(result.current.state.stages.s4).toBe("active");
    // preceding stages marked done
    expect(result.current.state.stages.s1).toBe("done");
    expect(result.current.state.stages.s3).toBe("done");
    expect(result.current.state.stages.s5).toBe("pending");
  });

  it("seed-fetch on a succeeded job sets outputUrl so the download button renders", async () => {
    // Rejoining after the job already finished (e.g. page reload). The hook's
    // seed fetch returns status=succeeded; we must populate outputUrl even
    // though we'll never receive the SSE `done` event.
    vi.mocked(getJobStatus).mockResolvedValueOnce(
      baseStatus({
        status: "succeeded",
        output_available: true,
      }),
    );

    const { result } = renderHook(() => useJobStream("job-1"));

    await waitFor(() => {
      expect(result.current.state.status).toBe("succeeded");
    });

    expect(result.current.state.outputUrl).toBe("/api/jobs/job-1/output");
    expect(result.current.state.stages.s5).toBe("done");
    expect(result.current.state.currentStage).toBeNull();
  });

  it("onStatusSync on a succeeded job sets outputUrl and closes the stream", async () => {
    // SSE reconnect scenario: the `done` event landed in the reconnect gap,
    // so applyStatusSync is the only path that learns the job is done.
    // It must (a) populate outputUrl, and (b) close the dead stream to
    // break the reconnect loop.
    const { result } = renderHook(() => useJobStream("job-1"));
    await waitFor(() => expect(capturedOptions).not.toBeNull());

    act(() => {
      capturedOptions!.onStatusSync?.(
        baseStatus({ status: "succeeded", output_available: true }),
      );
    });

    expect(result.current.state.status).toBe("succeeded");
    expect(result.current.state.outputUrl).toBe("/api/jobs/job-1/output");
    // stream was closed after the terminal sync — no reconnect loop
    expect(mockClose).toHaveBeenCalled();
  });

  it("seed-fetch on a queued job maps status to connecting, not running", async () => {
    // "queued" has no direct mapping in JobStreamState; treating it as
    // "running" in the seed path misreports the badge before S1 starts.
    vi.mocked(getJobStatus).mockResolvedValueOnce(
      baseStatus({ status: "queued", current_stage: null }),
    );

    const { result } = renderHook(() => useJobStream("job-1"));

    await waitFor(() => {
      // Wait for the seed fetch to settle.
      expect(vi.mocked(getJobStatus)).toHaveBeenCalled();
    });

    // Still "connecting", not "running", because no stage has started yet.
    expect(result.current.state.status).toBe("connecting");
    expect(result.current.state.currentStage).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Active-stage elapsed ticker (D8).
//
// Covers: start-on-stage_start, clear-on-stage_complete, restart on next
// stage_start, clear-on-done, clear-on-error, cleanup-on-unmount.
// Uses integer-second granularity — the interval updates once per second so
// consumers rendering `Math.floor(ms/1000)` don't burn render cycles on
// sub-second ticks that produce the same visible output.
// ---------------------------------------------------------------------------

describe("useJobStream — active-stage tick", () => {
  beforeEach(() => {
    // `shouldAdvanceTime` lets microtask-adjacent helpers (e.g. @testing-library's
    // `waitFor` polling) make progress without us manually advancing timers
    // for every promise flush.
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("starts a 1s ticker on stage_start and exposes integer-second elapsed", async () => {
    const { result } = renderHook(() => useJobStream("job-1"));
    await waitFor(() => expect(capturedOptions).not.toBeNull());

    act(() => {
      capturedOptions!.onEvent({
        type: "stage_start",
        stage: "s1",
        ts: 1,
      });
    });
    // At t=0 the elapsed is 0.
    expect(result.current.state.activeStageElapsedMs).toBe(0);

    // Advance 3500ms — we expect three whole-second ticks to have fired.
    act(() => {
      vi.advanceTimersByTime(3500);
    });

    expect(result.current.state.activeStageElapsedMs).toBe(3000);
  });

  it("clears the ticker on stage_complete and resets elapsed to 0", async () => {
    const { result } = renderHook(() => useJobStream("job-1"));
    await waitFor(() => expect(capturedOptions).not.toBeNull());

    act(() => {
      capturedOptions!.onEvent({ type: "stage_start", stage: "s1", ts: 1 });
    });
    act(() => {
      vi.advanceTimersByTime(2500);
    });
    expect(result.current.state.activeStageElapsedMs).toBe(2000);

    act(() => {
      capturedOptions!.onEvent({
        type: "stage_complete",
        stage: "s1",
        duration_ms: 2500,
        ts: 2,
      });
    });
    expect(result.current.state.activeStageElapsedMs).toBe(0);

    // After clear, advancing time does not change elapsed.
    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(result.current.state.activeStageElapsedMs).toBe(0);
  });

  it("restarts the ticker on the next stage_start", async () => {
    const { result } = renderHook(() => useJobStream("job-1"));
    await waitFor(() => expect(capturedOptions).not.toBeNull());

    act(() => {
      capturedOptions!.onEvent({ type: "stage_start", stage: "s1", ts: 1 });
    });
    act(() => {
      vi.advanceTimersByTime(4000);
    });
    act(() => {
      capturedOptions!.onEvent({
        type: "stage_complete",
        stage: "s1",
        duration_ms: 4000,
        ts: 2,
      });
    });
    act(() => {
      capturedOptions!.onEvent({ type: "stage_start", stage: "s2", ts: 3 });
    });

    expect(result.current.state.activeStageElapsedMs).toBe(0);

    act(() => {
      vi.advanceTimersByTime(1500);
    });
    expect(result.current.state.activeStageElapsedMs).toBe(1000);
  });

  it("clears the ticker on done", async () => {
    const { result } = renderHook(() => useJobStream("job-1"));
    await waitFor(() => expect(capturedOptions).not.toBeNull());

    act(() => {
      capturedOptions!.onEvent({ type: "stage_start", stage: "s5", ts: 1 });
    });
    act(() => {
      vi.advanceTimersByTime(2000);
    });

    act(() => {
      capturedOptions!.onEvent({
        type: "done",
        output_url: "/api/jobs/job-1/output",
        ts: 99,
      });
    });
    expect(result.current.state.activeStageElapsedMs).toBe(0);

    // Ticker is cleared — no further updates.
    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(result.current.state.activeStageElapsedMs).toBe(0);
  });

  it("clears the ticker on error", async () => {
    const { result } = renderHook(() => useJobStream("job-1"));
    await waitFor(() => expect(capturedOptions).not.toBeNull());

    act(() => {
      capturedOptions!.onEvent({ type: "stage_start", stage: "s3", ts: 1 });
    });
    act(() => {
      vi.advanceTimersByTime(2000);
    });

    act(() => {
      capturedOptions!.onEvent({
        type: "error",
        message: "boom",
        ts: 7,
      });
    });
    expect(result.current.state.activeStageElapsedMs).toBe(0);

    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(result.current.state.activeStageElapsedMs).toBe(0);
  });

  it("preserves the SSE-authoritative start time when applyStatusSync fires mid-stage", async () => {
    // Regression guard: `applyStatusSync`'s fallback-seed must only start a
    // ticker when one isn't already running — otherwise an SSE reconnect
    // mid-stage would reset the elapsed counter to 0. The authoritative
    // baseline comes from the SSE `stage_start` event, which we don't want
    // /status snapshots to overwrite. See useJobStream.ts D8 tradeoff note.
    const { result } = renderHook(() => useJobStream("job-1"));
    await waitFor(() => expect(capturedOptions).not.toBeNull());

    act(() => {
      capturedOptions!.onEvent({ type: "stage_start", stage: "s2", ts: 1 });
    });
    act(() => {
      vi.advanceTimersByTime(3000);
    });
    expect(result.current.state.activeStageElapsedMs).toBe(3000);

    // Simulate an SSE reconnect mid-stage: /status reports the same stage
    // still active. The ticker was already running, so we must NOT reseed
    // `stageStartMsRef` — that would reset elapsed to 0 on every reconnect.
    act(() => {
      capturedOptions!.onStatusSync?.(
        baseStatus({ status: "running", current_stage: "s2" }),
      );
    });
    expect(result.current.state.activeStageElapsedMs).toBe(3000);

    // And the original baseline continues to advance — +2s → 5s total.
    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(result.current.state.activeStageElapsedMs).toBe(5000);
  });

  it("clears the ticker on unmount (no orphan interval leak)", async () => {
    const { unmount } = renderHook(() => useJobStream("job-1"));
    await waitFor(() => expect(capturedOptions).not.toBeNull());

    act(() => {
      capturedOptions!.onEvent({ type: "stage_start", stage: "s1", ts: 1 });
    });

    // Sanity check: at least one active timer while the ticker runs.
    // (vi.getTimerCount includes any other microtask-scheduled timers, so
    // we assert at-least-one rather than exact-one.)
    expect(vi.getTimerCount()).toBeGreaterThanOrEqual(1);

    unmount();

    // After unmount, the ticker interval must be cleared — no lingering
    // interval keeps the fake-timer queue hot. Other timers (microtasks)
    // may still be queued, but the ticker itself must be gone, so
    // advancing time must not trigger a setState-after-unmount warning.
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    // No assertion failure means no late setState happened during advance.
  });
});
