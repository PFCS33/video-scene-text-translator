/**
 * Tests for <JobView>. We mock the `useJobStream` hook and the `deleteJob`
 * client call to exercise the composition logic without a real SSE loop.
 *
 * Coverage:
 *   1. Running job — renders stage progress + log panel, no result.
 *   2. Succeeded job — renders ResultPanel with the output URL plus a
 *      "Submit another" button that calls onReset.
 */

import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import type { JobStreamState, UseJobStreamResult } from "@/hooks/useJobStream";

// Mock the hook so each test fully controls what JobView sees.
vi.mock("@/hooks/useJobStream", () => ({
  useJobStream: vi.fn(),
}));

// Mock deleteJob so the "delete" button press doesn't hit fetch.
vi.mock("@/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/api/client")>();
  return {
    ...actual,
    deleteJob: vi.fn().mockResolvedValue({ deleted: "job-1", ts: 1 }),
  };
});

import { JobView } from "../JobView";
import { useJobStream } from "@/hooks/useJobStream";
import { ApiError, deleteJob } from "@/api/client";

function makeState(
  overrides: Partial<JobStreamState> = {},
): JobStreamState {
  return {
    status: "running",
    stages: {
      s1: "active",
      s2: "pending",
      s3: "pending",
      s4: "pending",
      s5: "pending",
    },
    stageDurations: {},
    logs: [],
    error: null,
    outputUrl: null,
    currentStage: "s1",
    activeStageElapsedMs: 0,
    ...overrides,
  };
}

function mockHook(state: JobStreamState, reset = vi.fn()) {
  vi.mocked(useJobStream).mockReturnValue({
    state,
    reset,
  } satisfies UseJobStreamResult);
}

beforeEach(() => {
  vi.mocked(useJobStream).mockReset();
  vi.mocked(deleteJob).mockClear();
});

describe("<JobView>", () => {
  it("renders stage progress and log panel for a running job", () => {
    mockHook(
      makeState({
        status: "running",
        logs: [{ level: "info", message: "kickoff", ts: 1 }],
      }),
    );

    render(<JobView jobId="11111111-2222-3333-4444-555555555555" />);

    // Job id prefix appears in the header.
    expect(screen.getByText(/11111111/)).toBeInTheDocument();

    // Stage pills present.
    expect(
      screen.getByRole("list", { name: /pipeline progress/i }),
    ).toBeInTheDocument();
    expect(screen.getByText("Detect")).toBeInTheDocument();

    // Log panel shows the seeded log line.
    expect(screen.getByText("kickoff")).toBeInTheDocument();

    // No result panel while running.
    expect(screen.queryByRole("link", { name: /download/i })).toBeNull();
  });

  it("renders ResultPanel + Submit-another on success", async () => {
    const onReset = vi.fn();
    const resetSpy = vi.fn();
    mockHook(
      makeState({
        status: "succeeded",
        stages: {
          s1: "done",
          s2: "done",
          s3: "done",
          s4: "done",
          s5: "done",
        },
        stageDurations: { s1: 1000, s2: 2000, s3: 3000, s4: 4000, s5: 5000 },
        outputUrl: "/api/jobs/job-1/output",
        currentStage: null,
      }),
      resetSpy,
    );

    const user = userEvent.setup();
    render(<JobView jobId="job-1" onReset={onReset} />);

    // Download link points at the output URL.
    const download = screen.getByRole("link", { name: /download/i });
    expect(download).toHaveAttribute("href", "/api/jobs/job-1/output");

    // Submit another clears state and bubbles to parent.
    await user.click(screen.getByRole("button", { name: /submit another/i }));
    expect(resetSpy).toHaveBeenCalled();
    expect(onReset).toHaveBeenCalled();
  });

  it("renders ErrorAlert and Submit-another on failed status", async () => {
    const onReset = vi.fn();
    const resetSpy = vi.fn();
    mockHook(
      makeState({
        status: "failed",
        stages: {
          s1: "done",
          s2: "active",
          s3: "pending",
          s4: "pending",
          s5: "pending",
        },
        currentStage: null,
        error: {
          message: "boom",
          traceback: "Traceback (most recent call last)...",
        },
      }),
      resetSpy,
    );

    const user = userEvent.setup();
    render(<JobView jobId="job-1" onReset={onReset} />);

    // ErrorAlert renders the message inline.
    expect(screen.getByText("boom")).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /pipeline failed/i }),
    ).toBeInTheDocument();

    // Submit another still works on the failed branch.
    await user.click(screen.getByRole("button", { name: /submit another/i }));
    expect(resetSpy).toHaveBeenCalled();
    expect(onReset).toHaveBeenCalled();
  });

  it("calls deleteJob then onReset when Delete job succeeds", async () => {
    const onReset = vi.fn();
    const resetSpy = vi.fn();
    mockHook(
      makeState({
        status: "succeeded",
        stages: {
          s1: "done",
          s2: "done",
          s3: "done",
          s4: "done",
          s5: "done",
        },
        outputUrl: "/api/jobs/job-1/output",
        currentStage: null,
      }),
      resetSpy,
    );
    vi.mocked(deleteJob).mockResolvedValueOnce({ deleted: "job-1", ts: 1 });

    const user = userEvent.setup();
    render(<JobView jobId="job-1" onReset={onReset} />);

    await user.click(screen.getByRole("button", { name: /delete job/i }));

    expect(vi.mocked(deleteJob)).toHaveBeenCalledWith("job-1");
    // Await the microtask queue so the post-resolve onReset fires.
    await vi.waitFor(() => {
      expect(resetSpy).toHaveBeenCalled();
      expect(onReset).toHaveBeenCalled();
    });
  });

  it("shows an inline alert and does NOT call onReset when Delete job hits 409", async () => {
    const onReset = vi.fn();
    const resetSpy = vi.fn();
    mockHook(
      makeState({
        status: "succeeded",
        stages: {
          s1: "done",
          s2: "done",
          s3: "done",
          s4: "done",
          s5: "done",
        },
        outputUrl: "/api/jobs/job-1/output",
        currentStage: null,
      }),
      resetSpy,
    );
    vi.mocked(deleteJob).mockRejectedValueOnce(
      new ApiError(409, "job still running"),
    );

    const user = userEvent.setup();
    render(<JobView jobId="job-1" onReset={onReset} />);

    await user.click(screen.getByRole("button", { name: /delete job/i }));

    // Inline alert surfaces on the 409 branch.
    expect(
      await screen.findByRole("heading", { name: /couldn't delete job/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/still running/i)).toBeInTheDocument();

    // onReset must not have been called — the job is still live.
    expect(resetSpy).not.toHaveBeenCalled();
    expect(onReset).not.toHaveBeenCalled();
  });
});
