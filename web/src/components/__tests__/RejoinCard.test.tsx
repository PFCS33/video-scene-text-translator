/**
 * Tests for <RejoinCard> — the right-column surface shown when the user's
 * submit is blocked by a concurrent running job (server returns 409 with
 * `active_job_id`). Behavior under test:
 *
 *   1. Renders heading, description, and Rejoin CTA (smoke).
 *   2. With no `blockingStatus`, renders a truncated job-id prefix and
 *      "—" placeholders for stage + started rows.
 *   3. With `blockingStatus`, renders the stage label (S3 · Edit) and a
 *      HH:MM:SS started-at time derived from `created_at`.
 *   4. Clicking the Rejoin button invokes `onRejoin` exactly once.
 *   5. Renders the static "your file stays queued" footer copy.
 *
 * Per plan D9: the metadata fetch is the parent's job — the card itself is
 * stateless. We don't assert pixel layout; we assert the presentational
 * contract the parent (App) is going to depend on.
 */

import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

import type { JobStatus } from "@/api/schemas";

import { RejoinCard } from "../right/RejoinCard";

function baseStatus(overrides: Partial<JobStatus> = {}): JobStatus {
  return {
    job_id: "abcdef1234567890",
    status: "running",
    source_lang: "en",
    target_lang: "es",
    created_at: 1700000000,
    current_stage: "s3",
    finished_at: null,
    error: null,
    output_available: false,
    ...overrides,
  };
}

describe("<RejoinCard>", () => {
  it("renders heading, description, and Rejoin CTA", () => {
    render(
      <RejoinCard
        blockingJobId="abcdef1234567890"
        blockingStatus={null}
        onRejoin={() => undefined}
      />,
    );

    expect(
      screen.getByRole("heading", { name: /server is busy/i }),
    ).toBeInTheDocument();
    // Description mentions the concurrency constraint.
    expect(
      screen.getByText(/another job is currently running/i),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /rejoin running job/i }),
    ).toBeInTheDocument();
  });

  it("renders the full job id and — placeholders when blockingStatus is null", () => {
    render(
      <RejoinCard
        blockingJobId="abcdef1234567890"
        blockingStatus={null}
        onRejoin={() => undefined}
      />,
    );

    // Full 16-char id renders (no truncation).
    expect(screen.getByText("abcdef1234567890")).toBeInTheDocument();

    // Stage + Started rows fall back to "—" when status is null. Two rows,
    // so two em dashes.
    const placeholders = screen.getAllByText("—");
    expect(placeholders.length).toBeGreaterThanOrEqual(2);
  });

  it("renders the stage label and started-at time when blockingStatus is provided", () => {
    render(
      <RejoinCard
        blockingJobId="abcdef1234567890"
        blockingStatus={baseStatus({
          current_stage: "s3",
          created_at: 1700000000,
        })}
        onRejoin={() => undefined}
      />,
    );

    // Stage cell renders "S3 · Edit" — assert both halves (don't pin the
    // exact separator glyph so we stay robust to a future " - " swap).
    expect(screen.getByText(/S3/)).toBeInTheDocument();
    expect(screen.getByText(/Edit/)).toBeInTheDocument();

    // Started cell renders an HH:MM:SS stamp. Locale-dependent format (12h
    // vs 24h), so pattern-match the digits-colons-digits shape.
    expect(screen.getByText(/\d{1,2}:\d{2}:\d{2}/)).toBeInTheDocument();
  });

  it("invokes onRejoin exactly once when the primary button is clicked", () => {
    const onRejoin = vi.fn();

    render(
      <RejoinCard
        blockingJobId="abcdef1234567890"
        blockingStatus={null}
        onRejoin={onRejoin}
      />,
    );

    fireEvent.click(
      screen.getByRole("button", { name: /rejoin running job/i }),
    );

    expect(onRejoin).toHaveBeenCalledTimes(1);
  });

  it("renders the 'your file stays queued' footer copy", () => {
    render(
      <RejoinCard
        blockingJobId="abcdef1234567890"
        blockingStatus={null}
        onRejoin={() => undefined}
      />,
    );

    expect(
      screen.getByText(/your file stays queued/i),
    ).toBeInTheDocument();
  });
});
