/**
 * Tests for <FailureCard> — the right-column surface shown when the pipeline
 * terminates with `job.error`. Replaces <ErrorAlert> (whose file is kept
 * around until Step 14 so old <JobView> still compiles).
 *
 * Behavior under test:
 *   1. Renders the raw `message` as a heading (no human-friendly mapping —
 *      per plan's deferred list).
 *   2. Collapsible <details> holds the full traceback, verbatim.
 *   3. Description line falls back to generic copy when traceback is absent.
 *   4. Description line uses the first line of the traceback when present.
 *   5. "Copy error" button writes `message + traceback` to the clipboard.
 *   6. With no traceback, clipboard receives just `message`.
 *   7. Button label flips to "Copied" for ~2s after click, then returns.
 *
 * jsdom has no `navigator.clipboard` by default — stub a `writeText` spy on
 * the global navigator in `beforeEach` so each test starts from a clean slate.
 */

import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";

import { FailureCard } from "../right/FailureCard";

describe("<FailureCard>", () => {
  let writeText: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, {
      clipboard: { writeText },
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders the message as a heading", () => {
    render(<FailureCard message="Pipeline crashed at stage 3" />);

    expect(
      screen.getByRole("heading", { name: /pipeline crashed at stage 3/i }),
    ).toBeInTheDocument();
  });

  it("renders the traceback inside a collapsible <details>", () => {
    const traceback =
      'Traceback (most recent call last):\n  File "edit.py", line 88\ntorch.cuda.OutOfMemoryError';

    render(
      <FailureCard
        message="CUDA out of memory"
        traceback={traceback}
      />,
    );

    // Open the collapsible; assertion is that the <pre> body contains the
    // raw traceback verbatim once opened.
    fireEvent.click(screen.getByText(/show traceback/i));

    // The description renders the first line, so the <pre> is the second
    // DOM node with the traceback prefix. We pin the deeper content
    // (File "edit.py") which only appears in the <pre>.
    expect(screen.getByText(/File "edit\.py"/)).toBeInTheDocument();
    expect(
      screen.getByText(/torch\.cuda\.OutOfMemoryError/),
    ).toBeInTheDocument();
  });

  it("falls back to a generic description when traceback is absent", () => {
    render(<FailureCard message="Something broke" traceback={null} />);

    expect(
      screen.getByText(/the pipeline hit an error and could not complete/i),
    ).toBeInTheDocument();
  });

  it("uses the first line of the traceback as the description", () => {
    render(
      <FailureCard
        message="bad input"
        traceback={"ValueError: bad input\n  File ...\n  more lines"}
      />,
    );

    expect(screen.getByText("ValueError: bad input")).toBeInTheDocument();
  });

  it("copies message + traceback to the clipboard when Copy error is clicked", () => {
    const message = "CUDA out of memory";
    const traceback = "Traceback: nope\n  details";

    render(<FailureCard message={message} traceback={traceback} />);

    fireEvent.click(screen.getByRole("button", { name: /copy error/i }));

    expect(writeText).toHaveBeenCalledTimes(1);
    expect(writeText).toHaveBeenCalledWith(`${message}\n\n${traceback}`);
  });

  it("copies just the message when traceback is absent", () => {
    render(<FailureCard message="boom" traceback={null} />);

    fireEvent.click(screen.getByRole("button", { name: /copy error/i }));

    expect(writeText).toHaveBeenCalledWith("boom");
  });

  it("flips the button label to 'Copied' after click, then back after 2s", () => {
    // Fake timers so we can drive the 2s revert deterministically.
    // `shouldAdvanceTime: true` keeps RTL's own microtasks progressing.
    vi.useFakeTimers({ shouldAdvanceTime: true });

    render(<FailureCard message="boom" traceback={null} />);

    const button = screen.getByRole("button", { name: /copy error/i });
    fireEvent.click(button);

    // Immediately after click, label reads "Copied".
    expect(
      screen.getByRole("button", { name: /copied/i }),
    ).toBeInTheDocument();

    // Fast-forward the 2s timer; label returns to "Copy error".
    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(
      screen.getByRole("button", { name: /copy error/i }),
    ).toBeInTheDocument();
  });
});
