/**
 * Tests for <IdlePlaceholder> — the centered empty-state surface shown in
 * the right column when no job is in flight. Truly static; one assertion
 * per visible element block is enough.
 */

import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { IdlePlaceholder } from "../right/IdlePlaceholder";

describe("<IdlePlaceholder>", () => {
  it("renders the eyebrow label and body copy", () => {
    render(<IdlePlaceholder />);

    expect(screen.getByText(/WAITING FOR A JOB/i)).toBeInTheDocument();
    // Body copy references the left-column action and what will appear here.
    expect(screen.getByText(/pick a file/i)).toBeInTheDocument();
    expect(
      screen.getByText(/appear in this window/i),
    ).toBeInTheDocument();
  });
});
