/**
 * Tests for <LogPanel>. We assert:
 *   1. messages render in the given order, one line per entry.
 *   2. on a log append, the scroll container's `scrollTop` follows to
 *      `scrollHeight` (auto-scroll contract).
 *
 * For #2 jsdom doesn't perform real layout, so scrollHeight stays 0. We
 * sidestep that by mocking scrollHeight on the element under test and
 * observing that the component *writes* scrollTop to it.
 */

import { describe, expect, it } from "vitest";
import { fireEvent, render } from "@testing-library/react";

import { LogPanel } from "../LogPanel";

describe("<LogPanel>", () => {
  it("renders all messages in order", () => {
    const logs = [
      { level: "info" as const, message: "alpha", ts: 1 },
      { level: "warning" as const, message: "beta", ts: 2 },
      { level: "error" as const, message: "gamma", ts: 3 },
    ];

    const { container } = render(<LogPanel logs={logs} />);

    const lines = container.querySelectorAll("[data-testid='log-line']");
    expect(lines).toHaveLength(3);
    expect(lines[0]!.textContent).toContain("alpha");
    expect(lines[1]!.textContent).toContain("beta");
    expect(lines[2]!.textContent).toContain("gamma");
    // Ordering preserved.
    const indexOfAlpha = container.innerHTML.indexOf("alpha");
    const indexOfGamma = container.innerHTML.indexOf("gamma");
    expect(indexOfAlpha).toBeLessThan(indexOfGamma);
  });

  it("auto-scrolls to the bottom when logs grow", () => {
    const first = [{ level: "info" as const, message: "first", ts: 1 }];
    const { container, rerender } = render(<LogPanel logs={first} />);

    const panel = container.querySelector(
      "[data-testid='log-panel']",
    ) as HTMLElement;
    expect(panel).not.toBeNull();

    // Fake a filled-in scroll area since jsdom doesn't lay out. The component
    // should read scrollHeight on effect and assign it to scrollTop.
    Object.defineProperty(panel, "scrollHeight", {
      configurable: true,
      value: 1234,
    });

    rerender(
      <LogPanel
        logs={[
          ...first,
          { level: "info" as const, message: "second", ts: 2 },
        ]}
      />,
    );

    expect(panel.scrollTop).toBe(1234);
  });

  it("does NOT auto-scroll when user has scrolled up", () => {
    // Start with a baseline log + simulate "filled panel" geometry.
    const first = [{ level: "info" as const, message: "first", ts: 1 }];
    const { container, rerender } = render(<LogPanel logs={first} />);
    const panel = container.querySelector(
      "[data-testid='log-panel']",
    ) as HTMLElement;

    // Simulate: scrollHeight grew to 1000, clientHeight is 192 (the h-48
    // class maps to 12rem = 192px at default font-size). User scrolled
    // up — scrollTop is now well below the bottom.
    Object.defineProperty(panel, "scrollHeight", {
      configurable: true,
      value: 1000,
    });
    Object.defineProperty(panel, "clientHeight", {
      configurable: true,
      value: 192,
    });
    panel.scrollTop = 100; // far from bottom (1000 - 100 - 192 = 708 px off)

    // Fire a scroll event so the component updates its isAtBottom ref.
    fireEvent.scroll(panel);

    // Now a new log arrives. scrollTop must NOT jump to scrollHeight.
    rerender(
      <LogPanel
        logs={[
          ...first,
          { level: "info" as const, message: "second", ts: 2 },
        ]}
      />,
    );

    expect(panel.scrollTop).toBe(100);
  });

  it("resumes auto-scroll once the user returns to the bottom", () => {
    const first = [{ level: "info" as const, message: "first", ts: 1 }];
    const { container, rerender } = render(<LogPanel logs={first} />);
    const panel = container.querySelector(
      "[data-testid='log-panel']",
    ) as HTMLElement;

    Object.defineProperty(panel, "scrollHeight", {
      configurable: true,
      value: 1000,
    });
    Object.defineProperty(panel, "clientHeight", {
      configurable: true,
      value: 192,
    });

    // Step 1: user scrolls up.
    panel.scrollTop = 100;
    fireEvent.scroll(panel);

    rerender(
      <LogPanel
        logs={[
          ...first,
          { level: "info" as const, message: "second", ts: 2 },
        ]}
      />,
    );
    expect(panel.scrollTop).toBe(100); // sanity — no jank

    // Step 2: user scrolls back to the bottom.
    // With scrollHeight=1000 and clientHeight=192, bottom = 1000-192 = 808.
    panel.scrollTop = 808;
    fireEvent.scroll(panel);

    // Step 3: a new log arrives. scrollTop should auto-follow.
    Object.defineProperty(panel, "scrollHeight", {
      configurable: true,
      value: 1050,
    });
    rerender(
      <LogPanel
        logs={[
          ...first,
          { level: "info" as const, message: "second", ts: 2 },
          { level: "info" as const, message: "third", ts: 3 },
        ]}
      />,
    );

    expect(panel.scrollTop).toBe(1050);
  });
});
