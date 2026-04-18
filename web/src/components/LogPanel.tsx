/**
 * <LogPanel> — monospace scroll container for pipeline log lines.
 *
 * Auto-scroll with user-scroll escape hatch
 * -----------------------------------------
 * New logs auto-scroll to the bottom ONLY when the user is already at the
 * bottom (within a small threshold). If the user has scrolled up to read
 * earlier lines, new arrivals don't jank the view — they keep reading.
 * The moment they scroll back down to the bottom, auto-follow resumes.
 *
 * Implementation: track `isAtBottom` in a ref, updated on scroll events.
 * The auto-scroll effect only fires scrollTop=scrollHeight when that flag
 * is true at the moment of the new log arrival.
 *
 * Level styling
 * -------------
 * - info    -> default muted foreground
 * - warning -> amber (readable in both themes via Tailwind's dark variant)
 * - error   -> destructive token
 *
 * Timestamps are seconds-since-epoch floats from the server (LogEvent.ts),
 * which we convert to a local HH:MM:SS label via `Date`. The server emits
 * `ts` as a float `time.time()` — milliseconds = ts * 1000.
 */

import { useEffect, useRef } from "react";

import type { LogLevel } from "@/api/schemas";
import { cn } from "@/lib/utils";

export interface LogPanelProps {
  logs: Array<{ level: LogLevel; message: string; ts: number }>;
}

const LEVEL_CLASS: Record<LogLevel, string> = {
  info: "text-muted-foreground",
  warning: "text-yellow-600 dark:text-yellow-400",
  error: "text-destructive",
};

// Distance (px) from the bottom that still counts as "at the bottom" for
// auto-follow purposes. Absorbs sub-pixel rounding and browser edge cases.
const BOTTOM_THRESHOLD_PX = 8;

function formatTs(ts: number): string {
  const d = new Date(ts * 1000);
  // Pad helpers; avoids pulling in date-fns for one formatter.
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

function isScrolledToBottom(el: HTMLElement): boolean {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= BOTTOM_THRESHOLD_PX;
}

export function LogPanel({ logs }: LogPanelProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  // Default to true so the panel auto-follows on first mount (the initial
  // scrollHeight is 0, which trivially satisfies "at bottom").
  const isAtBottomRef = useRef(true);

  useEffect(() => {
    const el = panelRef.current;
    if (!el) return;
    // Only auto-scroll if the user was already at the bottom before this
    // log arrived. If they'd scrolled up, leave their view alone.
    if (isAtBottomRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [logs.length]);

  return (
    <div
      ref={panelRef}
      data-testid="log-panel"
      aria-label="Pipeline logs"
      onScroll={(e) => {
        isAtBottomRef.current = isScrolledToBottom(e.currentTarget);
      }}
      className="h-48 overflow-y-auto rounded-md border bg-muted/30 p-3 font-mono text-xs leading-relaxed"
    >
      {logs.length === 0 ? (
        <p className="text-muted-foreground italic">Waiting for logs…</p>
      ) : (
        logs.map((log, i) => (
          <div
            key={i}
            data-testid="log-line"
            className={cn("whitespace-pre-wrap", LEVEL_CLASS[log.level])}
          >
            <span className="opacity-60">[{formatTs(log.ts)}]</span>{" "}
            <span className="uppercase">{log.level}</span>{" "}
            <span>{log.message}</span>
          </div>
        ))
      )}
    </div>
  );
}
