/**
 * <FailureCard> — right-column surface for the terminal "failed" phase.
 *
 * Replaces <ErrorAlert>. The old file is intentionally left in place until
 * Step 14 deletes <JobView> (which still imports it), so the test suite
 * stays green commit-by-commit (plan D4).
 *
 * Contract (plan.md, Step 12 + deferred-list):
 *   - Title is the raw `error.message`. No human-friendly mapping layer —
 *     mapping `OutOfMemoryError` → "Your GPU is too small" is out of scope.
 *   - Description line uses the first line of the traceback when present,
 *     or a generic fallback when it isn't. Both cases give the user a
 *     one-glance summary without forcing them to open the <details>.
 *   - Full traceback behind a collapsible <details> — native element,
 *     native keyboard + screen-reader support.
 *   - "Copy error" button writes `message + "\n\n" + traceback` (or just
 *     `message` when there's no traceback) to the clipboard, then flips
 *     its own label to "Copied" for 2s as a low-key confirmation.
 *   - No Retry button. "Submit another" on the left column covers it (the
 *     plan's deferred list is explicit about this).
 *
 * Accessibility:
 *   - <section aria-labelledby> wires the <h2> as the card's label.
 *   - The red label strip is decorative — `aria-hidden` because it
 *     repeats info already covered by the heading.
 *   - The Copy-error button exposes its state change via `aria-live` so
 *     SR users hear the "Copied" confirmation.
 */

import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";

interface FailureCardProps {
  message: string;
  traceback?: string | null;
}

const COPIED_RESET_MS = 2000;

function firstLineOf(s: string): string {
  // `split("\n")[0]` is safe on empty strings (returns ""), but we call
  // this only when `traceback` has length, so that edge case never fires
  // in practice. Still, defend against a lone leading "\n".
  const first = s.split("\n")[0] ?? "";
  return first.trim();
}

export function FailureCard({
  message,
  traceback,
}: FailureCardProps): JSX.Element {
  const headingId = "failure-card-heading";
  const hasTraceback = Boolean(traceback && traceback.trim().length > 0);

  // Local UI state only — `copied` drives the button label flip. No
  // parent-observable side effects.
  const [copied, setCopied] = useState(false);
  // Track the reset timer so we can clear it if the component unmounts
  // between click and the 2s revert (would otherwise setState-after-
  // unmount).
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current !== null) {
        clearTimeout(timerRef.current);
      }
    };
  }, []);

  const description = hasTraceback
    ? firstLineOf(traceback as string)
    : "The pipeline hit an error and could not complete.";

  const handleCopy = (): void => {
    const payload = hasTraceback ? `${message}\n\n${traceback}` : message;
    // `writeText` returns a Promise that can reject if the document isn't
    // focused or clipboard permission is denied. We swallow the rejection
    // — failing silently is better than surfacing a second error while
    // the user is already looking at a failure card.
    void navigator.clipboard?.writeText(payload);

    setCopied(true);
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
    }
    timerRef.current = setTimeout(() => {
      setCopied(false);
      timerRef.current = null;
    }, COPIED_RESET_MS);
  };

  return (
    <section
      aria-labelledby={headingId}
      className="mx-auto w-full max-w-[560px] overflow-hidden rounded-md border border-[color:var(--err-line)] bg-card"
    >
      {/* Decorative label strip — heading below duplicates the meaning. */}
      <div
        aria-hidden="true"
        className="border-b border-[color:var(--err-line)] bg-[color:var(--err-soft)] px-4 py-2 font-mono text-[11px] font-semibold uppercase tracking-wider text-[color:var(--err)]"
      >
        <span className="mr-1.5">&#x25CF;</span>
        PIPELINE FAILED
      </div>

      <div className="space-y-3 px-5 py-4">
        <h2
          id={headingId}
          className="text-lg font-semibold text-foreground"
        >
          {message}
        </h2>
        <p className="text-sm text-muted-foreground">{description}</p>

        {hasTraceback && (
          <details>
            <summary className="cursor-pointer font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
              Show traceback
            </summary>
            <pre className="mt-2 max-h-64 overflow-y-auto whitespace-pre-wrap rounded border border-border bg-[color:var(--bg-2)] p-3 font-mono text-[11px] leading-relaxed text-foreground">
              {traceback}
            </pre>
          </details>
        )}

        <div className="flex justify-end">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={handleCopy}
            aria-live="polite"
          >
            {copied ? "Copied" : "Copy error"}
          </Button>
        </div>
      </div>
    </section>
  );
}
