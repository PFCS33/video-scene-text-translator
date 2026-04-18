/**
 * <ResultPanel> — right-column surface shown on `done`. Translates to the
 * mockup's "RESULT · SAME WINDOW" card: a full-width output video with an
 * OUTPUT corner tag, plus a download action row below.
 *
 * Layout:
 *   - Outer card frame (rounded, bordered, own surface color).
 *   - Video frame (relative wrapper) holds the <video controls> + an absolutely
 *     positioned "OUTPUT" corner tag. The tag is `aria-hidden` because it's
 *     cosmetic — the video's `aria-label` already carries the job id.
 *   - Action row: left-aligned caption (mono job-id prefix) + right-aligned
 *     download anchor.
 *
 * Primary-button color decision (A vs B from Step 11):
 *   (A) use shadcn's `variant="default"` — accent blue
 *   (B) introduce a custom `--ok` green to match the mockup's `.btn.success`
 *   We went with (A). The OUTPUT corner tag + the Succeeded StatusBand
 *   upstream already signal "done"; the button doesn't need to be green to
 *   reinforce that. Keeping the app-wide primary-CTA vocabulary consistent
 *   (same blue as "Start translation", "Rejoin running job") reads as more
 *   cohesive than a one-off green.
 *
 * We render a plain `<a download>` styled via `buttonVariants` instead of
 * wrapping `<Button asChild>` because right-click → Save As works on any
 * anchor with no extra wiring.
 */

import { Download } from "lucide-react";

import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export interface ResultPanelProps {
  jobId: string;
  outputUrl: string;
}

export function ResultPanel({ jobId, outputUrl }: ResultPanelProps): JSX.Element {
  const jobIdShort = jobId.slice(0, 8);

  return (
    <div className="rounded-md border border-border overflow-hidden bg-[color:var(--bg-1)] flex flex-col">
      <div className="relative bg-black">
        {/* Corner tag — cosmetic, so `aria-hidden`. Positioned absolute so it
            floats over the first frame without pushing layout. */}
        <div
          aria-hidden
          className="absolute top-2 left-2 z-10 rounded bg-background/80 px-2 py-1 font-mono text-[9px] uppercase tracking-wider text-muted-foreground"
        >
          OUTPUT
        </div>
        <video
          controls
          preload="metadata"
          className="w-full aspect-video bg-black"
          aria-label={`Translated output for job ${jobId}`}
        >
          <source src={outputUrl} type="video/mp4" />
          Your browser does not support embedded video playback.
        </video>
      </div>

      <div className="flex items-center gap-3 px-4 py-3 border-t border-border">
        <div className="flex min-w-0 flex-1 flex-col">
          <span className="text-xs text-muted-foreground">Translated MP4</span>
          <span className="font-mono text-[11px] text-[color:var(--ink-3)] truncate">
            job {jobIdShort}…
          </span>
        </div>
        <a
          href={outputUrl}
          download={`job-${jobId}-output.mp4`}
          className={cn(
            buttonVariants({ variant: "default" }),
            "no-underline",
          )}
        >
          <Download aria-hidden />
          Download
        </a>
      </div>
    </div>
  );
}
