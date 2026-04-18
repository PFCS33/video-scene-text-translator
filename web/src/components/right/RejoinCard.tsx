/**
 * <RejoinCard> — right-column surface for the "server busy" phase.
 *
 * The user tried to submit while another job is already running on the
 * server. The server returned 409 with `active_job_id`; the parent (`<App>`)
 * fetched `/status` for that id (D9) and passes the result here. This
 * component is pure presentation + one CTA callback — no state, no effects,
 * no network. All metadata comes from props.
 *
 * `blockingStatus` is null while the parent's /status fetch is in flight or
 * if it fails outright; the card still renders with the id + generic copy
 * so the user can always rejoin.
 *
 * Deferred behavior (plan.md): "from other session" marker, heuristic ETA,
 * and auto-resubmit-on-free are all omitted. The footer copy is kept.
 */

import type { JobStatus, Stage } from "@/api/schemas";

import { Button } from "@/components/ui/button";
import { STAGE_LABEL } from "@/lib/stages";

interface RejoinCardProps {
  blockingJobId: string;
  /** null while /status is in flight or if the fetch failed. */
  blockingStatus: JobStatus | null;
  onRejoin: () => void;
}

function formatStage(stage: Stage | null | undefined): string | null {
  if (!stage) return null;
  return `${stage.toUpperCase()} \u00B7 ${STAGE_LABEL[stage]}`;
}

function formatStartedAt(createdAt: number | null | undefined): string | null {
  if (createdAt == null) return null;
  // created_at is seconds since epoch (server convention — see schemas.py).
  return new Date(createdAt * 1000).toLocaleTimeString();
}

export function RejoinCard({
  blockingJobId,
  blockingStatus,
  onRejoin,
}: RejoinCardProps): JSX.Element {
  const headingId = "rejoin-card-heading";

  const stageLabel = formatStage(blockingStatus?.current_stage) ?? "\u2014";
  const startedLabel =
    formatStartedAt(blockingStatus?.created_at) ?? "\u2014";

  // Full id in the metadata cell so the user can copy it for debugging or
  // link against it in another tab. The StatusBand chrome still shows the
  // 8-char prefix as a compact indicator.

  return (
    <section
      aria-labelledby={headingId}
      className="w-full overflow-hidden rounded-md border border-[color:var(--warn-line)] bg-card"
    >
      {/* Decorative label strip. `aria-hidden` because its content is
          entirely duplicated by the heading + description below. */}
      <div
        aria-hidden="true"
        className="border-b border-[color:var(--warn-line)] bg-[color:var(--warn-soft)] px-3 py-1.5 font-mono text-[11px] font-semibold uppercase tracking-wider text-[color:var(--warn)]"
      >
        <span className="mr-1.5">&#x25CF;</span>
        Another job is running
      </div>

      <div className="px-6 py-5">
        <h2
          id={headingId}
          className="mt-1 text-xl font-semibold text-foreground"
        >
          Server is busy
        </h2>
        <p className="mt-2 text-sm text-muted-foreground">
          Another job is currently running on the server. Rejoin its progress
          view or wait for it to finish before submitting your own.
        </p>

        <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 border-y border-border py-3">
          <dt className="text-[11px] uppercase tracking-wider text-muted-foreground">
            Running job
          </dt>
          <dd className="break-all font-mono text-sm text-foreground">
            {blockingJobId}
          </dd>

          <dt className="text-[11px] uppercase tracking-wider text-muted-foreground">
            Stage
          </dt>
          <dd className="font-mono text-sm text-foreground">{stageLabel}</dd>

          <dt className="text-[11px] uppercase tracking-wider text-muted-foreground">
            Started
          </dt>
          <dd className="font-mono text-sm text-foreground">{startedLabel}</dd>
        </dl>

        <Button
          type="button"
          variant="default"
          className="mt-4 w-full"
          onClick={onRejoin}
        >
          Rejoin running job
        </Button>

        <p className="mt-3 text-center font-mono text-[11px] text-[color:var(--ink-3)]">
          or wait for the server to free up {"\u00B7"} your file stays queued
        </p>
      </div>
    </section>
  );
}
