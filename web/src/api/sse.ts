/**
 * Wrapper around the browser `EventSource` for the pipeline SSE stream.
 *
 * The FastAPI route (server/app/routes.py::stream_events) emits each event
 * with an `event:` name equal to `SSEEvent.type` (set via sse_starlette's
 * `event` key) and the JSON-serialized payload as `data:`. We register one
 * named listener per event type and forward all of them through a single
 * typed `onEvent` callback.
 *
 * Reconnect (plan.md D16)
 * -----------------------
 * Browsers auto-reconnect dropped EventSource connections, but events that
 * fire during the gap are lost. MVP accepts log-line loss, but the UI needs
 * to stay honest about which stage is running — so on reconnect we fetch
 * `/status` once and hand the result to `onStatusSync`. The consumer can
 * use that to force the progress bar forward.
 */

import type { JobStatus, SSEEvent } from "./schemas";
import { eventsUrl, getJobStatus } from "./client";

export type EventListener = (event: SSEEvent) => void;
export type StatusListener = (status: JobStatus) => void;
export type ErrorListener = (error: unknown) => void;

export interface StreamOptions {
  /** Fired for every well-formed SSE event. */
  onEvent: EventListener;
  /** Fired once per reconnect attempt with the current status snapshot. */
  onStatusSync?: StatusListener;
  /** Fired on framing/parse errors and transport errors. */
  onError?: ErrorListener;
}

export interface EventStream {
  close(): void;
  readonly readyState: number;
}

/** Event names emitted by the server. Kept in sync with SSEEvent.type literals. */
const EVENT_NAMES = [
  "stage_start",
  "stage_complete",
  "log",
  "done",
  "error",
] as const;

/**
 * Parse a raw SSE `data:` payload into an `SSEEvent`.
 *
 * EventSource already handles frame assembly, so this helper just JSON-parses
 * the payload and performs minimal structural checks; full validation lives
 * server-side (Pydantic `extra="forbid"`). Exported for unit tests.
 */
export function parseSseData(raw: string): SSEEvent {
  const obj = JSON.parse(raw);
  if (typeof obj !== "object" || obj === null) {
    throw new Error(`invalid SSE payload (not an object): ${raw}`);
  }
  if (typeof (obj as { type?: unknown }).type !== "string") {
    throw new Error(`invalid SSE payload (missing type): ${raw}`);
  }
  return obj as SSEEvent;
}

export function openEventStream(
  jobId: string,
  options: StreamOptions,
): EventStream {
  const source = new EventSource(eventsUrl(jobId));

  for (const name of EVENT_NAMES) {
    source.addEventListener(name, (ev) => {
      try {
        const data = parseSseData((ev as MessageEvent).data);
        options.onEvent(data);
      } catch (err) {
        options.onError?.(err);
      }
    });
  }

  source.onerror = async (ev) => {
    // D16: on auto-reconnect, the browser will flip readyState to CONNECTING
    // before the socket comes back. Poll /status once so the UI doesn't
    // drift during the gap.
    if (source.readyState === EventSource.CONNECTING && options.onStatusSync) {
      try {
        const status = await getJobStatus(jobId);
        options.onStatusSync(status);
      } catch (err) {
        options.onError?.(err);
        return;
      }
    }
    options.onError?.(ev);
  };

  return {
    close() {
      source.close();
    },
    get readyState() {
      return source.readyState;
    },
  };
}
