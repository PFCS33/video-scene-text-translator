/**
 * TypeScript mirror of server/app/schemas.py. DO NOT drift.
 *
 * Any change to a Pydantic model in schemas.py MUST be mirrored here.
 * If the surface grows enough that this is error-prone, switch to
 * OpenAPI-based codegen per plan.md R7.
 *
 * Naming quirks:
 *   - Python `LogEvent`  -> TS `LogEventPayload`   (DOM `LogEvent` exists)
 *   - Python `ErrorEvent`-> TS `ErrorEventPayload` (DOM `ErrorEvent` exists)
 * The discriminator lives on the shared `type` string field, so consumers
 * narrow the union via `switch (ev.type)` without touching the class names.
 */

// ---------------------------------------------------------------------------
// Literals — keep in sync with the `Literal[...]` aliases in schemas.py.
// ---------------------------------------------------------------------------

export type Stage = "s1" | "s2" | "s3" | "s4" | "s5";
export type JobStatusLiteral = "queued" | "running" | "succeeded" | "failed";
export type LogLevel = "info" | "warning" | "error";

// ---------------------------------------------------------------------------
// SSE events — one interface per Pydantic model, tagged on `type`.
// ---------------------------------------------------------------------------

export interface StageStartEvent {
  type: "stage_start";
  stage: Stage;
  ts: number;
}

export interface StageCompleteEvent {
  type: "stage_complete";
  stage: Stage;
  duration_ms: number;
  ts: number;
}

export interface LogEventPayload {
  type: "log";
  level: LogLevel;
  message: string;
  ts: number;
}

export interface DoneEvent {
  type: "done";
  output_url: string;
  ts: number;
}

export interface ErrorEventPayload {
  type: "error";
  message: string;
  ts: number;
  traceback?: string | null;
}

export type SSEEvent =
  | StageStartEvent
  | StageCompleteEvent
  | LogEventPayload
  | DoneEvent
  | ErrorEventPayload;

// ---------------------------------------------------------------------------
// Job request/response models.
// ---------------------------------------------------------------------------

export interface JobCreateResponse {
  job_id: string;
}

export interface JobStatus {
  job_id: string;
  status: JobStatusLiteral;
  source_lang: string;
  target_lang: string;
  created_at: number;
  current_stage: Stage | null;
  finished_at: number | null;
  error: string | null;
  output_available: boolean;
}

export interface Language {
  code: string;
  label: string;
}

// ---------------------------------------------------------------------------
// Error detail shapes.
// ---------------------------------------------------------------------------

/** Detail body attached to a 409 from `POST /api/jobs` (see routes.py). */
export interface ConcurrentJobErrorDetail {
  error: "concurrent_job";
  active_job_id: string | null;
}
