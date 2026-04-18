/**
 * Thin fetch wrapper around the FastAPI surface in server/app/routes.py.
 *
 * Responsibilities:
 *   - Typed JSON GET/POST helpers with consistent error handling.
 *   - Multipart POST for `/api/jobs` (file + two language form fields).
 *   - `ApiError` with structured info on non-2xx, including a convenience
 *     accessor for FastAPI's 409 concurrent-job detail shape (R8).
 *
 * The `outputUrl` / `eventsUrl` helpers exist so UI code never hand-builds
 * API paths — a grep for `/api/jobs/` should only find this file.
 */

import type {
  ConcurrentJobErrorDetail,
  JobCreateResponse,
  JobStatus,
  Language,
} from "./schemas";

const BASE = "/api";

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/**
 * Thrown by every function in this module on non-2xx responses.
 *
 * `detail` holds FastAPI's unwrapped `detail` payload when the response was
 * JSON, or the raw response text otherwise. Code paths that need structured
 * handling should prefer dedicated accessors like `concurrentJobDetail`
 * rather than poking at `detail` directly.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(status: number, detail: unknown, message?: string) {
    super(message ?? `API error ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }

  /** Structured 409 payload when present, else null. */
  get concurrentJobDetail(): ConcurrentJobErrorDetail | null {
    if (this.status !== 409) return null;
    const d = this.detail;
    if (
      typeof d === "object" &&
      d !== null &&
      "error" in d &&
      (d as { error: unknown }).error === "concurrent_job"
    ) {
      return d as ConcurrentJobErrorDetail;
    }
    return null;
  }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Parse a `fetch` Response into T, throwing ApiError on non-2xx.
 *
 * FastAPI wraps error bodies as `{"detail": ...}`; we unwrap that so
 * `ApiError.detail` always points at the *inner* payload. This keeps the
 * 409 concurrent-job handling path straightforward.
 */
async function handleResponse<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    let detail: unknown = null;
    const contentType = resp.headers.get("content-type") ?? "";
    try {
      if (contentType.includes("application/json")) {
        detail = await resp.json();
      } else {
        detail = await resp.text();
      }
    } catch {
      // Body already consumed or malformed — fall through with null detail.
    }
    if (
      detail &&
      typeof detail === "object" &&
      "detail" in (detail as Record<string, unknown>)
    ) {
      detail = (detail as { detail: unknown }).detail;
    }
    throw new ApiError(resp.status, detail);
  }

  const contentType = resp.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    return (await resp.json()) as T;
  }
  // Fall back to text — used by the few endpoints that might return plain
  // text (none right now, but keeps the helper honest).
  return (await resp.text()) as unknown as T;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export async function getHealth(): Promise<{ status: string }> {
  const resp = await fetch(`${BASE}/health`);
  return handleResponse(resp);
}

export async function getLanguages(): Promise<Language[]> {
  const resp = await fetch(`${BASE}/languages`);
  return handleResponse(resp);
}

export async function createJob(
  video: File,
  sourceLang: string,
  targetLang: string,
): Promise<JobCreateResponse> {
  const body = new FormData();
  body.append("video", video);
  body.append("source_lang", sourceLang);
  body.append("target_lang", targetLang);
  const resp = await fetch(`${BASE}/jobs`, { method: "POST", body });
  return handleResponse(resp);
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const resp = await fetch(
    `${BASE}/jobs/${encodeURIComponent(jobId)}/status`,
  );
  return handleResponse(resp);
}

/**
 * Delete a terminal job. Returns the server's `{deleted, ts}` shape.
 * Throws ApiError(409) if the job is still running.
 */
export async function deleteJob(
  jobId: string,
): Promise<{ deleted: string; ts: number }> {
  const resp = await fetch(`${BASE}/jobs/${encodeURIComponent(jobId)}`, {
    method: "DELETE",
  });
  return handleResponse(resp);
}

/** URL of the final MP4 — suitable for `<a href>` or `<video src>`. */
export function outputUrl(jobId: string): string {
  return `${BASE}/jobs/${encodeURIComponent(jobId)}/output`;
}

/** URL of the SSE stream — used by `sse.ts`. */
export function eventsUrl(jobId: string): string {
  return `${BASE}/jobs/${encodeURIComponent(jobId)}/events`;
}
