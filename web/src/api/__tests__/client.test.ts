/**
 * Unit tests for api/client.ts. Mocks global `fetch` via `vi.stubGlobal`.
 *
 * We only exercise the wrapper behaviors — happy-path decoding, multipart
 * shape, FastAPI `{detail: ...}` unwrapping, and the `ApiError` accessor for
 * the 409 concurrent-job body. End-to-end integration with the real server
 * lives in tests on the server side.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  createJob,
  deleteJob,
  eventsUrl,
  getJobStatus,
  getLanguages,
  outputUrl,
} from "../client";
import type { JobStatus, Language } from "../schemas";

type FetchArgs = Parameters<typeof fetch>;

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json" },
    ...init,
  });
}

function errorJsonResponse(
  status: number,
  body: unknown,
): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("api/client", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("getLanguages returns the parsed array", async () => {
    const langs: Language[] = [
      { code: "en", label: "English" },
      { code: "es", label: "Spanish" },
    ];
    fetchMock.mockResolvedValueOnce(jsonResponse(langs));

    const out = await getLanguages();

    expect(out).toEqual(langs);
    const [url] = fetchMock.mock.calls[0] as FetchArgs;
    expect(url).toBe("/api/languages");
  });

  it("createJob sends multipart form data with the right field names", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ job_id: "abc-123" }));
    const file = new File([new Uint8Array([1, 2, 3])], "clip.mp4", {
      type: "video/mp4",
    });

    const out = await createJob(file, "en", "es");

    expect(out).toEqual({ job_id: "abc-123" });
    const [url, init] = fetchMock.mock.calls[0] as FetchArgs;
    expect(url).toBe("/api/jobs");
    expect(init?.method).toBe("POST");
    const body = init?.body as FormData;
    expect(body).toBeInstanceOf(FormData);
    expect(body.get("source_lang")).toBe("en");
    expect(body.get("target_lang")).toBe("es");
    const videoField = body.get("video");
    expect(videoField).toBeInstanceOf(File);
    expect((videoField as File).name).toBe("clip.mp4");
  });

  it("getJobStatus returns the parsed status body", async () => {
    const status: JobStatus = {
      job_id: "abc",
      status: "running",
      source_lang: "en",
      target_lang: "es",
      created_at: 1_700_000_000,
      current_stage: "s2",
      finished_at: null,
      error: null,
      output_available: false,
    };
    fetchMock.mockResolvedValueOnce(jsonResponse(status));

    const out = await getJobStatus("abc");

    expect(out).toEqual(status);
    const [url] = fetchMock.mock.calls[0] as FetchArgs;
    expect(url).toBe("/api/jobs/abc/status");
  });

  it("getJobStatus URL-encodes the job id", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({} as JobStatus));
    await getJobStatus("a b/c");
    const [url] = fetchMock.mock.calls[0] as FetchArgs;
    expect(url).toBe("/api/jobs/a%20b%2Fc/status");
  });

  it("deleteJob issues DELETE to the right URL and returns the payload", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ deleted: "abc", ts: 1_700_000_010 }),
    );

    const out = await deleteJob("abc");

    expect(out).toEqual({ deleted: "abc", ts: 1_700_000_010 });
    const [url, init] = fetchMock.mock.calls[0] as FetchArgs;
    expect(url).toBe("/api/jobs/abc");
    expect(init?.method).toBe("DELETE");
  });

  it("outputUrl and eventsUrl return the expected paths", () => {
    expect(outputUrl("abc")).toBe("/api/jobs/abc/output");
    expect(eventsUrl("abc")).toBe("/api/jobs/abc/events");
    // encoding
    expect(outputUrl("a b")).toBe("/api/jobs/a%20b/output");
  });

  it("non-2xx throws ApiError with correct status and unwrapped detail", async () => {
    fetchMock.mockResolvedValueOnce(
      errorJsonResponse(400, { detail: "unsupported source_lang: xx" }),
    );

    await expect(getLanguages()).rejects.toMatchObject({
      name: "ApiError",
      status: 400,
      detail: "unsupported source_lang: xx",
    });
  });

  it("ApiError.concurrentJobDetail extracts the 409 body", async () => {
    fetchMock.mockResolvedValueOnce(
      errorJsonResponse(409, {
        detail: { error: "concurrent_job", active_job_id: "existing-id" },
      }),
    );

    let caught: ApiError | null = null;
    try {
      await createJob(new File([], "x.mp4"), "en", "es");
    } catch (e) {
      caught = e as ApiError;
    }

    expect(caught).toBeInstanceOf(ApiError);
    expect(caught?.status).toBe(409);
    expect(caught?.concurrentJobDetail).toEqual({
      error: "concurrent_job",
      active_job_id: "existing-id",
    });
  });

  it("ApiError.concurrentJobDetail returns null for non-409 errors", async () => {
    fetchMock.mockResolvedValueOnce(
      errorJsonResponse(413, { detail: "upload too big" }),
    );

    let caught: ApiError | null = null;
    try {
      await createJob(new File([], "x.mp4"), "en", "es");
    } catch (e) {
      caught = e as ApiError;
    }

    expect(caught?.status).toBe(413);
    expect(caught?.concurrentJobDetail).toBeNull();
  });

  it("handleResponse tolerates non-JSON error bodies", async () => {
    fetchMock.mockResolvedValueOnce(
      new Response("upstream timeout", {
        status: 504,
        headers: { "content-type": "text/plain" },
      }),
    );

    await expect(getLanguages()).rejects.toMatchObject({
      name: "ApiError",
      status: 504,
      detail: "upstream timeout",
    });
  });
});
