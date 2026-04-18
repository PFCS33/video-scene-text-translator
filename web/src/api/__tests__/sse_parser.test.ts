/**
 * Unit tests for `parseSseData` — the framing work is done by the browser's
 * EventSource, so these tests only cover the JSON decode + structural checks.
 *
 * Each fixture mirrors the wire format we expect from server/app/routes.py
 * (sse_starlette emits `data:` as JSON-serialized `event.model_dump_json()`).
 */

import { describe, expect, it } from "vitest";

import type {
  DoneEvent,
  ErrorEventPayload,
  LogEventPayload,
  StageCompleteEvent,
  StageStartEvent,
} from "../schemas";
import { parseSseData } from "../sse";

describe("parseSseData", () => {
  it("parses a stage_start event and narrows to StageStartEvent", () => {
    const raw = JSON.stringify({
      type: "stage_start",
      stage: "s1",
      ts: 1_700_000_000.5,
    });

    const ev = parseSseData(raw);

    expect(ev.type).toBe("stage_start");
    // Narrowed to StageStartEvent — `stage` field is accessible.
    if (ev.type !== "stage_start") throw new Error("narrowing failed");
    const narrowed: StageStartEvent = ev;
    expect(narrowed.stage).toBe("s1");
    expect(narrowed.ts).toBeCloseTo(1_700_000_000.5);
  });

  it("parses a stage_complete event carrying duration_ms", () => {
    const raw = JSON.stringify({
      type: "stage_complete",
      stage: "s3",
      duration_ms: 12_345,
      ts: 1_700_000_001,
    });

    const ev = parseSseData(raw);

    if (ev.type !== "stage_complete") throw new Error("narrowing failed");
    const narrowed: StageCompleteEvent = ev;
    expect(narrowed.stage).toBe("s3");
    expect(narrowed.duration_ms).toBe(12_345);
  });

  it("parses a log event with the level field", () => {
    const raw = JSON.stringify({
      type: "log",
      level: "warning",
      message: "s4: ema fallback",
      ts: 1_700_000_002,
    });

    const ev = parseSseData(raw);

    if (ev.type !== "log") throw new Error("narrowing failed");
    const narrowed: LogEventPayload = ev;
    expect(narrowed.level).toBe("warning");
    expect(narrowed.message).toBe("s4: ema fallback");
  });

  it("parses a done event with output_url", () => {
    const raw = JSON.stringify({
      type: "done",
      output_url: "/api/jobs/abc/output",
      ts: 1_700_000_099,
    });

    const ev = parseSseData(raw);

    if (ev.type !== "done") throw new Error("narrowing failed");
    const narrowed: DoneEvent = ev;
    expect(narrowed.output_url).toBe("/api/jobs/abc/output");
  });

  it("parses an error event with optional traceback", () => {
    const rawWith = JSON.stringify({
      type: "error",
      message: "boom",
      traceback: "Traceback (most recent call last)...",
      ts: 1_700_000_050,
    });
    const rawWithout = JSON.stringify({
      type: "error",
      message: "boom",
      ts: 1_700_000_050,
    });

    const a = parseSseData(rawWith);
    const b = parseSseData(rawWithout);

    if (a.type !== "error" || b.type !== "error") {
      throw new Error("narrowing failed");
    }
    const ea: ErrorEventPayload = a;
    const eb: ErrorEventPayload = b;
    expect(ea.traceback).toMatch(/^Traceback/);
    // Omitted `traceback` decodes as undefined — acceptable for an optional
    // field tagged `str | None` on the server (Pydantic omits None by default
    // on `model_dump_json()` only when `exclude_none=True`, which we don't
    // set — so in practice the server emits `"traceback": null`, but the
    // client must tolerate both).
    expect(eb.traceback ?? null).toBeNull();
  });

  it("throws on non-JSON input", () => {
    expect(() => parseSseData("not json")).toThrow();
  });

  it("throws on a JSON object missing `type`", () => {
    const raw = JSON.stringify({ stage: "s1", ts: 1 });
    expect(() => parseSseData(raw)).toThrow(/missing type/);
  });

  it("throws on a JSON primitive (array, number, null)", () => {
    expect(() => parseSseData("null")).toThrow(/not an object/);
    expect(() => parseSseData("42")).toThrow(/not an object/);
    // Arrays ARE objects in JS land; they still fail the `type` check.
    expect(() => parseSseData("[1,2,3]")).toThrow(/missing type/);
  });
});
