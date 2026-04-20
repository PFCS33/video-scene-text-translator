import type { Stage } from "@/api/schemas";

/** The five pipeline stages, ordered s1..s5. */
export const STAGES: readonly Stage[] = ["s1", "s2", "s3", "s4", "s5"] as const;

/** Human-facing label for each stage. Matches the backend's stage vocabulary. */
export const STAGE_LABEL: Readonly<Record<Stage, string>> = {
  s1: "Detect",
  s2: "Frontalize",
  s3: "Edit",
  s4: "Propagate",
  s5: "Revert",
};

/**
 * Elapsed time (ms) after which the active-stage tile surfaces a stall
 * warning. Kept in rough agreement with the server's default
 * `PIPELINE_LIVENESS_TIMEOUT_S = 180` in
 * `server/app/pipeline_runner.py` — both signals are independent
 * ("no client-side progress tick" vs. "no server-side emit"), but
 * aligning the thresholds means the badge and the watchdog log-line tend
 * to surface together.
 *
 * MANUAL SYNC POINT: if the server default changes, update this constant
 * so the two thresholds don't drift apart. No runtime coupling — the
 * client never reads the server's value.
 */
export const STALL_THRESHOLD_MS = 180_000;
