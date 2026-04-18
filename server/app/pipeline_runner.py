"""Real PipelineRunner: loads the VideoPipeline and runs it for one job.

Design (plan.md D10, D11, D13, R6):

* Build a ``PipelineConfig`` per request by loading ``code/config/adv.yaml``
  and overriding ``input_video``, ``output_video``,
  ``translation.source_lang``, ``translation.target_lang``. All other knobs
  stay at their YAML defaults. (D13)

* Attach a ``logging.Handler`` to the pipeline-side ``src`` logger tree at
  the start of the run. The handler converts every INFO / WARNING / ERROR /
  CRITICAL ``LogRecord`` into a ``LogEvent`` and forwards it through the
  ``emit`` callable that ``JobManager`` supplies. Detached in a ``finally``
  block so a failed run doesn't leave stale handlers dangling. (D10)

* Wrap ``VideoPipeline``'s string-based ``progress_callback`` into
  ``StageStartEvent`` / ``StageCompleteEvent`` models. Start timestamps are
  tracked per-stage so completion events carry a real ``duration_ms``. (D11)

* Emit exactly one ``DoneEvent`` on success. On any pipeline exception,
  raise — ``JobManager``'s ``except`` block synthesizes the ``ErrorEvent``
  through the same emit closure. The runner never constructs an
  ``ErrorEvent`` directly.

* Keep ``VideoPipeline`` / ``PipelineConfig`` imports *inside* function
  bodies so ``import server.app.pipeline_runner`` does not pull in torch /
  paddle. FastAPI startup stays light, and non-GPU tests can import this
  module freely.

* ``sys.path`` shim (R6): the pipeline code uses
  ``sys.path.insert(0, <repo>/code)`` to resolve ``src.*``. Mirror that at
  import time so the lazy imports below find the pipeline package without
  re-running ``scripts/run_pipeline.py``-style bootstrap inside each
  function. See ``code/scripts/run_pipeline.py`` for the pattern.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from .schemas import (
    DoneEvent,
    LogEvent,
    SSEEvent,
    StageCompleteEvent,
    StageStartEvent,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# sys.path shim — make <repo>/code/src importable (R6).
# server/app/pipeline_runner.py  →  parents[2]  =  <repo_root>
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CODE_DIR = _REPO_ROOT / "code"
_ADV_YAML = _CODE_DIR / "config" / "adv.yaml"

if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))


# ---------------------------------------------------------------------------
# Stage-number → canonical stage code mapping used in the SSE schema.
# Keep in sync with server.app.schemas.Stage (Literal["s1"..."s5"]).
# ---------------------------------------------------------------------------
_STAGE_MAP: dict[str, str] = {
    "1": "s1", "2": "s2", "3": "s3", "4": "s4", "5": "s5",
}


def _parse_stage_event(event_str: str) -> tuple[str, str] | None:
    """Parse a ``VideoPipeline.progress_callback`` string.

    Returns ``(stage_code, phase)`` — e.g. ``("s1", "start")`` or
    ``("s3", "done")`` — for recognized inputs, or ``None`` for anything
    else. Strict parser: the pipeline contract is that the string is
    exactly ``"stage_{N}_{start|done}"`` for ``N in 1..5``. Anything else
    is treated as garbage and ignored.
    """
    if not event_str:
        return None
    parts = event_str.split("_")
    if len(parts) != 3 or parts[0] != "stage":
        return None
    n, phase = parts[1], parts[2]
    if n not in _STAGE_MAP or phase not in ("start", "done"):
        return None
    return _STAGE_MAP[n], phase


class _PipelineLogHandler(logging.Handler):
    """Bridge ``logging.LogRecord`` → SSE ``LogEvent`` (plan.md D10).

    Attached to the ``src`` logger tree on runner entry, detached in a
    ``finally`` block on exit. Filtered to INFO / WARNING / ERROR /
    CRITICAL so DEBUG noise stays out of the client log panel; CRITICAL
    collapses to "error" since ``LogLevel`` is a closed ``Literal``.
    """

    _LEVEL_MAP: dict[int, str] = {
        logging.INFO: "info",
        logging.WARNING: "warning",
        logging.ERROR: "error",
        logging.CRITICAL: "error",
    }

    def __init__(self, emit: Callable[[SSEEvent], None]):
        super().__init__(level=logging.INFO)
        self._emit = emit

    def emit(self, record: logging.LogRecord) -> None:  # noqa: A003
        level = self._LEVEL_MAP.get(record.levelno)
        if level is None:
            return
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 — defensive: never crash the pipeline
            message = str(record.msg)
        # `level` is guaranteed to be one of the LogLevel literal members
        # by the mapping above; the type checker still can't prove it.
        self._emit(LogEvent(level=level, message=message, ts=time.time()))  # type: ignore[arg-type]


def _transcode_to_browser_safe(output_path: Path) -> None:
    """Re-encode ``output_path`` in place to H.264 + faststart (plan.md R3).

    OpenCV's ``VideoWriter`` with the ``mp4v`` fourcc on this build of
    libavcodec emits ``FMP4``-tagged streams (MPEG-4 Part 2). Chrome and
    Firefox refuse to play those in ``<video>``, so we shell out to ffmpeg
    after the pipeline finishes and rewrite the file as ``avc1`` /
    ``yuv420p`` with the moov atom at the head (``+faststart``) so it
    streams cleanly over HTTP.

    Strategy: encode to a sibling ``.browser.mp4``, then atomically swap.
    If ffmpeg fails we surface a ``RuntimeError`` so ``JobManager`` flips
    the job to ``failed`` (rather than silently leaving an unplayable
    file behind).

    Cost: ffmpeg is a separate process, so this scales with output length
    not pipeline complexity. For a 5-second 720p clip on this box it
    completes in well under a second; the cost is negligible compared to
    the pipeline's S3 + S4 stages.
    """
    tmp_path = output_path.with_suffix(".browser.mp4")
    cmd = [
        "ffmpeg",
        "-y",                       # overwrite tmp_path if it exists
        "-loglevel", "error",       # quiet; we capture stderr on failure
        "-i", str(output_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",      # required for broad browser support
        "-movflags", "+faststart",  # moov atom at head → streamable
        str(tmp_path),
    ]
    try:
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ffmpeg not found on PATH; required for browser-safe MP4 "
                "transcode (plan.md R3). Install via `apt install ffmpeg`."
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = (
                exc.stderr.decode("utf-8", errors="replace")
                if exc.stderr else ""
            )
            raise RuntimeError(
                f"ffmpeg transcode failed (rc={exc.returncode}): {stderr}"
            ) from exc

        # Atomic swap — `Path.replace` is atomic on POSIX when src and dst
        # are on the same filesystem (they are, by construction: siblings).
        # After this, tmp_path no longer exists.
        tmp_path.replace(output_path)
    finally:
        # Any failure path (ffmpeg missing, ffmpeg errored, unexpected
        # PermissionError/OSError, etc.) must not leak the half-written
        # tmp_path. If the swap succeeded, tmp_path is already gone and
        # this is a no-op.
        if tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()


def _resolve_checkpoint_path(raw: str | None) -> str | None:
    """Resolve a checkpoint path relative to ``code/``.

    Paths in ``adv.yaml`` are relative to ``code/`` because the CLI
    (``code/scripts/run_pipeline.py``) runs with that as CWD. The web
    server runs uvicorn from the repo root, so those same relative paths
    resolve one level too high. Normalize by anchoring to ``_CODE_DIR``.
    Absolute paths pass through unchanged; ``None`` stays ``None``.
    """
    if raw is None:
        return None
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    return str((_CODE_DIR / p).resolve())


def _build_config(
    *,
    input_path: Path,
    output_path: Path,
    source_lang: str,
    target_lang: str,
):
    """Load adv.yaml and override per-request fields (plan.md D13).

    The ``PipelineConfig`` import is lazy so callers of
    ``run_pipeline_job`` don't pull in torch via the pipeline's transitive
    imports. ``config.input_video`` / ``config.output_video`` are top-level
    strings; language codes live under ``config.translation``.

    Relative checkpoint paths in ``adv.yaml`` are resolved against
    ``code/`` (mirrors CLI CWD); see ``_resolve_checkpoint_path``.
    """
    # Lazy imports — see module docstring for rationale (R6 + light server).
    from src.config import PipelineConfig  # type: ignore[import-not-found]

    # NOTE: relies on `from_yaml` returning a fresh object per call. If that
    # changes (e.g. YAML parse cached), the mutations below would race across
    # concurrent jobs. JobManager is single-worker today, so this is safe.
    config = PipelineConfig.from_yaml(str(_ADV_YAML))
    config.input_video = str(input_path)
    config.output_video = str(output_path)
    config.translation.source_lang = source_lang
    config.translation.target_lang = target_lang

    # Resolve CLI-relative checkpoint paths against code/ so uvicorn's
    # repo-root CWD doesn't break loads. hasattr-guarded so test fakes
    # (minimal PipelineConfig stubs without these subsections) don't blow up.
    _FIELDS = (
        ("detection", "cotracker_checkpoint"),
        ("detection", "cotracker_online_checkpoint"),
        ("propagation", "inpainter_checkpoint_path"),
        ("propagation", "bpn_checkpoint_path"),
        ("revert", "pre_inpaint_checkpoint"),
        ("revert", "refiner_checkpoint_path"),
    )
    for section_name, field_name in _FIELDS:
        section = getattr(config, section_name, None)
        if section is None or not hasattr(section, field_name):
            continue
        setattr(
            section,
            field_name,
            _resolve_checkpoint_path(getattr(section, field_name)),
        )

    # S5 alignment refiner is optional and its checkpoint is not shipped with
    # the repo. Disable it if the resolved path doesn't exist so the pipeline
    # runs without the refiner rather than crashing at load time.
    revert = getattr(config, "revert", None)
    if (
        revert is not None
        and getattr(revert, "use_refiner", False)
        and getattr(revert, "refiner_checkpoint_path", None)
        and not Path(revert.refiner_checkpoint_path).exists()
    ):
        logger.warning(
            "revert.refiner_checkpoint_path missing (%s); disabling refiner",
            revert.refiner_checkpoint_path,
        )
        revert.use_refiner = False

    return config


def _run_demo_failure(
    fail_stage: str,
    emit: Callable[[SSEEvent], None],
) -> None:
    """DEMO: mock stage-start/complete events for every stage BEFORE
    `fail_stage`, then emit stage-start for `fail_stage` and return so the
    caller can raise. Keeps the stage strip visually consistent with the
    05-failed mockup (prior stages done, failing stage red, later pending).
    """
    stages = ["s1", "s2", "s3", "s4", "s5"]
    idx = stages.index(fail_stage)
    for prior in stages[:idx]:
        t0 = time.time()
        emit(StageStartEvent(stage=prior, ts=t0))  # type: ignore[arg-type]
        time.sleep(0.5)
        t1 = time.time()
        emit(
            StageCompleteEvent(  # type: ignore[arg-type]
                stage=prior,
                ts=t1,
                duration_ms=(t1 - t0) * 1000.0,
            )
        )
    emit(StageStartEvent(stage=fail_stage, ts=time.time()))  # type: ignore[arg-type]
    # brief dwell so the fail-stage tile registers as active before the raise
    time.sleep(0.5)


def run_pipeline_job(
    *,
    job_id: str,
    input_path: Path,
    output_path: Path,
    source_lang: str,
    target_lang: str,
    emit: Callable[[SSEEvent], None],
) -> None:
    """PipelineRunner entrypoint — conforms to ``JobManager.PipelineRunner``.

    Executed on the ``JobManager`` worker thread. Steps:
      1. Build a ``PipelineConfig`` from ``adv.yaml`` + per-request overrides.
      2. Attach a ``logging.Handler`` to the ``src`` logger tree that
         forwards log records to the ``emit`` callback as ``LogEvent`` s.
      3. Adapt the pipeline's ``progress_callback`` strings to
         ``StageStartEvent`` / ``StageCompleteEvent``, computing
         ``duration_ms = (done_ts - start_ts) * 1000``.
      4. Run the pipeline.
      5. Emit exactly one ``DoneEvent`` on success.

    Never emits an ``ErrorEvent`` directly — on failure the log handler is
    detached and the exception is re-raised so ``JobManager`` can
    synthesize the terminal ``ErrorEvent`` atomically with the status flip
    (D16).
    """
    # ------------------------------------------------------------------
    # DEMO HOOK — scripted failure for UI testing. Commented out; flip
    # the block back on + set DEMO_FAIL_STAGE=s1..s5 in the uvicorn env
    # to fake a crash at the named stage. See _run_demo_failure below.
    # ------------------------------------------------------------------
    # demo_fail_stage = os.environ.get("DEMO_FAIL_STAGE", "").strip().lower()
    # if demo_fail_stage in {"s1", "s2", "s3", "s4", "s5"}:
    #     _run_demo_failure(demo_fail_stage, emit)
    #     raise RuntimeError(
    #         f"CUDA out of memory at Stage {demo_fail_stage[1:]}\n"
    #         "Tried to allocate 2.3 GiB on a GPU that had 1.8 GiB free. "
    #         "Another process may be holding memory, or the edit model is "
    #         "too large for this GPU."
    #     )

    # Lazy import — module-level import must stay torch-free.
    from src.pipeline import VideoPipeline  # type: ignore[import-not-found]

    config = _build_config(
        input_path=input_path,
        output_path=output_path,
        source_lang=source_lang,
        target_lang=target_lang,
    )

    # Per-stage start timestamps → duration_ms on completion.
    stage_start_ts: dict[str, float] = {}

    def progress_cb(event_str: str) -> None:
        parsed = _parse_stage_event(event_str)
        if parsed is None:
            # Unknown progress strings are logged (debug — this is internal
            # pipeline telemetry, not user-facing) and swallowed. Crashing
            # the pipeline over a telemetry quirk would be a bad trade.
            logger.debug("Unparsed progress event: %r", event_str)
            return
        stage_code, phase = parsed
        now = time.time()
        if phase == "start":
            stage_start_ts[stage_code] = now
            emit(StageStartEvent(stage=stage_code, ts=now))  # type: ignore[arg-type]
        else:  # phase == "done"
            started = stage_start_ts.pop(stage_code, now)
            duration_ms = (now - started) * 1000.0
            emit(
                StageCompleteEvent(
                    stage=stage_code,  # type: ignore[arg-type]
                    duration_ms=duration_ms,
                    ts=now,
                )
            )

    # Attach the log bridge. We install on the `src` logger (parent of
    # `src.pipeline`, `src.stages.*`, etc.) so every pipeline logger in the
    # tree feeds through it. Ensure INFO propagates: if the parent level is
    # NOTSET or above INFO, bump it.
    #
    # NOTE — we LOWER the level but never restore it. A naive
    # save-prior-and-restore pattern races under concurrent runs: thread A
    # saves level L and bumps to INFO; thread B saves INFO (already bumped)
    # and bumps to INFO; thread A finishes and restores L; thread B finishes
    # and restores INFO — corrupting the original L. JobManager is
    # single-worker today, so the race can't fire in practice, but the fix
    # is cheap: since the only mutation is LOWER to INFO (never raise),
    # leaving it lowered is always safe. Pipeline logs are the only
    # consumer of this tree in our process.
    handler = _PipelineLogHandler(emit)
    src_logger = logging.getLogger("src")
    src_logger.addHandler(handler)
    if src_logger.level == logging.NOTSET or src_logger.level > logging.INFO:
        src_logger.setLevel(logging.INFO)

    try:
        pipeline = VideoPipeline(config, progress_callback=progress_cb)
        pipeline.run()
        # R3 mitigation: convert OpenCV's mp4v/FMP4 output to H.264 so
        # browsers can play it. Skip if the pipeline never wrote a file
        # (defensive — DoneEvent still fires; the route layer reports
        # "output not available" via /status if so).
        if output_path.exists():
            _transcode_to_browser_safe(output_path)
        emit(
            DoneEvent(
                output_url=f"/api/jobs/{job_id}/output",
                ts=time.time(),
            )
        )
    finally:
        # Always detach, even on exception. JobManager handles ErrorEvent.
        # Intentionally no setLevel restore — see note above.
        src_logger.removeHandler(handler)
