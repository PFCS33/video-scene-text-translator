"""In-memory job registry + single-worker pipeline executor (plan.md D4, D5, R8).

`JobManager` is the central coordinator for the web-client backend:

* Owns a single `ThreadPoolExecutor(max_workers=1)` so only one pipeline run
  is ever in flight (D4). A second submit while another job is active raises
  `ConcurrentJobError(active_job_id)`; the route handler turns that into a
  409 response whose body points the frontend at the running job (R8 —
  "rejoin existing run" UX).
* Holds a `dict[job_id, _JobRecord]` in memory — no persistence (MVP).
* Exposes a per-job `asyncio.Queue` populated from the worker thread via
  `loop.call_soon_threadsafe`. Consumers iterate events via `subscribe()`;
  the stream terminates on the first `DoneEvent` or `ErrorEvent` (D5).
* Surfaces `get_active_job_id()` so the frontend can offer a rejoin link
  instead of a hard 409 (R8).

The pipeline itself is injected as a `PipelineRunner` callable — tests stub
it to avoid touching GPU/AnyText2, and Step 6 will wire the real
`VideoPipeline` wrapper. The runner emits SSE events via an `emit` callback
the manager wraps; the runner never touches queues directly.

Event-loop gotcha:
    `__init__` calls `asyncio.get_running_loop()` to capture the loop for
    `call_soon_threadsafe`. That means JobManager MUST be constructed from
    inside a running event loop (FastAPI lifespan or pytest-asyncio). This is
    an intentional constraint — without a captured loop we'd have to
    introspect the current thread's loop on every emit, which is fragile.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
import uuid
from collections.abc import AsyncIterator, Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .schemas import (
    DoneEvent,
    ErrorEvent,
    SSEEvent,
    StageStartEvent,
)

logger = logging.getLogger(__name__)


class ConcurrentJobError(Exception):
    """Raised when a new job is submitted while another is active.

    The first positional arg is the active job_id so callers (route handler,
    tests) can surface a rejoin link.
    """


class UnknownJobError(KeyError):
    """Raised when a `job_id` is not present in the registry."""


class PipelineRunner(Protocol):
    """A pipeline runner callable used by `JobManager`.

    Invoked on the manager's worker thread with a job's identity + an `emit`
    callback. `emit` is thread-safe and enqueues an SSE event on the job's
    queue via `call_soon_threadsafe`.

    The runner may emit any `SSEEvent`. On normal completion it returns
    `None`; on failure it raises — the manager translates the exception into
    a terminal `ErrorEvent` and marks the job failed.

    Step 6 provides the real runner wrapping `VideoPipeline`. Tests provide
    stubs that emit a fixed sequence and optionally raise.
    """

    def __call__(
        self,
        *,
        job_id: str,
        input_path: Path,
        output_path: Path,
        source_lang: str,
        target_lang: str,
        emit: Callable[[SSEEvent], None],
    ) -> None: ...


@dataclass
class _JobRecord:
    """Mutable state for a single job.

    Fields prefixed `_` are internal bookkeeping and never leak outside the
    manager. The `events` queue is populated from the worker thread via
    `loop.call_soon_threadsafe` and consumed by `subscribe()` on the loop.
    """

    job_id: str
    status: str  # "queued" | "running" | "succeeded" | "failed"
    source_lang: str
    target_lang: str
    input_path: Path
    output_path: Path
    created_at: float
    finished_at: float | None = None
    current_stage: str | None = None
    error: str | None = None
    events: asyncio.Queue = field(default_factory=asyncio.Queue)
    _stage_start_ts: dict[str, float] = field(default_factory=dict)
    _future: Future | None = None


class JobManager:
    """In-memory job registry + single-worker pipeline executor."""

    def __init__(self, runner: PipelineRunner):
        self._runner = runner
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._jobs: dict[str, _JobRecord] = {}
        # Capture the running loop so the worker thread can schedule queue
        # puts via call_soon_threadsafe (see module docstring).
        self._loop = asyncio.get_running_loop()
        # Guards both _jobs mutations and the concurrency check so two
        # simultaneous submits can't both pass the "no active job" test.
        self._lock = asyncio.Lock()

    # ------------------------- public API -------------------------

    async def submit(
        self,
        *,
        source_lang: str,
        target_lang: str,
        input_path: Path,
        output_path: Path,
        job_id: str | None = None,
    ) -> str:
        """Enqueue a new pipeline run. Returns the new `job_id`.

        When `job_id` is omitted, a fresh UUID4 is allocated. Callers that
        need to reserve storage paths *before* submit (the route handler
        streams the upload into `uploads_dir(job_id)` before kicking off the
        pipeline) may supply their own id; passing one that's already in the
        registry raises `ValueError`.

        Raises `ConcurrentJobError(active_job_id)` if another job is already
        queued or running.
        """
        async with self._lock:
            active = self._active_job_id_locked()
            if active is not None:
                raise ConcurrentJobError(active)
            if job_id is None:
                job_id = str(uuid.uuid4())
            elif job_id in self._jobs:
                raise ValueError(f"job_id {job_id!r} already in registry")
            record = _JobRecord(
                job_id=job_id,
                status="queued",
                source_lang=source_lang,
                target_lang=target_lang,
                input_path=input_path,
                output_path=output_path,
                created_at=time.time(),
            )
            self._jobs[job_id] = record
            record._future = self._executor.submit(self._run_job, job_id)
            logger.info("submitted job %s", job_id)
        return job_id

    async def get_active_job_id(self) -> str | None:
        """Return the currently active (queued or running) job_id, if any."""
        async with self._lock:
            return self._active_job_id_locked()

    def get_status(self, job_id: str):
        """Return a `JobStatus` snapshot for `job_id`.

        Raises `UnknownJobError` if the id is unknown. Safe to call from any
        thread (read-only dict access + attribute reads).
        """
        # Imported lazily to avoid a cycle at module import time — schemas.py
        # imports from languages.py, not jobs.py, so this is purely a
        # cosmetic convenience.
        from .schemas import JobStatus

        record = self._jobs.get(job_id)
        if record is None:
            raise UnknownJobError(job_id)
        return JobStatus(
            job_id=record.job_id,
            status=record.status,  # type: ignore[arg-type]
            current_stage=record.current_stage,  # type: ignore[arg-type]
            created_at=record.created_at,
            finished_at=record.finished_at,
            error=record.error,
            source_lang=record.source_lang,
            target_lang=record.target_lang,
            output_available=(
                record.status == "succeeded" and record.output_path.exists()
            ),
        )

    async def delete(self, job_id: str) -> None:
        """Remove a terminal job from the registry.

        Raises `UnknownJobError` if the id is unknown, `ConcurrentJobError`
        if the job is still queued or running (D14). Does NOT delete files on
        disk — callers should invoke `storage.cleanup_job()` separately.
        """
        async with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise UnknownJobError(job_id)
            if record.status in ("queued", "running"):
                raise ConcurrentJobError(job_id)
            del self._jobs[job_id]
            logger.info("deleted job %s from registry", job_id)

    async def subscribe(self, job_id: str) -> AsyncIterator[SSEEvent]:
        """Async-iterate every SSE event for a job.

        Terminates on the first `DoneEvent` or `ErrorEvent`. The terminal
        event is yielded before the iterator returns, so callers always see
        the final outcome.

        Raises `UnknownJobError` if the job is unknown.
        """
        record = self._jobs.get(job_id)
        if record is None:
            raise UnknownJobError(job_id)
        while True:
            event = await record.events.get()
            yield event
            if isinstance(event, (DoneEvent, ErrorEvent)):
                return

    def shutdown(self, wait: bool = True) -> None:
        """Stop the worker executor. Call from app shutdown hook / tests."""
        self._executor.shutdown(wait=wait)

    # ------------------------- internal -------------------------

    def _active_job_id_locked(self) -> str | None:
        """Return the id of the queued/running job, if any.

        Caller must hold `self._lock`.
        """
        for rec in self._jobs.values():
            if rec.status in ("queued", "running"):
                return rec.job_id
        return None

    def _emit_threadsafe(self, job_id: str, event: SSEEvent) -> None:
        """Enqueue `event` on a job's queue from any thread."""
        record = self._jobs.get(job_id)
        if record is None:
            return
        # call_soon_threadsafe schedules on the main loop; asyncio.Queue is
        # not thread-safe but put_nowait inside the loop's thread is fine.
        self._loop.call_soon_threadsafe(record.events.put_nowait, event)

    def _run_job(self, job_id: str) -> None:
        """Worker-thread entrypoint: run the pipeline and emit terminal event.

        The nested `emit` closure is the single point where SSE events get
        enqueued. It also mutates the `_JobRecord` *before* calling
        `_emit_threadsafe` for any terminal event (DoneEvent / ErrorEvent),
        which closes the race between `subscribe()` yielding the terminal
        event and `/status` still reporting `"running"` (see plan.md D16:
        client reconnect re-syncs via GET /status before re-subscribing).
        """
        record = self._jobs[job_id]
        record.status = "running"

        def emit(event: SSEEvent) -> None:
            # Update current_stage as StageStart events fly by so
            # /status reflects where the pipeline actually is.
            if isinstance(event, StageStartEvent):
                record.current_stage = event.stage
                record._stage_start_ts[event.stage] = event.ts
            elif isinstance(event, DoneEvent):
                # Flip terminal state BEFORE enqueuing, so any client that
                # races to /status immediately after receiving DoneEvent
                # observes "succeeded" rather than "running".
                record.status = "succeeded"
                record.finished_at = time.time()
                record.current_stage = None
            elif isinstance(event, ErrorEvent):
                record.status = "failed"
                record.finished_at = time.time()
                record.current_stage = None
                record.error = event.message
            self._emit_threadsafe(job_id, event)

        try:
            self._runner(
                job_id=job_id,
                input_path=record.input_path,
                output_path=record.output_path,
                source_lang=record.source_lang,
                target_lang=record.target_lang,
                emit=emit,
            )
        except Exception as exc:  # noqa: BLE001
            # Route the exception through `emit` so status is flipped to
            # "failed" atomically with the ErrorEvent enqueue (same
            # terminal-state invariant as the success path above).
            err_event = ErrorEvent(
                message=str(exc),
                traceback=traceback.format_exc(),
                ts=time.time(),
            )
            logger.exception("job %s failed", job_id)
            emit(err_event)
            return

        # Safety net: runners are expected to emit DoneEvent themselves, but
        # a forgetful stub (or a future runner bug) could return without one.
        # Synthesize a DoneEvent so every job terminates with exactly one
        # Done or Error event. `emit` handles the status flip.
        if record.status == "running":
            done_event = DoneEvent(
                output_url=f"/api/jobs/{job_id}/output",
                ts=time.time(),
            )
            emit(done_event)
        logger.info("job %s succeeded", job_id)
