"""Tests for `server.app.jobs.JobManager` (plan.md D4, D5, D11, D14, R8).

These tests use a *stubbed* pipeline runner so we never touch GPU / AnyText2.
The runner is the dependency-injected callable the JobManager invokes on its
single worker thread; tests provide thread-controlled stubs that emit a
chosen sequence of SSE events and optionally raise.

Event-loop notes:
    JobManager captures the running loop at construction via
    `asyncio.get_running_loop()`, so every test runs inside an async function
    (pytest-asyncio `auto` mode picks them up automatically).

All `await`s that wait on the worker thread use a short `asyncio.wait_for`
timeout — if a test deadlocks it fails fast instead of hanging CI.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest

from server.app.jobs import (
    ConcurrentJobError,
    JobManager,
    PipelineRunner,
    UnknownJobError,
)
from server.app.schemas import (
    DoneEvent,
    ErrorEvent,
    LogEvent,
    SSEEvent,
    StageCompleteEvent,
    StageStartEvent,
)

# ---------------------------------------------------------------------------
# Stub-runner factories — produced fresh per test so no shared state leaks.
# ---------------------------------------------------------------------------


def make_blocking_stub(release: threading.Event) -> PipelineRunner:
    """Runner that emits a StageStart, blocks on `release`, then emits Done."""

    def runner(
        *,
        job_id: str,
        input_path: Path,
        output_path: Path,
        source_lang: str,
        target_lang: str,
        emit: Callable[[SSEEvent], None],
    ) -> None:
        emit(StageStartEvent(stage="s1", ts=time.time()))
        if not release.wait(timeout=5):
            raise AssertionError("stub runner never released")
        emit(DoneEvent(output_url=f"/api/jobs/{job_id}/output", ts=time.time()))

    return runner


def make_simple_stub() -> PipelineRunner:
    """Runner that emits s1 start/complete + done and returns immediately."""

    def runner(
        *,
        job_id: str,
        input_path: Path,
        output_path: Path,
        source_lang: str,
        target_lang: str,
        emit: Callable[[SSEEvent], None],
    ) -> None:
        t0 = time.time()
        emit(StageStartEvent(stage="s1", ts=t0))
        emit(StageCompleteEvent(stage="s1", duration_ms=1.0, ts=t0))
        emit(DoneEvent(output_url=f"/api/jobs/{job_id}/output", ts=t0))

    return runner


def make_sequence_stub(events: list[SSEEvent]) -> PipelineRunner:
    """Runner that replays a fixed list of events in order."""

    def runner(
        *,
        job_id: str,
        input_path: Path,
        output_path: Path,
        source_lang: str,
        target_lang: str,
        emit: Callable[[SSEEvent], None],
    ) -> None:
        for ev in events:
            emit(ev)

    return runner


def make_raising_stub(message: str = "boom") -> PipelineRunner:
    """Runner that emits one StageStart then raises RuntimeError."""

    def runner(
        *,
        job_id: str,
        input_path: Path,
        output_path: Path,
        source_lang: str,
        target_lang: str,
        emit: Callable[[SSEEvent], None],
    ) -> None:
        emit(StageStartEvent(stage="s1", ts=time.time()))
        raise RuntimeError(message)

    return runner


async def _drain_until_terminal(mgr: JobManager, job_id: str) -> list[SSEEvent]:
    """Collect every SSE event for a job until Done or Error, with a timeout."""

    async def _collect() -> list[SSEEvent]:
        out: list[SSEEvent] = []
        async for ev in mgr.subscribe(job_id):
            out.append(ev)
        return out

    return await asyncio.wait_for(_collect(), timeout=5)


def _default_paths(tmp_path: Path, job_no: int = 0) -> tuple[Path, Path]:
    inp = tmp_path / f"in_{job_no}.mp4"
    out = tmp_path / f"out_{job_no}.mp4"
    inp.write_bytes(b"\x00")  # input exists so it's realistic
    return inp, out


# ---------------------------------------------------------------------------
# Submission + UUID allocation
# ---------------------------------------------------------------------------


async def test_submit_returns_uuid4(tmp_path: Path):
    # Arrange
    mgr = JobManager(runner=make_simple_stub())
    inp, outp = _default_paths(tmp_path)

    try:
        # Act
        job_id = await mgr.submit(
            source_lang="en", target_lang="es",
            input_path=inp, output_path=outp,
        )

        # Assert — parses as UUID4
        parsed = uuid.UUID(job_id)
        assert parsed.version == 4

        # drain so the worker thread finishes cleanly
        await _drain_until_terminal(mgr, job_id)
    finally:
        mgr.shutdown()


async def test_submit_with_explicit_job_id_uses_that_id(tmp_path: Path):
    """Caller-supplied `job_id` kwarg is honored so storage paths can be
    allocated before submit() (routes.py streams upload into
    uploads_dir(job_id) before kicking off the pipeline).
    """
    # Arrange
    mgr = JobManager(runner=make_simple_stub())
    inp, outp = _default_paths(tmp_path)
    explicit_id = str(uuid.uuid4())

    try:
        # Act
        returned_id = await mgr.submit(
            job_id=explicit_id,
            source_lang="en", target_lang="es",
            input_path=inp, output_path=outp,
        )

        # Assert
        assert returned_id == explicit_id
        await _drain_until_terminal(mgr, returned_id)
    finally:
        mgr.shutdown()


async def test_submit_with_explicit_job_id_raises_if_duplicate(tmp_path: Path):
    """Passing a job_id already in the registry must raise ValueError.

    Two successive submits with the same explicit id would otherwise collide
    on the dict key and silently overwrite the prior record.
    """
    # Arrange
    mgr = JobManager(runner=make_simple_stub())
    inp, outp = _default_paths(tmp_path)
    explicit_id = str(uuid.uuid4())

    try:
        # Act — first submit with the explicit id, drain to terminal
        await mgr.submit(
            job_id=explicit_id,
            source_lang="en", target_lang="es",
            input_path=inp, output_path=outp,
        )
        await _drain_until_terminal(mgr, explicit_id)
        await _wait_status(mgr, explicit_id, terminal={"succeeded"})

        # Assert — re-submitting with the same id raises
        inp2, outp2 = _default_paths(tmp_path, 99)
        with pytest.raises(ValueError):
            await mgr.submit(
                job_id=explicit_id,
                source_lang="en", target_lang="es",
                input_path=inp2, output_path=outp2,
            )
    finally:
        mgr.shutdown()


async def test_second_submit_while_running_raises_concurrent(tmp_path: Path):
    # Arrange — stub A blocks until we release it.
    release_a = threading.Event()
    mgr = JobManager(runner=make_blocking_stub(release_a))
    inp_a, out_a = _default_paths(tmp_path, 0)
    inp_b, out_b = _default_paths(tmp_path, 1)

    try:
        job_a = await mgr.submit(
            source_lang="en", target_lang="es",
            input_path=inp_a, output_path=out_a,
        )

        # Wait for A to transition to running (first StageStart arrives).
        # This guarantees `status in {queued, running}` for the concurrency check.
        async def _first_event() -> SSEEvent:
            async for ev in mgr.subscribe(job_a):
                return ev
            raise AssertionError("subscribe ended before yielding any event")
        first = await asyncio.wait_for(_first_event(), timeout=5)
        assert isinstance(first, StageStartEvent)

        # Act — second submit must fail with ConcurrentJobError(job_a).
        with pytest.raises(ConcurrentJobError) as excinfo:
            await mgr.submit(
                source_lang="en", target_lang="es",
                input_path=inp_b, output_path=out_b,
            )

        # Assert — the error carries A's job_id so the UX can offer "rejoin".
        assert excinfo.value.args[0] == job_a
    finally:
        release_a.set()
        # try to let A finish gracefully before shutdown
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                _wait_status(mgr, job_a, terminal={"succeeded", "failed"}),
                timeout=5,
            )
        mgr.shutdown()


async def test_submit_after_prior_completes_succeeds(tmp_path: Path):
    # Arrange
    mgr = JobManager(runner=make_simple_stub())
    inp_a, out_a = _default_paths(tmp_path, 0)
    inp_b, out_b = _default_paths(tmp_path, 1)

    try:
        # Act — run A to completion
        job_a = await mgr.submit(
            source_lang="en", target_lang="es",
            input_path=inp_a, output_path=out_a,
        )
        await _drain_until_terminal(mgr, job_a)
        await _wait_status(mgr, job_a, terminal={"succeeded"})

        # Then submit B
        job_b = await mgr.submit(
            source_lang="en", target_lang="fr",
            input_path=inp_b, output_path=out_b,
        )

        # Assert
        assert job_b != job_a
        uuid.UUID(job_b)  # must parse
        await _drain_until_terminal(mgr, job_b)
    finally:
        mgr.shutdown()


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------


async def test_status_transitions_queued_running_succeeded(tmp_path: Path):
    # Arrange — stub emits 1 start, sleeps briefly, 1 complete, done.
    def runner(*, job_id, input_path, output_path, source_lang, target_lang, emit):
        t0 = time.time()
        emit(StageStartEvent(stage="s2", ts=t0))
        time.sleep(0.05)
        emit(StageCompleteEvent(stage="s2", duration_ms=50.0, ts=time.time()))
        emit(DoneEvent(output_url=f"/api/jobs/{job_id}/output", ts=time.time()))

    mgr = JobManager(runner=runner)
    inp, outp = _default_paths(tmp_path)

    try:
        # Act — submit + peek status right away
        job_id = await mgr.submit(
            source_lang="en", target_lang="es",
            input_path=inp, output_path=outp,
        )
        early = mgr.get_status(job_id)
        assert early.status in ("queued", "running")

        # Drain to completion
        events = await _drain_until_terminal(mgr, job_id)
        assert isinstance(events[-1], DoneEvent)

        # Assert — terminal state is clean. No polling: the emit() closure
        # flips status atomically with enqueuing the DoneEvent (D16 fix), so
        # by the time `subscribe()` yielded the DoneEvent the record is
        # already in its terminal state.
        final = mgr.get_status(job_id)
        assert final.status == "succeeded"
        assert final.finished_at is not None and final.finished_at >= final.created_at
        assert final.current_stage is None
        assert final.error is None
    finally:
        mgr.shutdown()


async def test_status_failed_on_runner_exception(tmp_path: Path):
    # Arrange
    mgr = JobManager(runner=make_raising_stub("boom"))
    inp, outp = _default_paths(tmp_path)

    try:
        # Act
        job_id = await mgr.submit(
            source_lang="en", target_lang="es",
            input_path=inp, output_path=outp,
        )
        events = await _drain_until_terminal(mgr, job_id)

        # Assert — terminal event is an ErrorEvent w/ traceback
        terminal = events[-1]
        assert isinstance(terminal, ErrorEvent)
        assert terminal.message == "boom"
        assert terminal.traceback is not None
        assert "RuntimeError" in terminal.traceback

        # And the record reflects the failure. No polling needed — emit()
        # flips status in lock-step with enqueuing the ErrorEvent (D16 fix).
        status = mgr.get_status(job_id)
        assert status.status == "failed"
        assert status.error == "boom"
        assert status.finished_at is not None
    finally:
        mgr.shutdown()


async def test_current_stage_updates_via_stage_start(tmp_path: Path):
    # Arrange — runner emits StageStart("s3") then blocks so we can peek.
    release = threading.Event()

    def runner(*, job_id, input_path, output_path, source_lang, target_lang, emit):
        emit(StageStartEvent(stage="s3", ts=time.time()))
        release.wait(timeout=5)
        emit(DoneEvent(output_url="/x", ts=time.time()))

    mgr = JobManager(runner=runner)
    inp, outp = _default_paths(tmp_path)

    try:
        job_id = await mgr.submit(
            source_lang="en", target_lang="es",
            input_path=inp, output_path=outp,
        )

        # Wait for the StageStart event so we know current_stage was applied
        # before peeking.
        async def _first() -> SSEEvent:
            async for ev in mgr.subscribe(job_id):
                return ev
            raise AssertionError("empty stream")
        first = await asyncio.wait_for(_first(), timeout=5)
        assert isinstance(first, StageStartEvent)

        # Act / Assert — current_stage reflects the emitted event
        status = mgr.get_status(job_id)
        assert status.current_stage == "s3"
        assert status.status == "running"
    finally:
        release.set()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                _wait_status(mgr, job_id, terminal={"succeeded", "failed"}),
                timeout=5,
            )
        mgr.shutdown()


# ---------------------------------------------------------------------------
# Subscribe() semantics
# ---------------------------------------------------------------------------


async def test_subscribe_yields_all_events_in_order(tmp_path: Path):
    # Arrange — fixed sequence
    t = time.time()
    sequence: list[SSEEvent] = [
        LogEvent(level="info", message="hello", ts=t),
        StageStartEvent(stage="s1", ts=t),
        StageCompleteEvent(stage="s1", duration_ms=5.0, ts=t),
        DoneEvent(output_url="/out", ts=t),
    ]
    mgr = JobManager(runner=make_sequence_stub(sequence))
    inp, outp = _default_paths(tmp_path)

    try:
        # Act
        job_id = await mgr.submit(
            source_lang="en", target_lang="es",
            input_path=inp, output_path=outp,
        )
        events = await _drain_until_terminal(mgr, job_id)

        # Assert — exact sequence preserved
        assert len(events) == len(sequence)
        for observed, expected in zip(events, sequence, strict=True):
            assert type(observed) is type(expected)
            assert observed.model_dump() == expected.model_dump()
    finally:
        mgr.shutdown()


async def test_subscribe_ends_on_error(tmp_path: Path):
    # Arrange
    mgr = JobManager(runner=make_raising_stub("kaboom"))
    inp, outp = _default_paths(tmp_path)

    try:
        # Act
        job_id = await mgr.submit(
            source_lang="en", target_lang="es",
            input_path=inp, output_path=outp,
        )
        events = await _drain_until_terminal(mgr, job_id)

        # Assert — iterator terminated, last event is ErrorEvent
        assert isinstance(events[-1], ErrorEvent)
    finally:
        mgr.shutdown()


# ---------------------------------------------------------------------------
# Delete + active-id lookup
# ---------------------------------------------------------------------------


async def test_delete_running_job_raises_concurrent(tmp_path: Path):
    # Arrange
    release = threading.Event()
    mgr = JobManager(runner=make_blocking_stub(release))
    inp, outp = _default_paths(tmp_path)

    try:
        job_id = await mgr.submit(
            source_lang="en", target_lang="es",
            input_path=inp, output_path=outp,
        )

        # Wait for running status via the first emitted event
        async def _first() -> SSEEvent:
            async for ev in mgr.subscribe(job_id):
                return ev
            raise AssertionError("empty stream")
        await asyncio.wait_for(_first(), timeout=5)

        # Act / Assert
        with pytest.raises(ConcurrentJobError):
            await mgr.delete(job_id)
    finally:
        release.set()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                _wait_status(mgr, job_id, terminal={"succeeded", "failed"}),
                timeout=5,
            )
        mgr.shutdown()


async def test_delete_succeeded_job_removes_from_registry(tmp_path: Path):
    # Arrange
    mgr = JobManager(runner=make_simple_stub())
    inp, outp = _default_paths(tmp_path)

    try:
        job_id = await mgr.submit(
            source_lang="en", target_lang="es",
            input_path=inp, output_path=outp,
        )
        await _drain_until_terminal(mgr, job_id)
        await _wait_status(mgr, job_id, terminal={"succeeded"})

        # Act
        await mgr.delete(job_id)

        # Assert — registry no longer knows about the job
        with pytest.raises(UnknownJobError):
            mgr.get_status(job_id)
    finally:
        mgr.shutdown()


async def test_get_active_job_id_tracks_lifecycle(tmp_path: Path):
    # Arrange
    release = threading.Event()
    mgr = JobManager(runner=make_blocking_stub(release))
    inp, outp = _default_paths(tmp_path)

    try:
        # Assert — empty registry → None
        assert await mgr.get_active_job_id() is None

        # Submit → some job_id
        job_id = await mgr.submit(
            source_lang="en", target_lang="es",
            input_path=inp, output_path=outp,
        )
        assert await mgr.get_active_job_id() == job_id

        # Let the job finish → None again
        release.set()
        await _drain_until_terminal(mgr, job_id)
        await _wait_status(mgr, job_id, terminal={"succeeded"})
        assert await mgr.get_active_job_id() is None
    finally:
        release.set()
        mgr.shutdown()


async def test_unknown_job_id_raises(tmp_path: Path):
    # Arrange
    mgr = JobManager(runner=make_simple_stub())

    try:
        # Act / Assert
        with pytest.raises(UnknownJobError):
            mgr.get_status("nope")
        with pytest.raises(UnknownJobError):
            await mgr.delete("nope")
    finally:
        mgr.shutdown()


# ---------------------------------------------------------------------------
# Terminal-state atomicity — pins the fix for the subscribe/status race (D16).
# ---------------------------------------------------------------------------


async def test_status_is_terminal_when_subscriber_sees_done_event(tmp_path: Path):
    """After subscribe() yields DoneEvent, get_status must return 'succeeded'.

    The client reconnect flow (plan.md D16) re-syncs via GET /status before
    re-subscribing. If the worker thread hasn't flipped `status` by the time
    SSE delivers the terminal event, a freshly-reconnected client that polls
    /status will see `status="running"` for a small window and may wrongly
    re-subscribe to an already-finished job.
    """
    # Arrange — stub that emits only DoneEvent (no StageStart) and returns.
    def stub(*, job_id, input_path, output_path, source_lang, target_lang, emit):
        emit(DoneEvent(output_url=f"/api/jobs/{job_id}/output", ts=time.time()))

    mgr = JobManager(runner=stub)
    inp, outp = _default_paths(tmp_path)

    try:
        # Act
        job_id = await mgr.submit(
            source_lang="en", target_lang="es",
            input_path=inp, output_path=outp,
        )

        # Drain until the DoneEvent, then IMMEDIATELY (no polling) check status.
        async def _check() -> None:
            async for event in mgr.subscribe(job_id):
                if isinstance(event, DoneEvent):
                    status = mgr.get_status(job_id)
                    # Critical: status must already be terminal at this point.
                    assert status.status == "succeeded", (
                        f"expected succeeded, got {status.status}"
                    )
                    assert status.finished_at is not None
                    assert status.current_stage is None
                    return
            raise AssertionError("stream ended without DoneEvent")

        await asyncio.wait_for(_check(), timeout=5)
    finally:
        mgr.shutdown()


async def test_status_is_terminal_when_subscriber_sees_error_event(tmp_path: Path):
    """After subscribe() yields ErrorEvent, get_status must return 'failed'.

    Same race concern as the DoneEvent case — the reconnect-via-/status flow
    requires the terminal record state to be visible before or atomically with
    the SSE terminal event.
    """
    # Arrange — stub that raises immediately, no StageStart emitted.
    def stub(*, job_id, input_path, output_path, source_lang, target_lang, emit):
        raise RuntimeError("boom")

    mgr = JobManager(runner=stub)
    inp, outp = _default_paths(tmp_path)

    try:
        # Act
        job_id = await mgr.submit(
            source_lang="en", target_lang="es",
            input_path=inp, output_path=outp,
        )

        async def _check() -> None:
            async for event in mgr.subscribe(job_id):
                if isinstance(event, ErrorEvent):
                    status = mgr.get_status(job_id)
                    assert status.status == "failed"
                    assert status.error == "boom"
                    assert status.finished_at is not None
                    return
            raise AssertionError("stream ended without ErrorEvent")

        await asyncio.wait_for(_check(), timeout=5)
    finally:
        mgr.shutdown()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _wait_status(
    mgr: JobManager,
    job_id: str,
    *,
    terminal: set[str],
    timeout: float = 5.0,
) -> None:
    """Poll `get_status` until the record reaches one of `terminal` statuses.

    `subscribe` yields the terminal SSE event before the worker thread has
    finished writing `status = "succeeded" | "failed"` and `finished_at`.
    Callers that assert on those fields need this tiny settle-loop.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if mgr.get_status(job_id).status in terminal:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(
        f"job {job_id} did not reach {terminal} within {timeout}s"
    )
