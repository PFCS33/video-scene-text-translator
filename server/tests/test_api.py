"""End-to-end HTTP tests for the six /api routes (plan.md Step 7).

The pipeline is always stubbed — these tests never touch GPU / AnyText2.
Each test builds a fresh FastAPI app (with its own lifespan) bound to:

* a caller-supplied ``PipelineRunner`` stub, and
* a tmp-path storage root (via the ``SERVER_STORAGE_ROOT`` env var).

Rationale for the per-test app factory
--------------------------------------

``JobManager.__init__`` calls ``asyncio.get_running_loop()`` so it must be
constructed inside a running loop. ``TestClient(app)`` used as a context
manager (``with TestClient(app) as c:``) drives the app's ``lifespan``
hook, which is the event-loop-aware construction point. The factory
``_make_app(runner)`` builds an app whose lifespan binds *our* stub
runner, sidestepping the module-level ``server.app.main.app`` which
hardcodes ``run_pipeline_job``.

SSE caveat
----------

Starlette's ``TestClient`` supports streaming responses via
``client.stream("GET", url)`` — we iterate ``response.iter_lines()``
to parse out ``event:`` / ``data:`` SSE frames. No ``httpx.ASGITransport``
gymnastics needed.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Iterable
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.app import routes
from server.app.jobs import JobManager, PipelineRunner
from server.app.routes import get_manager, router
from server.app.schemas import (
    DoneEvent,
    SSEEvent,
    StageCompleteEvent,
    StageStartEvent,
)

# ---------------------------------------------------------------------------
# App / client factory
# ---------------------------------------------------------------------------


def _make_app(runner: PipelineRunner) -> FastAPI:
    """Build a fresh FastAPI app wired to `runner` via a custom lifespan."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        mgr = JobManager(runner=runner)
        app.state.job_manager = mgr
        app.dependency_overrides[get_manager] = lambda: app.state.job_manager
        try:
            yield
        finally:
            mgr.shutdown(wait=True)

    app = FastAPI(title="test-app", lifespan=lifespan)
    app.include_router(router)

    @app.get("/api/health")
    def health() -> dict[str, str]:  # mirror main.py for the smoke test
        return {"status": "ok"}

    return app


@pytest.fixture(autouse=True)
def _reset_sse_starlette_app_status():
    """Clear sse-starlette's module-global ``AppStatus`` between tests.

    ``sse_starlette.sse.AppStatus.should_exit_event`` is lazily created
    as an ``anyio.Event`` on the first SSE request and bound to the
    event loop that served it. ``TestClient`` spins up a fresh loop per
    ``with TestClient(...)`` block, so a later test reusing the same
    process trips over "bound to a different event loop" when two SSE
    tests run in sequence. Resetting the module-global before and after
    each test dodges this cleanly.

    This fixture already exists in test_integration.py for the same
    reason — we mirror it here because test_api also opens SSE streams.
    """
    from sse_starlette import sse as _sse

    _sse.AppStatus.should_exit_event = None
    _sse.AppStatus.should_exit = False
    yield
    _sse.AppStatus.should_exit_event = None
    _sse.AppStatus.should_exit = False


@pytest.fixture
def storage_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect server storage to a tmp dir for the duration of the test."""
    root = tmp_path / "server-storage"
    root.mkdir()
    monkeypatch.setenv("SERVER_STORAGE_ROOT", str(root))
    return root


# ---------------------------------------------------------------------------
# Stub PipelineRunners
# ---------------------------------------------------------------------------


def _simple_runner_factory() -> PipelineRunner:
    """Runner that emits s1 start+complete, writes a dummy MP4, then done."""

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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"FAKE MP4 BYTES")
        emit(
            DoneEvent(
                output_url=f"/api/jobs/{job_id}/output", ts=time.time()
            )
        )

    return runner


def _blocking_runner_factory(release: threading.Event) -> PipelineRunner:
    """Runner that emits StageStart, then blocks on `release`, then done+file."""

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
            raise AssertionError("blocking runner never released")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"FAKE MP4 BYTES")
        emit(
            DoneEvent(
                output_url=f"/api/jobs/{job_id}/output", ts=time.time()
            )
        )

    return runner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def _post_job(
    client: TestClient,
    *,
    video_bytes: bytes = b"tinyvideo",
    filename: str = "clip.mp4",
    source_lang: str = "en",
    target_lang: str = "es",
):
    return client.post(
        "/api/jobs",
        files={"video": (filename, video_bytes, "video/mp4")},
        data={"source_lang": source_lang, "target_lang": target_lang},
    )


def _wait_for_status(
    client: TestClient,
    job_id: str,
    *,
    terminal: Iterable[str],
    timeout: float = 5.0,
) -> dict:
    """Poll /status until it returns one of `terminal`."""
    terminal_set = set(terminal)
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        resp = client.get(f"/api/jobs/{job_id}/status")
        assert resp.status_code == 200, resp.text
        last = resp.json()
        if last["status"] in terminal_set:
            return last
        time.sleep(0.02)
    raise AssertionError(
        f"job {job_id} never reached {terminal_set}; last status={last}"
    )


def _parse_sse_stream(resp) -> list[dict]:
    """Parse an SSE response into a list of ``{event, data}`` dicts.

    Each SSE frame is separated by a blank line; lines within a frame
    look like ``event: <type>`` / ``data: <json>``. We intentionally ignore
    ``id:`` / ``retry:`` / comment lines (``: ...``) since the server
    doesn't emit them.
    """
    frames: list[dict] = []
    current: dict = {}
    for raw in resp.iter_lines():
        # httpx yields str for iter_lines; starlette TestClient does too.
        line = raw if isinstance(raw, str) else raw.decode("utf-8")
        if line == "":
            if current:
                frames.append(current)
                current = {}
            continue
        if line.startswith(":"):
            # SSE comment (sse-starlette emits periodic pings as comments)
            continue
        if line.startswith("event:"):
            current["event"] = line[len("event:") :].strip()
        elif line.startswith("data:"):
            current["data"] = line[len("data:") :].strip()
    if current:
        frames.append(current)
    return frames


# ---------------------------------------------------------------------------
# GET /api/languages
# ---------------------------------------------------------------------------


def test_get_languages_returns_seven_curated_codes(storage_root: Path):
    # Arrange
    app = _make_app(_simple_runner_factory())

    # Act
    with TestClient(app) as client:
        resp = client.get("/api/languages")

    # Assert
    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload) == 7
    codes = {lang["code"] for lang in payload}
    assert codes == {"en", "es", "zh-CN", "fr", "de", "ja", "ko"}


# ---------------------------------------------------------------------------
# GET /api/health (sanity on the rebuilt app)
# ---------------------------------------------------------------------------


def test_health_still_ok_on_rebuilt_app(storage_root: Path):
    # Arrange
    app = _make_app(_simple_runner_factory())

    # Act
    with TestClient(app) as client:
        resp = client.get("/api/health")

    # Assert
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /api/jobs — happy path + validation + concurrency + size
# ---------------------------------------------------------------------------


def test_create_job_returns_uuid_job_id(storage_root: Path):
    # Arrange
    app = _make_app(_simple_runner_factory())

    # Act
    with TestClient(app) as client:
        resp = _post_job(client)

        # Assert — body shape
        assert resp.status_code == 200, resp.text
        body = resp.json()
        job_id = body["job_id"]
        assert _UUID_RE.match(job_id), f"not a uuid4: {job_id}"

        # Drain so the worker thread finishes cleanly before the fixture
        # tears down the lifespan (which shuts down the executor).
        _wait_for_status(client, job_id, terminal={"succeeded", "failed"})


def test_create_job_persists_upload_under_job_id_dir(storage_root: Path):
    """The uploaded bytes must land at uploads/<job_id>/<filename>."""
    # Arrange
    app = _make_app(_simple_runner_factory())
    payload = b"hello-upload"

    # Act
    with TestClient(app) as client:
        resp = _post_job(client, video_bytes=payload, filename="myclip.mp4")
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        # Assert — file exists with exact bytes under uploads/<id>/
        upload_file = storage_root / "uploads" / job_id / "myclip.mp4"
        assert upload_file.exists()
        assert upload_file.read_bytes() == payload

        _wait_for_status(client, job_id, terminal={"succeeded", "failed"})


def test_create_job_rejects_unknown_source_lang(storage_root: Path):
    # Arrange
    app = _make_app(_simple_runner_factory())

    # Act
    with TestClient(app) as client:
        resp = _post_job(client, source_lang="klingon")

    # Assert
    assert resp.status_code == 400
    assert "klingon" in resp.json()["detail"]


def test_create_job_rejects_unknown_target_lang(storage_root: Path):
    # Arrange
    app = _make_app(_simple_runner_factory())

    # Act
    with TestClient(app) as client:
        resp = _post_job(client, target_lang="klingon")

    # Assert
    assert resp.status_code == 400


def test_create_job_rejects_oversized_upload(
    storage_root: Path, monkeypatch: pytest.MonkeyPatch
):
    # Arrange — shrink the cap to 1 KiB for test speed.
    monkeypatch.setattr(routes, "MAX_UPLOAD_BYTES", 1024)
    app = _make_app(_simple_runner_factory())
    oversize = b"A" * 2048

    # Act
    with TestClient(app) as client:
        resp = _post_job(client, video_bytes=oversize)

    # Assert
    assert resp.status_code == 413
    assert "exceeds" in resp.json()["detail"]
    # Partial upload cleaned up — uploads/<anything>/ should not linger.
    uploads_root = storage_root / "uploads"
    if uploads_root.exists():
        # cleanup_job removes the per-job dir; no per-job dir should remain
        assert list(uploads_root.iterdir()) == []


def test_create_job_returns_409_with_active_job_id_on_concurrent_submit(
    storage_root: Path,
):
    """Second POST while one is running → 409 + active_job_id in detail."""
    # Arrange
    release = threading.Event()
    app = _make_app(_blocking_runner_factory(release))

    # Act
    with TestClient(app) as client:
        try:
            # Submit A
            resp_a = _post_job(client)
            assert resp_a.status_code == 200, resp_a.text
            job_a = resp_a.json()["job_id"]

            # Wait until /status says A is actually running (first
            # StageStart emitted → record.current_stage = "s1").
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                st = client.get(f"/api/jobs/{job_a}/status").json()
                if st["status"] == "running":
                    break
                time.sleep(0.02)
            else:
                raise AssertionError("job A never transitioned to running")

            # Submit B while A is still blocked
            resp_b = _post_job(client, filename="b.mp4")

            # Assert — 409 + active_job_id
            assert resp_b.status_code == 409, resp_b.text
            detail = resp_b.json()["detail"]
            assert detail["error"] == "concurrent_job"
            assert detail["active_job_id"] == job_a
        finally:
            release.set()
            _wait_for_status(client, job_a, terminal={"succeeded", "failed"})


# ---------------------------------------------------------------------------
# GET /api/jobs/{id}/status
# ---------------------------------------------------------------------------


def test_get_status_ok_after_submit(storage_root: Path):
    # Arrange
    app = _make_app(_simple_runner_factory())

    # Act
    with TestClient(app) as client:
        job_id = _post_job(client).json()["job_id"]
        final = _wait_for_status(
            client, job_id, terminal={"succeeded", "failed"}
        )

    # Assert
    assert final["status"] == "succeeded"
    assert final["source_lang"] == "en"
    assert final["target_lang"] == "es"
    assert final["job_id"] == job_id
    assert final["output_available"] is True


def test_get_status_404_for_unknown_id(storage_root: Path):
    # Arrange
    app = _make_app(_simple_runner_factory())

    # Act
    with TestClient(app) as client:
        resp = client.get("/api/jobs/does-not-exist/status")

    # Assert
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/jobs/{id}/events  (SSE)
# ---------------------------------------------------------------------------


def test_events_endpoint_streams_sse_to_terminal(storage_root: Path):
    # Arrange — gated runner so the SSE subscription is attached BEFORE any
    # event is emitted. Under multicast fan-out (plan.md D16, no replay),
    # a subscriber that opens after an event has been broadcast cannot see
    # that event.
    #
    # Why the background thread: Starlette's ``TestClient.stream()`` blocks
    # its ``__enter__`` until the server yields the first byte. If the
    # runner is gated by a threading.Event the test thread needs to set,
    # it'll deadlock — ``stream()`` won't return so the test never reaches
    # ``event.set()``. Instead we spawn a waiter thread that polls the
    # JobManager's subscriber list and trips the gate once the SSE route's
    # ``subscribe()`` has registered its queue.
    start_emitting = threading.Event()

    def runner(
        *,
        job_id: str,
        input_path: Path,
        output_path: Path,
        source_lang: str,
        target_lang: str,
        emit: Callable[[SSEEvent], None],
    ) -> None:
        if not start_emitting.wait(timeout=5):
            raise AssertionError("SSE subscriber never attached")
        t0 = time.time()
        emit(StageStartEvent(stage="s1", ts=t0))
        emit(StageCompleteEvent(stage="s1", duration_ms=1.0, ts=t0))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"FAKE MP4 BYTES")
        emit(
            DoneEvent(
                output_url=f"/api/jobs/{job_id}/output", ts=time.time()
            )
        )

    app = _make_app(runner)

    # Act
    with TestClient(app) as client:
        job_id = _post_job(client).json()["job_id"]

        def _release_when_subscribed() -> None:
            """Poll the manager for a subscriber, then flip the gate.

            We peek at ``app.state.job_manager._jobs[job_id]._subscribers``
            (white-box) because that's the exact invariant we need: the
            route's SSE handler has called ``subscribe()`` and appended
            its per-subscriber queue, so the runner's subsequent
            ``emit()`` broadcasts will reach it.
            """
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                mgr = app.state.job_manager
                record = mgr._jobs.get(job_id)
                if record is not None and record._subscribers:
                    start_emitting.set()
                    return
                time.sleep(0.005)

        waiter = threading.Thread(target=_release_when_subscribed, daemon=True)
        waiter.start()
        try:
            with client.stream("GET", f"/api/jobs/{job_id}/events") as resp:
                frames = _parse_sse_stream(resp)
        finally:
            start_emitting.set()  # belt-and-braces so runner never deadlocks
            waiter.join(timeout=2)

    # Assert — we see at least one of each expected event kind
    types_seen = {f["event"] for f in frames if "event" in f}
    assert "stage_start" in types_seen
    assert "stage_complete" in types_seen
    assert "done" in types_seen

    # And the done frame decodes back to an output_url pointing at this job
    done_frame = next(f for f in frames if f.get("event") == "done")
    done_data = json.loads(done_frame["data"])
    assert done_data["output_url"] == f"/api/jobs/{job_id}/output"


def test_events_endpoint_multicast_to_concurrent_subscribers(storage_root: Path):
    """Two concurrent /events streams on the same job both see every event.

    This is the HTTP-layer pin on plan.md's SSE multicast fan-out. The unit
    test in test_jobs.py (`test_multiple_concurrent_subscribers_both_receive_
    all_events`) covers `JobManager.subscribe` directly; this one exercises
    the full stack — sse-starlette `EventSourceResponse`, the route handler,
    and the TestClient streaming transport — so a regression in any of those
    layers would show up here instead of silently splitting events.
    """
    # Gated runner — runner waits until BOTH SSE subscribers have attached
    # their per-subscriber queues, then emits the sequence. Ensures neither
    # subscriber misses any event to the no-replay semantic.
    start_emitting = threading.Event()

    def runner(
        *,
        job_id: str,
        input_path: Path,
        output_path: Path,
        source_lang: str,
        target_lang: str,
        emit: Callable[[SSEEvent], None],
    ) -> None:
        if not start_emitting.wait(timeout=5):
            raise AssertionError("SSE subscribers never attached")
        t0 = time.time()
        emit(StageStartEvent(stage="s1", ts=t0))
        emit(StageCompleteEvent(stage="s1", duration_ms=1.0, ts=t0))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"FAKE MP4 BYTES")
        emit(
            DoneEvent(
                output_url=f"/api/jobs/{job_id}/output", ts=time.time()
            )
        )

    app = _make_app(runner)

    with TestClient(app) as client:
        job_id = _post_job(client).json()["job_id"]

        # Two independent consumers, each on its own thread — TestClient's
        # streaming API is synchronous and blocks the calling thread until
        # the server finishes the response. Each thread collects its frames
        # into a shared list that we inspect post-run.
        results: dict[str, list[dict]] = {"a": [], "b": []}

        def _consume(label: str) -> None:
            with client.stream("GET", f"/api/jobs/{job_id}/events") as resp:
                results[label] = _parse_sse_stream(resp)

        t_a = threading.Thread(target=_consume, args=("a",), daemon=True)
        t_b = threading.Thread(target=_consume, args=("b",), daemon=True)
        t_a.start()
        t_b.start()

        # Wait for BOTH subscribers to register their queues before firing
        # the runner's emit sequence.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            mgr = app.state.job_manager
            record = mgr._jobs.get(job_id)
            if record is not None and len(record._subscribers) >= 2:
                start_emitting.set()
                break
            time.sleep(0.005)
        else:
            start_emitting.set()  # safety release if polling timed out
            raise AssertionError(
                "fewer than 2 SSE subscribers attached within 5s"
            )

        t_a.join(timeout=10)
        t_b.join(timeout=10)

    # Assert — both subscribers received the full event sequence.
    for label, frames in results.items():
        types_seen = [f["event"] for f in frames if "event" in f]
        assert "stage_start" in types_seen, (
            f"subscriber {label} missing stage_start: {types_seen}"
        )
        assert "stage_complete" in types_seen, (
            f"subscriber {label} missing stage_complete: {types_seen}"
        )
        assert "done" in types_seen, (
            f"subscriber {label} missing done: {types_seen}"
        )


def test_events_endpoint_404_for_unknown_id(storage_root: Path):
    # Arrange
    app = _make_app(_simple_runner_factory())

    # Act
    with TestClient(app) as client:
        resp = client.get("/api/jobs/nope/events")

    # Assert
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/jobs/{id}/output
# ---------------------------------------------------------------------------


def test_download_output_404_before_job_finishes(storage_root: Path):
    """While the pipeline is blocked, output_available is False → 404."""
    # Arrange
    release = threading.Event()
    app = _make_app(_blocking_runner_factory(release))

    # Act
    with TestClient(app) as client:
        try:
            job_id = _post_job(client).json()["job_id"]
            # Wait for running so we know the pipeline has started but not
            # finished (the runner is blocked on `release`).
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                st = client.get(f"/api/jobs/{job_id}/status").json()
                if st["status"] == "running":
                    break
                time.sleep(0.02)

            resp = client.get(f"/api/jobs/{job_id}/output")

            # Assert
            assert resp.status_code == 404
        finally:
            release.set()
            _wait_for_status(client, job_id, terminal={"succeeded", "failed"})


def test_download_output_200_after_done(storage_root: Path):
    # Arrange
    app = _make_app(_simple_runner_factory())

    # Act
    with TestClient(app) as client:
        job_id = _post_job(client).json()["job_id"]
        _wait_for_status(client, job_id, terminal={"succeeded"})
        resp = client.get(f"/api/jobs/{job_id}/output")

    # Assert
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "video/mp4"
    assert "attachment" in resp.headers.get("content-disposition", "")
    assert "translated.mp4" in resp.headers["content-disposition"]
    assert resp.content == b"FAKE MP4 BYTES"


def test_download_output_404_for_unknown_id(storage_root: Path):
    # Arrange
    app = _make_app(_simple_runner_factory())

    # Act
    with TestClient(app) as client:
        resp = client.get("/api/jobs/nope/output")

    # Assert
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /api/jobs/{id}
# ---------------------------------------------------------------------------


def test_delete_succeeded_job_200_and_cleans_storage(storage_root: Path):
    # Arrange
    app = _make_app(_simple_runner_factory())

    # Act
    with TestClient(app) as client:
        job_id = _post_job(client).json()["job_id"]
        _wait_for_status(client, job_id, terminal={"succeeded"})

        resp = client.delete(f"/api/jobs/{job_id}")

    # Assert — HTTP 200 + storage dirs gone
    assert resp.status_code == 200, resp.text
    assert resp.json()["deleted"] == job_id
    assert not (storage_root / "uploads" / job_id).exists()
    assert not (storage_root / "outputs" / job_id).exists()


def test_delete_running_job_returns_409(storage_root: Path):
    # Arrange
    release = threading.Event()
    app = _make_app(_blocking_runner_factory(release))

    # Act
    with TestClient(app) as client:
        try:
            job_id = _post_job(client).json()["job_id"]
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if (
                    client.get(f"/api/jobs/{job_id}/status").json()["status"]
                    == "running"
                ):
                    break
                time.sleep(0.02)

            resp = client.delete(f"/api/jobs/{job_id}")

            # Assert — 409, storage NOT cleaned up
            assert resp.status_code == 409
            assert (storage_root / "uploads" / job_id).exists()
        finally:
            release.set()
            _wait_for_status(client, job_id, terminal={"succeeded", "failed"})

        # After release, a second DELETE should succeed (200)
        resp2 = client.delete(f"/api/jobs/{job_id}")
        assert resp2.status_code == 200


def test_delete_unknown_job_returns_404(storage_root: Path):
    # Arrange
    app = _make_app(_simple_runner_factory())

    # Act
    with TestClient(app) as client:
        resp = client.delete("/api/jobs/does-not-exist")

    # Assert
    assert resp.status_code == 404
