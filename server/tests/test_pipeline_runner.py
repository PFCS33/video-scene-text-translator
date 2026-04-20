"""Tests for `server.app.pipeline_runner` (plan.md D10, D11, D13, R6).

Why the `sys.modules` dance:
    `pipeline_runner.py` does `from src.pipeline import VideoPipeline` and
    `from src.config import PipelineConfig` *inside* function bodies (lazy),
    so the top-level module import stays free of torch/paddle. To verify the
    wiring without actually loading the real pipeline, each test injects fake
    `src.pipeline` / `src.config` modules into `sys.modules` via monkeypatch
    *before* calling `run_pipeline_job`. The lazy imports then hit the fakes.

    The runner also prepends `<repo>/code/` to `sys.path` on import (R6).
    That's harmless for these tests because the fakes already occupy
    `sys.modules["src.pipeline"]` / `sys.modules["src.config"]` — the import
    machinery never touches the real filesystem.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from server.app.pipeline_runner import (
    _LivenessWatchdog,
    _parse_stage_event,
    _PipelineLogHandler,
    _read_liveness_timeout,
    _transcode_to_browser_safe,
    run_pipeline_job,
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
# Helpers: fake src.pipeline / src.config modules
# ---------------------------------------------------------------------------


def _install_fake_config(monkeypatch, captured: dict | None = None) -> None:
    """Install a `src.config` module with a `PipelineConfig.from_yaml` stub.

    If `captured` is provided, it will be populated with the constructed
    config object(s) under the key `"configs"` so tests can assert on
    override behavior.
    """
    fake_config_module = types.ModuleType("src.config")

    class FakePipelineConfig:
        def __init__(self):
            self.input_video = None
            self.output_video = None
            self.translation = types.SimpleNamespace(
                source_lang=None, target_lang=None
            )

        @classmethod
        def from_yaml(cls, path: str):
            inst = cls()
            if captured is not None:
                captured.setdefault("yaml_paths", []).append(path)
                captured.setdefault("configs", []).append(inst)
            return inst

    fake_config_module.PipelineConfig = FakePipelineConfig
    monkeypatch.setitem(sys.modules, "src.config", fake_config_module)

    # Make sure the umbrella `src` package exists too, otherwise `from src.config
    # import ...` inside pipeline_runner will try to find a real `src` on disk.
    if "src" not in sys.modules:
        fake_src = types.ModuleType("src")
        monkeypatch.setitem(sys.modules, "src", fake_src)


def _install_fake_pipeline(monkeypatch, pipeline_cls) -> None:
    """Install a `src.pipeline` module whose `VideoPipeline` is `pipeline_cls`."""
    fake_pipeline_module = types.ModuleType("src.pipeline")
    fake_pipeline_module.VideoPipeline = pipeline_cls
    monkeypatch.setitem(sys.modules, "src.pipeline", fake_pipeline_module)
    if "src" not in sys.modules:
        monkeypatch.setitem(sys.modules, "src", types.ModuleType("src"))


def _make_stage_emitter_pipeline(
    stages: list[str], sleep_between: float = 0.0
):
    """Return a VideoPipeline stand-in that fires the given progress strings."""

    class FakeVideoPipeline:
        def __init__(self, config, progress_callback=None):
            self.config = config
            self.progress_callback = progress_callback

        def run(self):
            if self.progress_callback is not None:
                for s in stages:
                    if sleep_between:
                        time.sleep(sleep_between)
                    self.progress_callback(s)
            return MagicMock()

    return FakeVideoPipeline


# ---------------------------------------------------------------------------
# _parse_stage_event — pure unit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "s, expected",
    [
        ("stage_1_start", ("s1", "start")),
        ("stage_2_start", ("s2", "start")),
        ("stage_3_done", ("s3", "done")),
        ("stage_5_done", ("s5", "done")),
    ],
)
def test_parse_stage_event_recognizes_valid(s: str, expected):
    assert _parse_stage_event(s) == expected


@pytest.mark.parametrize(
    "s",
    [
        "garbage",
        "stage_6_start",           # invalid N
        "stage_0_start",           # invalid N
        "stage_1_middle",          # invalid phase
        "stage_1_start_extra",     # too many parts
        "",
        "stage__start",            # empty N
        "not_a_stage_1_start",     # extra prefix
    ],
)
def test_parse_stage_event_rejects_garbage(s: str):
    assert _parse_stage_event(s) is None


# ---------------------------------------------------------------------------
# _PipelineLogHandler — forwards INFO/WARNING/ERROR, drops DEBUG
# ---------------------------------------------------------------------------


def _make_record(level: int, message: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="src.test",
        level=level,
        pathname=__file__,
        lineno=0,
        msg=message,
        args=None,
        exc_info=None,
    )


def test_log_handler_forwards_info_warning_error():
    collected: list[SSEEvent] = []
    handler = _PipelineLogHandler(collected.append)

    handler.handle(_make_record(logging.INFO, "info-msg"))
    handler.handle(_make_record(logging.WARNING, "warn-msg"))
    handler.handle(_make_record(logging.ERROR, "err-msg"))
    handler.handle(_make_record(logging.DEBUG, "debug-msg"))  # dropped

    assert len(collected) == 3
    levels = [e.level for e in collected if isinstance(e, LogEvent)]
    messages = [e.message for e in collected if isinstance(e, LogEvent)]
    assert levels == ["info", "warning", "error"]
    assert messages == ["info-msg", "warn-msg", "err-msg"]
    # All timestamps are floats in a recent window.
    now = time.time()
    for e in collected:
        assert isinstance(e, LogEvent)
        assert now - 5 < e.ts <= now + 1


def test_log_handler_maps_critical_to_error():
    collected: list[SSEEvent] = []
    handler = _PipelineLogHandler(collected.append)
    handler.handle(_make_record(logging.CRITICAL, "boom"))
    assert len(collected) == 1
    assert isinstance(collected[0], LogEvent)
    assert collected[0].level == "error"


# ---------------------------------------------------------------------------
# run_pipeline_job — happy path: stage events + done
# ---------------------------------------------------------------------------


_ALL_STAGE_STRINGS = [
    "stage_1_start", "stage_1_done",
    "stage_2_start", "stage_2_done",
    "stage_3_start", "stage_3_done",
    "stage_4_start", "stage_4_done",
    "stage_5_start", "stage_5_done",
]


def test_run_pipeline_emits_stage_and_done_events(tmp_path: Path, monkeypatch):
    # Arrange
    _install_fake_config(monkeypatch)
    _install_fake_pipeline(
        monkeypatch, _make_stage_emitter_pipeline(_ALL_STAGE_STRINGS)
    )

    collected: list[SSEEvent] = []

    # Act
    run_pipeline_job(
        job_id="abc123",
        input_path=tmp_path / "in.mp4",
        output_path=tmp_path / "out.mp4",
        source_lang="en",
        target_lang="es",
        emit=collected.append,
    )

    # Assert
    starts = [e for e in collected if isinstance(e, StageStartEvent)]
    completes = [e for e in collected if isinstance(e, StageCompleteEvent)]
    dones = [e for e in collected if isinstance(e, DoneEvent)]
    errors = [e for e in collected if isinstance(e, ErrorEvent)]

    assert [e.stage for e in starts] == ["s1", "s2", "s3", "s4", "s5"]
    assert [e.stage for e in completes] == ["s1", "s2", "s3", "s4", "s5"]
    for e in completes:
        assert e.duration_ms >= 0

    assert len(dones) == 1
    assert dones[0].output_url == "/api/jobs/abc123/output"
    assert errors == []


def test_stage_complete_duration_is_positive(tmp_path: Path, monkeypatch):
    # Arrange — force ~10ms between stage_1_start and stage_1_done.
    _install_fake_config(monkeypatch)

    class SlowS1Pipeline:
        def __init__(self, config, progress_callback=None):
            self.progress_callback = progress_callback

        def run(self):
            assert self.progress_callback is not None
            self.progress_callback("stage_1_start")
            time.sleep(0.01)
            self.progress_callback("stage_1_done")

    _install_fake_pipeline(monkeypatch, SlowS1Pipeline)

    collected: list[SSEEvent] = []

    # Act
    run_pipeline_job(
        job_id="j1",
        input_path=tmp_path / "in.mp4",
        output_path=tmp_path / "out.mp4",
        source_lang="en",
        target_lang="es",
        emit=collected.append,
    )

    # Assert — loose ≥5ms threshold to avoid timing flake on loaded CI.
    s1_complete = next(
        e for e in collected
        if isinstance(e, StageCompleteEvent) and e.stage == "s1"
    )
    assert s1_complete.duration_ms >= 5.0


# ---------------------------------------------------------------------------
# Log handler integration — capture pipeline logs emitted during run()
# ---------------------------------------------------------------------------


def test_log_handler_captured_during_run(tmp_path: Path, monkeypatch):
    # Arrange — pipeline.run() logs a message on the `src.pipeline` logger.
    _install_fake_config(monkeypatch)

    class LoggingPipeline:
        def __init__(self, config, progress_callback=None):
            self.progress_callback = progress_callback

        def run(self):
            logging.getLogger("src.pipeline").info("hello from S1")

    _install_fake_pipeline(monkeypatch, LoggingPipeline)
    collected: list[SSEEvent] = []

    # Act
    run_pipeline_job(
        job_id="j1",
        input_path=tmp_path / "in.mp4",
        output_path=tmp_path / "out.mp4",
        source_lang="en",
        target_lang="es",
        emit=collected.append,
    )

    # Assert — our handler bridged the log record into the SSE stream.
    logs = [e for e in collected if isinstance(e, LogEvent)]
    matches = [e for e in logs if e.message == "hello from S1"]
    assert len(matches) == 1
    assert matches[0].level == "info"


def test_log_handler_removed_after_run(tmp_path: Path, monkeypatch):
    # Arrange
    _install_fake_config(monkeypatch)
    _install_fake_pipeline(
        monkeypatch, _make_stage_emitter_pipeline([])  # no progress events
    )
    src_logger = logging.getLogger("src")
    before = [
        h for h in src_logger.handlers if isinstance(h, _PipelineLogHandler)
    ]
    before_level = src_logger.level

    # Act
    run_pipeline_job(
        job_id="j1",
        input_path=tmp_path / "in.mp4",
        output_path=tmp_path / "out.mp4",
        source_lang="en",
        target_lang="es",
        emit=lambda e: None,
    )

    # Assert — no lingering handler, and level is restored.
    after = [
        h for h in src_logger.handlers if isinstance(h, _PipelineLogHandler)
    ]
    assert len(after) == len(before) == 0
    assert src_logger.level == before_level


def test_log_handler_removed_on_exception(tmp_path: Path, monkeypatch):
    # Arrange
    _install_fake_config(monkeypatch)

    class CrashingPipeline:
        def __init__(self, config, progress_callback=None):
            pass

        def run(self):
            raise RuntimeError("kaboom")

    _install_fake_pipeline(monkeypatch, CrashingPipeline)
    src_logger = logging.getLogger("src")
    before_level = src_logger.level
    collected: list[SSEEvent] = []

    # Act + Assert — exception propagates up.
    with pytest.raises(RuntimeError, match="kaboom"):
        run_pipeline_job(
            job_id="j1",
            input_path=tmp_path / "in.mp4",
            output_path=tmp_path / "out.mp4",
            source_lang="en",
            target_lang="es",
            emit=collected.append,
        )

    # Handler removed, level restored, no DoneEvent emitted.
    leftover = [
        h for h in src_logger.handlers if isinstance(h, _PipelineLogHandler)
    ]
    assert leftover == []
    assert src_logger.level == before_level
    assert not any(isinstance(e, DoneEvent) for e in collected)


# ---------------------------------------------------------------------------
# Config override — D13
# ---------------------------------------------------------------------------


def test_config_override_sets_input_output_and_langs(tmp_path: Path, monkeypatch):
    # Arrange — capture the config object the runner mutates.
    captured: dict = {}
    _install_fake_config(monkeypatch, captured=captured)
    _install_fake_pipeline(
        monkeypatch, _make_stage_emitter_pipeline([])
    )

    # Act
    run_pipeline_job(
        job_id="j1",
        input_path=Path("/tmp/a.mp4"),
        output_path=Path("/tmp/b.mp4"),
        source_lang="en",
        target_lang="zh-CN",
        emit=lambda e: None,
    )

    # Assert — from_yaml loaded the adv.yaml file and overrides landed.
    assert len(captured["configs"]) == 1
    cfg = captured["configs"][0]
    assert cfg.input_video == "/tmp/a.mp4"
    assert cfg.output_video == "/tmp/b.mp4"
    assert cfg.translation.source_lang == "en"
    assert cfg.translation.target_lang == "zh-CN"

    # And the loader was pointed at adv.yaml specifically.
    assert len(captured["yaml_paths"]) == 1
    assert captured["yaml_paths"][0].endswith("config/adv.yaml")


# ---------------------------------------------------------------------------
# Unknown progress strings — logged, not raised
# ---------------------------------------------------------------------------


def test_unparsed_progress_string_is_logged_not_raised(
    tmp_path: Path, monkeypatch
):
    # Arrange — pipeline emits one valid event + one garbage.
    _install_fake_config(monkeypatch)
    _install_fake_pipeline(
        monkeypatch,
        _make_stage_emitter_pipeline(["stage_1_start", "stage_7_done"]),
    )
    collected: list[SSEEvent] = []

    # Act — must not raise.
    run_pipeline_job(
        job_id="j1",
        input_path=tmp_path / "in.mp4",
        output_path=tmp_path / "out.mp4",
        source_lang="en",
        target_lang="es",
        emit=collected.append,
    )

    # Assert — only s1 StageStart is in the stream; no stray complete events.
    starts = [e for e in collected if isinstance(e, StageStartEvent)]
    completes = [e for e in collected if isinstance(e, StageCompleteEvent)]
    assert [e.stage for e in starts] == ["s1"]
    assert completes == []  # the garbage "stage_7_done" was swallowed


# ---------------------------------------------------------------------------
# _transcode_to_browser_safe — R3 mitigation
# ---------------------------------------------------------------------------
#
# OpenCV's ``VideoWriter`` with the ``mp4v`` fourcc on this box produces
# files tagged ``FMP4`` (FFmpeg-flavored MPEG-4 Part 2), which Chrome and
# Firefox refuse to play in ``<video>``. The integration smoke test in
# ``test_integration.py`` caught this and failed at the codec check.
#
# Mitigation per plan.md R3: shell out to ffmpeg after the pipeline writes
# the output, transcoding to H.264 (``libx264`` / ``avc1``) with
# ``+faststart`` so the moov atom lands at the head and the file streams.
# These tests verify the helper builds the right command and replaces the
# original file in place. We mock ``subprocess.run`` so the unit test never
# spawns ffmpeg — that path is exercised end-to-end by the integration
# smoke test instead.


def test_transcode_replaces_input_file_with_browser_safe_copy(
    tmp_path: Path, monkeypatch
):
    # Arrange — fake input MP4 + a stub subprocess.run that "produces" the
    # transcoded file by copying the input bytes to the output path.
    src = tmp_path / "out.mp4"
    src.write_bytes(b"FAKE_FMP4_BYTES")

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        # ffmpeg writes its output file as the last positional arg.
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"FAKE_AVC1_BYTES")

        class _CompletedStub:
            returncode = 0
            stderr = b""
            stdout = b""

        return _CompletedStub()

    import server.app.pipeline_runner as runner_mod
    monkeypatch.setattr(runner_mod.subprocess, "run", fake_run)

    # Act
    _transcode_to_browser_safe(src)

    # Assert — file replaced in place, original bytes gone.
    assert src.exists()
    assert src.read_bytes() == b"FAKE_AVC1_BYTES"
    # And the ffmpeg invocation looked right.
    assert captured["cmd"][0] == "ffmpeg"
    assert "-c:v" in captured["cmd"]
    assert "libx264" in captured["cmd"]
    assert "+faststart" in captured["cmd"]


def test_transcode_raises_runtime_error_when_ffmpeg_fails(
    tmp_path: Path, monkeypatch
):
    # Arrange — subprocess.run raises CalledProcessError as it would in real
    # ffmpeg failure (check=True path).
    import subprocess as _subprocess
    src = tmp_path / "out.mp4"
    src.write_bytes(b"FAKE_FMP4_BYTES")

    def fake_run(cmd, **kwargs):
        raise _subprocess.CalledProcessError(
            returncode=1, cmd=cmd, stderr=b"bad things"
        )

    import server.app.pipeline_runner as runner_mod
    monkeypatch.setattr(runner_mod.subprocess, "run", fake_run)

    # Act + Assert — surfaces as RuntimeError so JobManager can mark failed.
    with pytest.raises(RuntimeError, match="ffmpeg"):
        _transcode_to_browser_safe(src)


def test_transcode_cleans_tmp_on_unexpected_exception(
    tmp_path: Path, monkeypatch
):
    # Arrange — subprocess.run raises an exception NOT caught by either of
    # the named except branches (FileNotFoundError / CalledProcessError).
    # A PermissionError simulates a transient OS failure. The try/finally
    # must clean up the .browser.mp4 temp file regardless.
    src = tmp_path / "out.mp4"
    src.write_bytes(b"FAKE_FMP4_BYTES")
    original_bytes = src.read_bytes()

    def fake_run(cmd, **kwargs):
        # Create the tmp output (so ffmpeg "started" writing) and THEN fail
        # with a non-ffmpeg-specific error. Tests the finally cleanup path.
        tmp_out = Path(cmd[-1])
        tmp_out.write_bytes(b"partial")
        raise PermissionError("no")

    import server.app.pipeline_runner as runner_mod
    monkeypatch.setattr(runner_mod.subprocess, "run", fake_run)

    # Act + Assert — unexpected exception propagates (not wrapped).
    with pytest.raises(PermissionError):
        _transcode_to_browser_safe(src)

    # tmp_path must be cleaned up by the finally block.
    tmp_path_artifact = src.with_suffix(".browser.mp4")
    assert not tmp_path_artifact.exists(), (
        f"tmp file {tmp_path_artifact} survived an unexpected exception"
    )
    # Original file untouched — the atomic swap never happened.
    assert src.exists()
    assert src.read_bytes() == original_bytes


def test_run_pipeline_transcodes_output_after_pipeline_run(
    tmp_path: Path, monkeypatch
):
    """End-to-end runner test: transcode runs after the pipeline finishes."""
    # Arrange — fake pipeline that writes a "FMP4" file.
    _install_fake_config(monkeypatch)

    out_path = tmp_path / "out.mp4"

    class WriterPipeline:
        def __init__(self, config, progress_callback=None):
            self.cfg = config

        def run(self):
            Path(self.cfg.output_video).parent.mkdir(parents=True, exist_ok=True)
            Path(self.cfg.output_video).write_bytes(b"FAKE_FMP4_BYTES")

    _install_fake_pipeline(monkeypatch, WriterPipeline)

    transcode_calls: list[Path] = []

    def fake_transcode(path: Path) -> None:
        transcode_calls.append(path)
        path.write_bytes(b"FAKE_AVC1_BYTES")

    import server.app.pipeline_runner as runner_mod
    monkeypatch.setattr(
        runner_mod, "_transcode_to_browser_safe", fake_transcode
    )

    collected: list[SSEEvent] = []

    # Act
    run_pipeline_job(
        job_id="abc",
        input_path=tmp_path / "in.mp4",
        output_path=out_path,
        source_lang="en",
        target_lang="es",
        emit=collected.append,
    )

    # Assert — transcode invoked exactly once on the output path, AFTER
    # the pipeline wrote the file, and the DoneEvent followed.
    assert transcode_calls == [out_path]
    assert out_path.read_bytes() == b"FAKE_AVC1_BYTES"
    dones = [e for e in collected if isinstance(e, DoneEvent)]
    assert len(dones) == 1


def test_run_pipeline_skips_transcode_when_output_missing(
    tmp_path: Path, monkeypatch
):
    """No-output short-circuit: don't try to transcode a file that isn't there.

    A pipeline run that fails before writing any output (e.g. config
    validation error caught upstream) shouldn't trigger ffmpeg on a
    missing file. Defensive — the runner swallows this as a noop so the
    DoneEvent path isn't hit either (the run still emits StageEvents up
    to the failure point).
    """
    _install_fake_config(monkeypatch)
    _install_fake_pipeline(monkeypatch, _make_stage_emitter_pipeline([]))

    transcode_calls: list[Path] = []
    import server.app.pipeline_runner as runner_mod
    monkeypatch.setattr(
        runner_mod,
        "_transcode_to_browser_safe",
        lambda p: transcode_calls.append(p),
    )

    out_path = tmp_path / "never_written.mp4"
    # Don't create out_path — simulate the "pipeline didn't write output".

    collected: list[SSEEvent] = []
    run_pipeline_job(
        job_id="j1",
        input_path=tmp_path / "in.mp4",
        output_path=out_path,
        source_lang="en",
        target_lang="es",
        emit=collected.append,
    )

    # Transcode skipped, but DoneEvent still emitted (runner doesn't
    # validate output existence — that's the route's job).
    assert transcode_calls == []
    dones = [e for e in collected if isinstance(e, DoneEvent)]
    assert len(dones) == 1


# ---------------------------------------------------------------------------
# _LivenessWatchdog — plan.md Step 3, layer 2
#
# Purpose: a daemon thread that watches for emit() going silent and logs an
# ERROR record when the gap exceeds PIPELINE_LIVENESS_TIMEOUT_S. These tests
# use a 1s timeout (via monkeypatch.setenv) to keep runtime short. The
# _LivenessWatchdog clamps its poll interval to `timeout_s / 2` when the
# timeout is small, so a 1s timeout polls twice per second — short tests
# complete in 2-4 seconds each.
#
# The watchdog is log-only: no ErrorEvent, no status mutation, no cancel.
# These tests verify that invariant by asserting on the record type / level
# and checking that DoneEvent still fires on normal completion.
# ---------------------------------------------------------------------------


def test_read_liveness_timeout_default(monkeypatch):
    monkeypatch.delenv("PIPELINE_LIVENESS_TIMEOUT_S", raising=False)
    assert _read_liveness_timeout() == 180.0


def test_read_liveness_timeout_env_override(monkeypatch):
    monkeypatch.setenv("PIPELINE_LIVENESS_TIMEOUT_S", "42.5")
    assert _read_liveness_timeout() == 42.5


def test_read_liveness_timeout_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("PIPELINE_LIVENESS_TIMEOUT_S", "not-a-number")
    assert _read_liveness_timeout() == 180.0


def test_read_liveness_timeout_nonpositive_falls_back(monkeypatch):
    monkeypatch.setenv("PIPELINE_LIVENESS_TIMEOUT_S", "0")
    assert _read_liveness_timeout() == 180.0
    monkeypatch.setenv("PIPELINE_LIVENESS_TIMEOUT_S", "-5")
    assert _read_liveness_timeout() == 180.0


def test_watchdog_fires_after_silence():
    """Silence longer than timeout → ERROR log fires.

    Uses an isolated logger (not ``src``) so the assertion is tight and
    doesn't pick up noise from parallel tests. Timeout is 1s, we sleep 2s
    without calling notify().
    """
    test_logger = logging.getLogger(f"test.watchdog.{id(object())}")
    test_logger.setLevel(logging.ERROR)
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):  # noqa: A003
            records.append(record)

    h = _Capture()
    test_logger.addHandler(h)
    try:
        wd = _LivenessWatchdog(test_logger, timeout_s=1.0)
        wd.start()
        time.sleep(2.0)
        wd.stop()
    finally:
        test_logger.removeHandler(h)

    # Should have at least one ERROR record with the expected message shape.
    assert len(records) >= 1, "watchdog never fired after 2s silence"
    for r in records:
        assert r.levelno == logging.ERROR
        # Uses %-formatting; getMessage() resolves it.
        assert "no progress" in r.getMessage()


def test_watchdog_resets_on_notify():
    """Periodic notify() keeps the clock reset → no fires.

    Calls notify() every 0.3s for 2s with a 1s timeout. The gap never
    reaches 1s, so the watchdog never fires.
    """
    test_logger = logging.getLogger(f"test.watchdog.{id(object())}")
    test_logger.setLevel(logging.ERROR)
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):  # noqa: A003
            records.append(record)

    h = _Capture()
    test_logger.addHandler(h)
    try:
        wd = _LivenessWatchdog(test_logger, timeout_s=1.0)
        wd.start()
        # Hit notify() 6 times at 0.3s intervals → 1.8s of activity.
        for _ in range(6):
            time.sleep(0.3)
            wd.notify()
        wd.stop()
    finally:
        test_logger.removeHandler(h)

    assert records == [], (
        f"watchdog fired despite periodic notify(): {[r.getMessage() for r in records]}"
    )


def test_watchdog_suppresses_repeat_fires_within_window():
    """3 timeout windows of silence → ~3 fires, not continuous.

    With timeout_s=1.0 and 3s of silence, the watchdog's internal window
    counter should produce exactly 3 fires (one per crossing of the
    1s, 2s, 3s boundary). The poll interval is clamped to timeout/2=0.5s,
    so a naive "fire on every tick" implementation would produce ~6 fires.
    The test asserts a tight upper bound (<= 4) to catch regressions.
    """
    test_logger = logging.getLogger(f"test.watchdog.{id(object())}")
    test_logger.setLevel(logging.ERROR)
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):  # noqa: A003
            records.append(record)

    h = _Capture()
    test_logger.addHandler(h)
    try:
        wd = _LivenessWatchdog(test_logger, timeout_s=1.0)
        wd.start()
        time.sleep(3.2)  # a hair past 3s to ensure the 3rd window tripped
        wd.stop()
    finally:
        test_logger.removeHandler(h)

    # At least 2 fires (windows 1 and 2), at most 4 (windows 1-3 + some
    # slack for scheduler jitter). Zero would mean the watchdog never ran
    # and 10+ would mean we're spamming every poll tick — both regressions.
    assert 2 <= len(records) <= 4, (
        f"expected ~3 fires for 3s silence with 1s timeout, got {len(records)}: "
        f"{[r.getMessage() for r in records]}"
    )


def test_watchdog_stops_cleanly_on_success(tmp_path: Path, monkeypatch):
    """Normal run → watchdog thread is joined, no orphan daemon thread."""
    monkeypatch.setenv("PIPELINE_LIVENESS_TIMEOUT_S", "1")
    _install_fake_config(monkeypatch)
    _install_fake_pipeline(monkeypatch, _make_stage_emitter_pipeline([]))

    before = {t.name for t in threading.enumerate()}

    run_pipeline_job(
        job_id="j1",
        input_path=tmp_path / "in.mp4",
        output_path=tmp_path / "out.mp4",
        source_lang="en",
        target_lang="es",
        emit=lambda e: None,
    )

    # Give the join a moment to settle even though stop() should have
    # already joined synchronously. Any orphan is a real leak.
    time.sleep(0.1)
    after = {t.name for t in threading.enumerate()}
    orphans = {
        n for n in (after - before) if n == "pipeline-liveness-watchdog"
    }
    assert orphans == set(), f"watchdog thread leaked after success: {orphans}"


def test_watchdog_stops_cleanly_on_exception(tmp_path: Path, monkeypatch):
    """Pipeline raises → watchdog is still stopped in the finally block."""
    monkeypatch.setenv("PIPELINE_LIVENESS_TIMEOUT_S", "1")
    _install_fake_config(monkeypatch)

    class CrashingPipeline:
        def __init__(self, config, progress_callback=None):
            pass

        def run(self):
            raise RuntimeError("boom")

    _install_fake_pipeline(monkeypatch, CrashingPipeline)
    before = {t.name for t in threading.enumerate()}

    with pytest.raises(RuntimeError, match="boom"):
        run_pipeline_job(
            job_id="j1",
            input_path=tmp_path / "in.mp4",
            output_path=tmp_path / "out.mp4",
            source_lang="en",
            target_lang="es",
            emit=lambda e: None,
        )

    time.sleep(0.1)
    after = {t.name for t in threading.enumerate()}
    orphans = {
        n for n in (after - before) if n == "pipeline-liveness-watchdog"
    }
    assert orphans == set(), (
        f"watchdog thread leaked after exception: {orphans}"
    )


def test_watchdog_fires_during_run_when_pipeline_hangs(
    tmp_path: Path, monkeypatch,
):
    """End-to-end: a slow pipeline run surfaces a watchdog ERROR LogEvent.

    This is the integration test that proves the whole chain is wired:
    watchdog fires → src_logger.error → _PipelineLogHandler → LogEvent →
    forwarded through emit → captured by the test sink.
    """
    monkeypatch.setenv("PIPELINE_LIVENESS_TIMEOUT_S", "1")
    _install_fake_config(monkeypatch)

    class HangingPipeline:
        def __init__(self, config, progress_callback=None):
            pass

        def run(self):
            # Silent sleep longer than the timeout — no progress_callback,
            # no log records from our side.
            time.sleep(2.0)

    _install_fake_pipeline(monkeypatch, HangingPipeline)
    collected: list[SSEEvent] = []

    run_pipeline_job(
        job_id="j1",
        input_path=tmp_path / "in.mp4",
        output_path=tmp_path / "out.mp4",
        source_lang="en",
        target_lang="es",
        emit=collected.append,
    )

    # At least one LogEvent with level=error and the watchdog message.
    log_events = [e for e in collected if isinstance(e, LogEvent)]
    watchdog_errors = [
        e for e in log_events
        if e.level == "error" and "no progress" in e.message
    ]
    assert len(watchdog_errors) >= 1, (
        f"watchdog never surfaced via SSE; got log events: "
        f"{[(e.level, e.message) for e in log_events]}"
    )
    # And the run still completed normally — DoneEvent fires because the
    # watchdog is log-only, not a cancel.
    assert any(isinstance(e, DoneEvent) for e in collected), (
        "DoneEvent missing — watchdog should be log-only, not a cancel"
    )
    # No ErrorEvent was synthesized by the runner (that's JobManager's job).
    assert not any(isinstance(e, ErrorEvent) for e in collected)
