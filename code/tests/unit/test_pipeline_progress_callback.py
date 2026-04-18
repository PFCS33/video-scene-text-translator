"""Tests for VideoPipeline.progress_callback hook (D11).

The pipeline emits stage transition events to an optional callback so
external code (e.g. the web server) can drive a progress UI without
parsing log messages.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_frames(n: int = 2) -> list[tuple[int, np.ndarray]]:
    """Return n dummy BGR frames indexed from 0."""
    return [(i, np.zeros((8, 8, 3), dtype=np.uint8)) for i in range(n)]


def _make_reader_mock(frames: list[tuple[int, np.ndarray]]):
    """Return a MagicMock that behaves like VideoReader when used as a
    context manager."""
    reader = MagicMock()
    reader.fps = 30.0
    reader.frame_size = (8, 8)
    reader.iter_frames.return_value = iter(frames)
    cm = MagicMock()
    cm.__enter__.return_value = reader
    cm.__exit__.return_value = False
    return cm


def _make_writer_mock():
    """Return a MagicMock that behaves like VideoWriter when used as a
    context manager."""
    writer = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = writer
    cm.__exit__.return_value = False
    return cm


def _stub_stages(pipeline, *, s1_returns):
    """Replace each stage's .run() with a MagicMock returning the minimum
    shape the next stage expects.

    - S1 -> list[TextTrack] (or empty list to short-circuit)
    - S2 -> list[TextTrack]
    - S3 -> list[TextTrack]
    - S4 -> dict[int, Any]  (propagated ROIs keyed by frame_idx)
    - S5 -> list[np.ndarray] (output frames)
    """
    sentinel_tracks = s1_returns
    pipeline.s1.run = MagicMock(return_value=sentinel_tracks)
    pipeline.s2.run = MagicMock(return_value=sentinel_tracks)
    pipeline.s3.run = MagicMock(return_value=sentinel_tracks)
    pipeline.s4.run = MagicMock(return_value={})
    pipeline.s5.run = MagicMock(
        return_value=[np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProgressCallback:
    def test_emits_all_10_events_in_order_on_happy_path(
        self, default_config, sample_track
    ):
        """Happy path: all 5 stages run → exactly 10 events in order."""
        from src.pipeline import VideoPipeline

        events: list[str] = []
        with patch("src.pipeline.VideoReader") as vr, \
             patch("src.pipeline.VideoWriter") as vw:
            vr.return_value = _make_reader_mock(_fake_frames(2))
            vw.return_value = _make_writer_mock()

            pipeline = VideoPipeline(
                default_config, progress_callback=events.append
            )
            _stub_stages(pipeline, s1_returns=[sample_track])

            pipeline.run()

        assert events == [
            "stage_1_start",
            "stage_1_done",
            "stage_2_start",
            "stage_2_done",
            "stage_3_start",
            "stage_3_done",
            "stage_4_start",
            "stage_4_done",
            "stage_5_start",
            "stage_5_done",
        ]

    def test_emits_only_stage_1_events_when_no_tracks_found(
        self, default_config
    ):
        """Short-circuit path: S1 returns [] → only S1 events emitted."""
        from src.pipeline import VideoPipeline

        events: list[str] = []
        with patch("src.pipeline.VideoReader") as vr, \
             patch("src.pipeline.VideoWriter") as vw:
            vr.return_value = _make_reader_mock(_fake_frames(2))
            vw.return_value = _make_writer_mock()

            pipeline = VideoPipeline(
                default_config, progress_callback=events.append
            )
            _stub_stages(pipeline, s1_returns=[])

            pipeline.run()

        assert events == ["stage_1_start", "stage_1_done"]

    def test_runs_without_callback_when_omitted(
        self, default_config, sample_track
    ):
        """Default (no callback) must not raise and must still execute."""
        from src.pipeline import VideoPipeline

        with patch("src.pipeline.VideoReader") as vr, \
             patch("src.pipeline.VideoWriter") as vw:
            vr.return_value = _make_reader_mock(_fake_frames(2))
            vw.return_value = _make_writer_mock()

            pipeline = VideoPipeline(default_config)  # no progress_callback
            _stub_stages(pipeline, s1_returns=[sample_track])

            # Should not raise.
            pipeline.run()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
