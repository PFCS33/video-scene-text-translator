"""Video I/O utilities for reading and writing video files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoReader:
    """Lazy frame-by-frame video reader with context manager support."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.path}")

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def frame_size(self) -> tuple[int, int]:
        """Returns (width, height)."""
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def read_frame(self, idx: int) -> np.ndarray | None:
        """Read a specific frame by index. Returns None if read fails."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        return frame if ret else None

    def iter_frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (frame_idx, frame_bgr) for all frames."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield idx, frame
            idx += 1

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *args) -> None:
        self.close()


class VideoWriter:
    """Frame-by-frame video writer with context manager support."""

    def __init__(
        self,
        path: str | Path,
        fps: float,
        frame_size: tuple[int, int],
        codec: str = "mp4v",
    ):
        self.path = str(path)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(self.path, fourcc, fps, frame_size)
        if not self._writer.isOpened():
            raise IOError(f"Cannot create video writer: {self.path}")

    def write_frame(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *args) -> None:
        self.close()
