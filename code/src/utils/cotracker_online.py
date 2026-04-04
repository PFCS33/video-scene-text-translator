"""Online (sliding-window) CoTracker point tracker for streaming video processing."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CoTrackerOnlineFlowTracker:
    """Chunked point tracker using CoTracker3 online mode.

    Streams frames forward from the start of the track via VideoReader.
    The reference frame must be within the first sliding window
    (``window_len`` frames) so the model sees the query points' appearance
    immediately.  Only ~2*step frames are held in GPU memory at any time.
    """

    def __init__(self, config):
        self.config = config
        self._model = None
        self._device = None

    def _init_model(self):
        if self._model is not None:
            return
        import torch
        from cotracker.predictor import CoTrackerOnlinePredictor

        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self._model = CoTrackerOnlinePredictor(
            checkpoint=self.config.cotracker_online_checkpoint,
            v2=False,
            window_len=self.config.cotracker_online_window_len,
        ).to(device)
        self._device = device
        logger.info(
            "CoTracker3 online initialized on %s (step=%d)",
            device, self._model.step,
        )

    @property
    def step(self) -> int:
        """Number of new frames consumed per online iteration."""
        self._init_model()
        return self._model.step

    def track_points_online(
        self,
        video_reader,
        frame_idxs: list[int],
        ref_idx: int,
        ref_points: np.ndarray,
    ) -> dict[int, np.ndarray]:
        """Track points from a reference frame across ``frame_idxs`` using online mode.

        Streams frames forward from the start of ``frame_idxs``.  The
        reference frame must be near the start (within ``window_len``
        frames) so the model sees the query appearance in the first window.

        Args:
            video_reader: ``VideoReader`` instance (supports ``read_frame``).
            frame_idxs: sorted list of frame indices to track through.
            ref_idx: index of the reference frame (must be in ``frame_idxs``).
            ref_points: (N, 2) query points in (x, y) pixel coords.

        Returns:
            dict mapping frame_idx -> (N, 2) tracked points.
        """
        import torch

        self._init_model()

        step = self._model.step
        n_points = ref_points.shape[0]
        T = len(frame_idxs)
        t_ref = frame_idxs.index(ref_idx)

        # Build queries: (1, N, 3) with format (t, x, y)
        queries = torch.zeros(1, n_points, 3, device=self._device)
        queries[0, :, 0] = t_ref
        queries[0, :, 1] = torch.tensor(ref_points[:, 0], dtype=torch.float32)
        queries[0, :, 2] = torch.tensor(ref_points[:, 1], dtype=torch.float32)

        # Stream frames and feed in overlapping windows of 2*step
        window_buf: list[torch.Tensor] = []  # each: (C, H, W)
        all_tracks = None
        is_first_step = True
        frames_fed = 0

        for idx in frame_idxs:
            frame = video_reader.read_frame(idx)
            if frame is None:
                logger.warning("Failed to read frame %d, reusing previous", idx)
                if window_buf:
                    frame_t = window_buf[-1]
                else:
                    frame_t = torch.zeros(3, 1080, 1920, device=self._device)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_t = (
                    torch.tensor(rgb, device=self._device)
                    .permute(2, 0, 1)
                    .float()
                )

            window_buf.append(frame_t)
            frames_fed += 1

            # Trigger processing every `step` frames, once we have >= 2*step
            if frames_fed % step == 0 and frames_fed >= step * 2:
                chunk_frames = window_buf[-(step * 2):]
                chunk = torch.stack(chunk_frames).unsqueeze(0)  # (1, 2*step, C, H, W)

                pred_tracks, _ = self._model(
                    video_chunk=chunk,
                    is_first_step=is_first_step,
                    queries=queries if is_first_step else None,
                )
                if is_first_step:
                    is_first_step = False
                    # Init returns (None, None); process same window again
                    pred_tracks, _ = self._model(video_chunk=chunk)
                if pred_tracks is not None:
                    all_tracks = pred_tracks

                # Trim buffer: keep last `step` frames for overlap
                window_buf = window_buf[-step:]

        # Handle remaining frames that didn't fill a complete step
        if frames_fed % step != 0 and frames_fed >= step * 2:
            chunk_frames = window_buf[-(step * 2):]
            if len(chunk_frames) < step * 2:
                pad_size = step * 2 - len(chunk_frames)
                chunk_frames = chunk_frames + [chunk_frames[-1]] * pad_size
            chunk = torch.stack(chunk_frames).unsqueeze(0)
            pred_tracks, _ = self._model(
                video_chunk=chunk,
                is_first_step=is_first_step,
                queries=queries if is_first_step else None,
            )
            if is_first_step:
                is_first_step = False
                pred_tracks, _ = self._model(video_chunk=chunk)
            if pred_tracks is not None:
                all_tracks = pred_tracks

        if all_tracks is None:
            logger.warning("CoTracker online produced no tracks")
            return {}

        tracks_np = all_tracks[0].cpu().numpy()   # (T_out, N, 2)
        t_out = tracks_np.shape[0]

        result: dict[int, np.ndarray] = {}
        for t in range(min(t_out, T)):
            result[frame_idxs[t]] = tracks_np[t].astype(np.float32)

        return result
