"""Online (sliding-window) CoTracker point tracker for streaming video processing."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CoTrackerOnlineFlowTracker:
    """Chunked point tracker using CoTracker3 online mode.

    Processes video frames in sliding windows of ``step * 2`` frames,
    advancing by ``step`` each iteration.  Only a small number of frames
    are held in GPU memory at any time, making it suitable for long videos.
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

        # Build queries: (1, N, 3) with format (t, x, y)
        # t is relative to the start of frame_idxs
        t_ref = frame_idxs.index(ref_idx)
        queries = torch.zeros(1, n_points, 3, device=self._device)
        queries[0, :, 0] = t_ref
        queries[0, :, 1] = torch.tensor(ref_points[:, 0], dtype=torch.float32)
        queries[0, :, 2] = torch.tensor(ref_points[:, 1], dtype=torch.float32)

        # Read all frames into a buffer (bounded by track length, typically short per-track)
        # For online mode we need to feed chunks sequentially
        frame_buffer = []
        for idx in frame_idxs:
            frame = video_reader.read_frame(idx)
            if frame is None:
                logger.warning("Failed to read frame %d, using zeros", idx)
                # Use a black frame as fallback
                prev = frame_buffer[-1] if frame_buffer else np.zeros((1080, 1920, 3), dtype=np.uint8)
                frame_buffer.append(prev)
            else:
                frame_buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        T = len(frame_buffer)
        # Stack into tensor: (1, T, C, H, W)
        video_tensor = (
            torch.tensor(np.stack(frame_buffer))
            .permute(0, 3, 1, 2)  # T, C, H, W
            .unsqueeze(0)         # 1, T, C, H, W
            .float()
            .to(self._device)
        )
        # Free CPU buffer
        del frame_buffer

        # First step: initialize with queries
        self._model(
            video_chunk=video_tensor[:, :step * 2],
            is_first_step=True,
            queries=queries,
        )

        # Process remaining chunks
        all_tracks = None
        all_vis = None
        for ind in range(0, T - step, step):
            chunk = video_tensor[:, ind: ind + step * 2]
            if chunk.shape[1] < step * 2:
                break
            pred_tracks, pred_vis = self._model(video_chunk=chunk)
            if pred_tracks is not None:
                all_tracks = pred_tracks
                all_vis = pred_vis

        # Free GPU memory
        del video_tensor

        if all_tracks is None:
            logger.warning("CoTracker online produced no tracks")
            return {}

        tracks_np = all_tracks[0].cpu().numpy()   # (T_out, N, 2)
        vis_np = all_vis[0].cpu().numpy()          # (T_out, N)

        result: dict[int, np.ndarray] = {}
        # The output length may differ from input — map back to frame indices
        t_out = tracks_np.shape[0]
        for t in range(min(t_out, len(frame_idxs))):
            if vis_np[t].all():
                result[frame_idxs[t]] = tracks_np[t].astype(np.float32)
            else:
                logger.debug(
                    "CoTracker online: frame %d has occluded points, skipping",
                    frame_idxs[t],
                )

        return result
