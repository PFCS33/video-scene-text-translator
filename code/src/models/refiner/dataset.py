"""Dataset for ROI alignment refiner training.

Data root: /workspace/tpm_dataset/ (shared with BPN training).
Structure: {video_name}/{track_name}/frame_NNNNNN.png

Produces two sample types, mixed per batch via the ``real_pair_fraction``
attribute (mutable — the training loop ramps it across epochs):

    Type A (synthetic self-pairs, supervised):
        One frame -> source; apply random +/- N px 4-corner perturbation -> target.
        Ground truth = the known corner offsets.

    Type B (real in-track pairs, self-supervised):
        Two random distinct frames from the same track, random source/target
        ordering. No ground truth; trained with illumination-robust losses.

Split by video name (not by track). See plan.md §1.1 for the full design.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from .warp import canonical_corners, corners_to_H, warp_image

logger = logging.getLogger(__name__)


class RefinerDataset(Dataset):
    """Dataset yielding (source, target, optional GT corners) triples.

    Notes:
        * ``real_pair_fraction`` is mutable — training code sets it per epoch
          to implement the ramp schedule from plan.md §1.6.
        * Uses ``random.random`` for type/frame selection. DataLoader workers
          inherit the parent RNG state on fork, so the training script must
          install a ``worker_init_fn`` that re-seeds per worker if multiple
          workers are used, otherwise sample draws will be duplicated.
    """

    def __init__(
        self,
        data_root: str | Path,
        video_names: list[str],
        image_size: tuple[int, int] = (64, 128),
        pairs_per_track: int = 64,
        real_pair_fraction: float = 0.0,
        syn_perturbation_px: float = 8.0,
        photometric_strength: float = 1.0,
        min_track_length: int = 2,
        seed: int = 42,
        cache_in_ram: bool = False,
    ):
        """
        Args:
            data_root: Path to tpm_dataset root.
            video_names: List of video folder names to include (split control).
            image_size: (H, W) network input size. All ROIs resized to this.
            pairs_per_track: Upper bound on samples contributed by one track
                per epoch. Dataset length = sum over tracks of
                ``min(track_length, pairs_per_track)``.
            real_pair_fraction: Probability in [0, 1] of drawing a Type B
                (real) sample. Mutable — the training loop updates this
                attribute per epoch according to the ramp schedule.
            syn_perturbation_px: Uniform ±range for Type A corner offsets.
            photometric_strength: Global multiplier on augmentation ranges.
                Type B uses half of this by convention (real pairs already
                have real appearance variance).
            min_track_length: Skip tracks with fewer frames than this.
            seed: Used for deterministic discovery ordering only, not for
                sample-time randomness.
            cache_in_ram: If True, preload all frames into a single contiguous
                uint8 ndarray (BPN-style, fork-safe).
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.image_size = image_size  # (H, W)
        self.pairs_per_track = int(pairs_per_track)
        self.real_pair_fraction = float(real_pair_fraction)
        self.syn_perturbation_px = float(syn_perturbation_px)
        self.photometric_strength = float(photometric_strength)
        self.min_track_length = int(min_track_length)
        self.cache_in_ram = cache_in_ram

        self.track_paths: list[list[str]] = []  # track_idx -> frame paths
        self.track_names: list[str] = []        # track_idx -> "video/track"
        self.samples: list[int] = []            # sample_idx -> track_idx

        self._build_tracks(video_names, seed)

        self._image_cache: np.ndarray | None = None
        self._cache_indices: list[list[int]] | None = None
        if cache_in_ram:
            self._preload_cache()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_tracks(self, video_names: list[str], seed: int) -> None:
        # seed kept for future deterministic track subsetting; unused today
        del seed
        skipped = 0
        for video_name in video_names:
            vdir = self.data_root / video_name
            if not vdir.is_dir():
                logger.warning("Video folder not found: %s", vdir)
                continue
            track_dirs = sorted(
                d for d in vdir.iterdir()
                if d.is_dir() and d.name.startswith("track_")
            )
            for tdir in track_dirs:
                frames = sorted(
                    str(f) for f in tdir.iterdir()
                    if f.suffix.lower() in (".png", ".jpg", ".jpeg")
                )
                if len(frames) < self.min_track_length:
                    skipped += 1
                    continue
                track_idx = len(self.track_paths)
                self.track_paths.append(frames)
                self.track_names.append(f"{video_name}/{tdir.name}")
                n_samples = min(len(frames), self.pairs_per_track)
                self.samples.extend([track_idx] * n_samples)

        logger.info(
            "RefinerDataset: %d videos, %d tracks kept, %d samples "
            "(skipped %d short tracks)",
            len(video_names), len(self.track_paths), len(self.samples), skipped,
        )

    def _preload_cache(self) -> None:
        H, W = self.image_size
        unique_paths = sorted({p for frames in self.track_paths for p in frames})
        path_to_idx = {p: i for i, p in enumerate(unique_paths)}

        logger.info(
            "Caching %d unique frames (decoded %dx%d uint8) into RAM",
            len(unique_paths), H, W,
        )
        self._image_cache = np.empty((len(unique_paths), H, W, 3), dtype=np.uint8)
        for i, p in enumerate(tqdm(unique_paths, desc="preload", unit="img")):
            img = Image.open(p).convert("RGB").resize((W, H), Image.BILINEAR)
            self._image_cache[i] = np.array(img, dtype=np.uint8)
        self._cache_indices = [
            [path_to_idx[p] for p in frames] for frames in self.track_paths
        ]
        total_gb = self._image_cache.nbytes / 1e9
        logger.info("Cache complete: %.2f GB", total_gb)

    # ------------------------------------------------------------------
    # Frame loading
    # ------------------------------------------------------------------

    def _load_frame(self, track_idx: int, frame_idx: int) -> torch.Tensor:
        """Return one frame as a ``(3, H, W)`` float32 tensor in [0, 1]."""
        H, W = self.image_size
        if self.cache_in_ram:
            assert self._image_cache is not None and self._cache_indices is not None
            cache_idx = self._cache_indices[track_idx][frame_idx]
            arr = self._image_cache[cache_idx].copy()
        else:
            path = self.track_paths[track_idx][frame_idx]
            img = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
            arr = np.array(img, dtype=np.uint8)
        return torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        track_idx = self.samples[idx]
        n_frames = len(self.track_paths[track_idx])

        if n_frames < 2 or random.random() >= self.real_pair_fraction:
            return self._make_synthetic_sample(track_idx, n_frames)
        return self._make_real_sample(track_idx, n_frames)

    def _make_synthetic_sample(self, track_idx: int, n_frames: int) -> dict[str, Any]:
        H, W = self.image_size
        frame_i = random.randrange(n_frames)
        source = self._load_frame(track_idx, frame_i)  # (3, H, W)

        # Random 4-corner perturbation in ±syn_perturbation_px pixels.
        delta = (torch.rand(4, 2) * 2 - 1) * self.syn_perturbation_px
        src_corners = canonical_corners(H, W)
        dst_corners = src_corners + delta
        H_gt = corners_to_H(src_corners.unsqueeze(0), dst_corners.unsqueeze(0))[0]

        # target = warp(source, H_gt). Border triangles that sample outside
        # the source are filled with zero (padding_mode="zeros"); the loss's
        # validity mask handles these at training time.
        target = warp_image(source.unsqueeze(0), H_gt.unsqueeze(0), (H, W))[0]

        source = self._augment(source, self.photometric_strength)
        target = self._augment(target, self.photometric_strength)

        return {
            "source": source,
            "target": target,
            "delta_corners_gt": delta,
            "has_gt_corners": torch.tensor(True),
            "sample_type": "syn",
        }

    def _make_real_sample(self, track_idx: int, n_frames: int) -> dict[str, Any]:
        i, j = random.sample(range(n_frames), 2)
        if random.random() < 0.5:
            i, j = j, i
        source = self._load_frame(track_idx, i)
        target = self._load_frame(track_idx, j)

        strength = self.photometric_strength * 0.5
        source = self._augment(source, strength)
        target = self._augment(target, strength)

        return {
            "source": source,
            "target": target,
            "delta_corners_gt": torch.zeros(4, 2),
            "has_gt_corners": torch.tensor(False),
            "sample_type": "real",
        }

    # ------------------------------------------------------------------
    # Photometric augmentation
    # ------------------------------------------------------------------

    def _augment(self, img: torch.Tensor, strength: float) -> torch.Tensor:
        """Independent photometric augmentation on a single image tensor.

        Applied to source and target independently so the network cannot rely
        on raw RGB equality. See plan.md §1.3 for rationale.
        """
        if strength <= 0:
            return img.clamp(0, 1)

        # Brightness: ± (0.2 * strength)
        img = img + (torch.rand(1).item() - 0.5) * 0.4 * strength

        # Contrast: scale around per-image mean by [1 - 0.2s, 1 + 0.2s]
        scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.4 * strength
        mean = img.mean()
        img = (img - mean) * scale + mean
        img = img.clamp(0, 1)

        # Gamma: [0.65, 1.35] at strength=1
        gamma = 1.0 + (torch.rand(1).item() - 0.5) * 0.7 * strength
        img = img.clamp_min(1e-6).pow(gamma).clamp(0, 1)

        # Gaussian noise with 50% probability
        if random.random() < 0.5:
            img = (img + torch.randn_like(img) * 0.02 * strength).clamp(0, 1)

        # Gaussian blur with 50% probability
        if random.random() < 0.5:
            sigma = random.uniform(0, 1.5) * strength
            if sigma > 0.1:
                img = self._gaussian_blur(img, sigma)

        return img.clamp(0, 1)

    @staticmethod
    def _gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
        """Separable Gaussian blur on a ``(C, H, W)`` tensor. No torchvision dep."""
        ksize = max(3, int(2 * round(3 * sigma) + 1))
        if ksize % 2 == 0:
            ksize += 1
        x = torch.arange(ksize, dtype=img.dtype, device=img.device) - (ksize - 1) / 2
        k = torch.exp(-x.pow(2) / (2 * sigma * sigma))
        k = k / k.sum()

        C = img.shape[0]
        k_h = k.view(1, 1, 1, ksize).expand(C, 1, 1, ksize)
        k_v = k.view(1, 1, ksize, 1).expand(C, 1, ksize, 1)

        img = img.unsqueeze(0)
        img = F.pad(img, (ksize // 2, ksize // 2, 0, 0), mode="reflect")
        img = F.conv2d(img, k_h, groups=C)
        img = F.pad(img, (0, 0, ksize // 2, ksize // 2), mode="reflect")
        img = F.conv2d(img, k_v, groups=C)
        return img.squeeze(0)


# ----------------------------------------------------------------------
# Sanity visualization — invoked via ``python -m src.models.refiner.dataset``
# ----------------------------------------------------------------------

def visualize_samples(
    dataset: RefinerDataset,
    out_path: str | Path,
    n_samples: int = 16,
) -> None:
    """Dump a vertical grid of ``[source | target | abs_diff]`` rows to PNG.

    Useful for spot-checking that synthetic and real pairs look right
    before burning GPU time on training.
    """
    rows: list[np.ndarray] = []
    labels: list[str] = []
    # Sample indices are grouped by track in self.samples, so `range(n)` only
    # hits one or two tracks. Evenly spread the indices to see diverse tracks.
    n = min(n_samples, len(dataset))
    if n == 0:
        return
    stride = max(1, len(dataset) // n)
    idxs = [(i * stride) % len(dataset) for i in range(n)]
    for i in idxs:
        sample = dataset[i]
        S = sample["source"].permute(1, 2, 0).numpy()
        T = sample["target"].permute(1, 2, 0).numpy()
        diff = np.abs(S - T)
        row = np.concatenate([S, T, diff], axis=1)  # (H, 3W, 3)
        rows.append(row)
        labels.append(sample["sample_type"])

    grid = np.concatenate(rows, axis=0)
    grid_u8 = (grid.clip(0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(grid_u8)
    img.save(out_path)
    logger.info("Wrote %d sample rows (%s) to %s", len(rows), ",".join(labels), out_path)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    p = argparse.ArgumentParser(description="Refiner dataset sanity visualizer")
    p.add_argument("--data-root", default="/workspace/tpm_dataset")
    p.add_argument("--videos", nargs="+", required=True,
                   help="video folder names under data-root")
    p.add_argument("--out", default="refiner_samples.png")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--real-fraction", type=float, default=0.5)
    p.add_argument("--image-size", nargs=2, type=int, default=[64, 128],
                   metavar=("H", "W"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds = RefinerDataset(
        data_root=args.data_root,
        video_names=args.videos,
        image_size=tuple(args.image_size),
        real_pair_fraction=args.real_fraction,
    )
    print(f"Dataset: {len(ds)} samples across {len(ds.track_paths)} tracks")
    visualize_samples(ds, args.out, n_samples=args.n)
    print(f"Wrote {args.out}")
