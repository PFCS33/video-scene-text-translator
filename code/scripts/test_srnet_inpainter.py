"""Smoke-test the SRNet background inpainter on extracted ROIs.

Picks one frame from each track folder under test_output/roi_extraction_2,
runs the inpainter, and writes a side-by-side (original | inpainted) PNG
into test_output/srnet_inpaint_vis/.

Usage:
    cd code && python scripts/test_srnet_inpainter.py \
        --checkpoint ../third_party/SRNet/checkpoints/trained_final_5M_.model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Repo layout: code/scripts/test_srnet_inpainter.py
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from src.stages.s4_propagation.srnet_inpainter import SRNetInpainter  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default=str(REPO_ROOT / "third_party/SRNet/checkpoints/trained_final_5M_.model"))
    parser.add_argument("--roi-root", type=str,
                        default=str(REPO_ROOT / "test_output/roi_extraction_2"))
    parser.add_argument("--out-dir", type=str,
                        default=str(REPO_ROOT / "test_output/srnet_inpaint_vis"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-tracks", type=int, default=20,
                        help="Number of tracks to sample (one frame each)")
    parser.add_argument("--frames-per-track", type=int, default=3,
                        help="Number of frames to sample per track")
    args = parser.parse_args()

    roi_root = Path(args.roi_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SRNet from {args.checkpoint} on {args.device}...")
    inpainter = SRNetInpainter(checkpoint_path=args.checkpoint, device=args.device)
    print("Loaded.")

    track_dirs = sorted(d for d in roi_root.iterdir()
                        if d.is_dir() and d.name.startswith("track_"))
    track_dirs = track_dirs[:args.max_tracks]
    print(f"Processing {len(track_dirs)} tracks")

    saved = 0
    for tdir in track_dirs:
        frames = sorted(tdir.glob("frame_*.png"))
        if not frames:
            continue

        # Pick a few evenly-spaced frames per track
        n = min(args.frames_per_track, len(frames))
        idxs = [int(round(i * (len(frames) - 1) / max(1, n - 1))) for i in range(n)]

        for j, idx in enumerate(idxs):
            roi_path = frames[idx]
            roi = cv2.imread(str(roi_path), cv2.IMREAD_COLOR)
            if roi is None:
                continue

            inpainted = inpainter.inpaint(roi)
            assert inpainted.shape == roi.shape, \
                f"Shape mismatch: {inpainted.shape} vs {roi.shape}"

            # Side-by-side: original | inpainted | abs-diff (3x amplified)
            diff = np.abs(roi.astype(int) - inpainted.astype(int))
            diff = (diff * 3).clip(0, 255).astype(np.uint8)
            vis = np.concatenate([roi, inpainted, diff], axis=1)

            # Upscale for legibility (ROIs are tiny)
            scale = max(1, 256 // roi.shape[0])
            vis = cv2.resize(vis, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_NEAREST)

            out_path = out_dir / f"{tdir.name}__frame{idx:06d}.png"
            cv2.imwrite(str(out_path), vis)
            saved += 1

    print(f"Saved {saved} visualizations to {out_dir}")


if __name__ == "__main__":
    main()
