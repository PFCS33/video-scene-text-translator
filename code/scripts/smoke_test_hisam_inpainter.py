"""Smoke-test the Hi-SAM segmentation-based background inpainter.

Runs Hi-SAM stroke segmentation + cv2.inpaint (Navier-Stokes by default) on a
set of extracted ROIs, and writes a side-by-side
(original | mask | inpainted | 3x-amplified-diff) PNG per ROI into
test_output/hisam_inpaint_vis/.

Parallels scripts/test_srnet_inpainter.py. Not a pytest — run it by hand when
you want to eyeball Hi-SAM's quality on real canonical ROIs, tune
--mask-dilation-px, or compare "ns" vs "telea".

Usage (from repo root):
    ./.venv/bin/python code/scripts/smoke_test_hisam_inpainter.py \\
        --checkpoint third_party/Hi-SAM/pretrained_checkpoint/sam_tss_l_textseg.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from src.stages.s4_propagation.segmentation_inpainter import (  # noqa: E402
    SegmentationBasedInpainter,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            REPO_ROOT
            / "third_party/Hi-SAM/pretrained_checkpoint/sam_tss_l_textseg.pth"
        ),
    )
    parser.add_argument("--model-type", type=str, default="vit_l",
                        choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--inpaint-method", type=str, default="ns",
                        choices=["ns", "telea"])
    parser.add_argument("--mask-dilation-px", type=int, default=3)
    parser.add_argument("--use-patch-mode", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--roi-root",
        type=str,
        default=str(REPO_ROOT / "test_output/roi_extraction_2"),
        help="Directory containing track_*/frame_*.png ROIs, per the TPM data "
             "gen pipeline layout.",
    )
    parser.add_argument(
        "--roi-path",
        type=str,
        default=None,
        help="Single image to test instead of --roi-root. Overrides roi-root.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO_ROOT / "test_output/hisam_inpaint_vis"),
    )
    parser.add_argument("--max-tracks", type=int, default=20)
    parser.add_argument("--frames-per-track", type=int, default=3)
    return parser.parse_args()


def _mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    """Render a uint8 {0, 255} mask as a BGR visualization (white text on black)."""
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def _make_vis(roi: np.ndarray, mask: np.ndarray, inpainted: np.ndarray) -> np.ndarray:
    """Build (original | mask | inpainted | diff×3) side-by-side."""
    diff = np.abs(roi.astype(int) - inpainted.astype(int))
    diff = (diff * 3).clip(0, 255).astype(np.uint8)
    return np.concatenate([roi, _mask_to_bgr(mask), inpainted, diff], axis=1)


def _collect_roi_paths(args: argparse.Namespace) -> list[Path]:
    """Find ROI images to process, honoring --roi-path or --roi-root."""
    if args.roi_path:
        p = Path(args.roi_path)
        if not p.exists():
            raise SystemExit(f"--roi-path {p} does not exist")
        return [p]

    roi_root = Path(args.roi_root)
    if not roi_root.exists():
        raise SystemExit(
            f"--roi-root {roi_root} does not exist. Point --roi-path at an "
            "image or run the TPM data gen pipeline first to populate "
            "test_output/roi_extraction_2/."
        )

    track_dirs = sorted(
        d for d in roi_root.iterdir()
        if d.is_dir() and d.name.startswith("track_")
    )[: args.max_tracks]

    paths: list[Path] = []
    for tdir in track_dirs:
        frames = sorted(tdir.glob("frame_*.png"))
        if not frames:
            continue
        n = min(args.frames_per_track, len(frames))
        idxs = [int(round(i * (len(frames) - 1) / max(1, n - 1))) for i in range(n)]
        for idx in idxs:
            paths.append(frames[idx])
    return paths


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Hi-SAM from {args.checkpoint} on {args.device}...")
    inpainter = SegmentationBasedInpainter(
        checkpoint_path=args.checkpoint,
        device=args.device,
        model_type=args.model_type,
        mask_dilation_px=args.mask_dilation_px,
        inpaint_method=args.inpaint_method,
        use_patch_mode=args.use_patch_mode,
    )
    print("Loaded.")

    roi_paths = _collect_roi_paths(args)
    print(f"Processing {len(roi_paths)} ROIs")

    saved = 0
    for roi_path in roi_paths:
        roi = cv2.imread(str(roi_path), cv2.IMREAD_COLOR)
        if roi is None:
            continue

        # Reach into the inpainter to also grab the mask for visualization.
        # The wrapper's inpaint() discards the mask; _ensure_segmenter lets us
        # run segmentation separately without double-loading the model.
        # NOTE: Hi-SAM inference runs twice per ROI here (once for the mask
        # we visualize, once inside inpaint()). That's fine for a dev smoke
        # tool but don't time throughput from this script.
        segmenter = inpainter._ensure_segmenter()
        mask = segmenter.segment(roi)
        dilated = (
            cv2.dilate(mask,
                       cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                       iterations=args.mask_dilation_px)
            if args.mask_dilation_px > 0 else mask
        )
        inpainted = inpainter.inpaint(roi)

        vis = _make_vis(roi, dilated, inpainted)

        # Upscale tiny ROIs for legibility.
        scale = max(1, 256 // roi.shape[0])
        if scale > 1:
            vis = cv2.resize(vis, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_NEAREST)

        stem = roi_path.parent.name + "__" + roi_path.stem if roi_path.parent.name.startswith("track_") else roi_path.stem
        out_path = out_dir / f"{stem}.png"
        cv2.imwrite(str(out_path), vis)
        saved += 1

    print(f"Saved {saved} visualizations to {out_dir}")


if __name__ == "__main__":
    main()
