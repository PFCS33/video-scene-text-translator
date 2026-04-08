"""End-to-end smoke test for S4 with the full TPM (LCM + BPN).

Builds a synthetic frame and plants a real ROI from
test_output/roi_extraction_2 into it as the reference. Adds two more
frames where the same ROI has been Gaussian-blurred to different
strengths, simulating focus / motion blur. Runs S4 with use_lcm=True
and use_bpn=True and saves a side-by-side visualization comparing:

    target ROI | LCM only | LCM + BPN

For BPN to be working correctly, the LCM+BPN column should be visibly
blurrier on rows 1 and 2 than the LCM-only column.

Usage:
    cd code && python scripts/test_s4_bpn_e2e.py \
        --inpainter-checkpoint ../third_party/SRNet/checkpoints/trained_final_5M_.model \
        --bpn-checkpoint checkpoints/bpn/bpn_stage2_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from src.config import PipelineConfig, PropagationConfig  # noqa: E402
from src.data_types import BBox, Quad, TextDetection, TextTrack  # noqa: E402
from src.stages.s4_propagation import PropagationStage  # noqa: E402


def make_blurred_track(
    roi_path: Path, frame_size: tuple[int, int]
) -> tuple[TextTrack, dict[int, np.ndarray]]:
    """Reference + two progressively blurrier targets."""
    H, W = frame_size
    roi = cv2.imread(str(roi_path), cv2.IMREAD_COLOR)
    if roi is None:
        raise RuntimeError(f"Could not read {roi_path}")
    roi_h, roi_w = roi.shape[:2]

    x0, y0 = 80, 60
    quad = Quad(points=np.array([
        [x0, y0],
        [x0 + roi_w, y0],
        [x0 + roi_w, y0 + roi_h],
        [x0, y0 + roi_h],
    ], dtype=np.float32))
    bbox = BBox(x=x0, y=y0, width=roi_w, height=roi_h)

    def plant(blur_sigma: float) -> np.ndarray:
        frame = np.full((H, W, 3), 80, dtype=np.uint8)
        if blur_sigma > 0:
            ksize = max(3, int(blur_sigma * 6) | 1)  # odd
            blurred = cv2.GaussianBlur(roi, (ksize, ksize), blur_sigma)
        else:
            blurred = roi
        frame[y0:y0 + roi_h, x0:x0 + roi_w] = blurred
        return frame

    frames = {
        0: plant(0.0),    # reference, sharp
        1: plant(1.5),    # mildly blurred
        2: plant(3.5),    # heavily blurred
    }

    src_pts = quad.points
    dst_pts = np.array([
        [0, 0], [roi_w, 0], [roi_w, roi_h], [0, roi_h]
    ], dtype=np.float32)
    H_to_frontal, _ = cv2.findHomography(src_pts, dst_pts)
    H_from_frontal = np.linalg.inv(H_to_frontal)

    detections = {}
    for fi in (0, 1, 2):
        detections[fi] = TextDetection(
            frame_idx=fi, quad=quad, bbox=bbox,
            text="text", ocr_confidence=0.9,
            H_to_frontal=H_to_frontal,
            H_from_frontal=H_from_frontal,
            homography_valid=True,
        )

    track = TextTrack(
        track_id=0,
        source_text="text", target_text="text",
        source_lang="en", target_lang="en",
        detections=detections,
        reference_frame_idx=0,
        canonical_size=(roi_w, roi_h),
        edited_roi=roi.copy(),
    )
    return track, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inpainter-checkpoint", type=str,
                        default=str(REPO_ROOT / "third_party/SRNet/checkpoints/trained_final_5M_.model"))
    parser.add_argument("--bpn-checkpoint", type=str,
                        default=str(REPO_ROOT / "code/checkpoints/bpn/bpn_stage2_best.pt"))
    parser.add_argument("--roi-root", type=str,
                        default=str(REPO_ROOT / "test_output/roi_extraction_2"))
    parser.add_argument("--out-dir", type=str,
                        default=str(REPO_ROOT / "test_output/s4_bpn_vis"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-tracks", type=int, default=8)
    parser.add_argument("--bpn-image-size", type=int, nargs=2, default=[64, 128],
                        help="(H, W) the BPN was trained at")
    parser.add_argument("--bpn-kernel-size", type=int, default=41)
    args = parser.parse_args()

    roi_root = Path(args.roi_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    track_dirs = sorted(d for d in roi_root.iterdir()
                        if d.is_dir() and d.name.startswith("track_"))
    track_dirs = track_dirs[:args.max_tracks]
    print(f"Testing on {len(track_dirs)} tracks")

    lcm_only_cfg = PipelineConfig(propagation=PropagationConfig(
        use_lcm=True,
        inpainter_backend="srnet",
        inpainter_checkpoint_path=args.inpainter_checkpoint,
        inpainter_device=args.device,
    ))
    lcm_bpn_cfg = PipelineConfig(propagation=PropagationConfig(
        use_lcm=True,
        inpainter_backend="srnet",
        inpainter_checkpoint_path=args.inpainter_checkpoint,
        inpainter_device=args.device,
        use_bpn=True,
        bpn_checkpoint_path=args.bpn_checkpoint,
        bpn_device=args.device,
        bpn_image_size=tuple(args.bpn_image_size),
        bpn_kernel_size=args.bpn_kernel_size,
    ))
    lcm_stage = PropagationStage(lcm_only_cfg)
    bpn_stage = PropagationStage(lcm_bpn_cfg)

    for tdir in track_dirs:
        rois = sorted(tdir.glob("frame_*.png"))
        if len(rois) < 1:
            continue
        roi_path = rois[len(rois) // 2]

        track_a, frames_a = make_blurred_track(roi_path, frame_size=(256, 512))
        track_b, frames_b = make_blurred_track(roi_path, frame_size=(256, 512))

        lcm_out = lcm_stage.run([track_a], frames_a)
        bpn_out = bpn_stage.run([track_b], frames_b)

        rows = []
        for fi in (0, 1, 2):
            target_roi = cv2.warpPerspective(
                frames_a[fi],
                track_a.detections[fi].H_to_frontal,
                track_a.canonical_size,
            )
            lcm = lcm_out.get(fi, [None])[0]
            bpn = bpn_out.get(fi, [None])[0]
            if lcm is None or bpn is None:
                continue

            scale = max(1, 96 // target_roi.shape[0])
            t = cv2.resize(target_roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            l = cv2.resize(lcm.roi_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            b = cv2.resize(bpn.roi_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            rows.append(np.concatenate([t, l, b], axis=1))

        if not rows:
            continue
        vis = np.concatenate(rows, axis=0)
        out_path = out_dir / f"{tdir.name}__{roi_path.stem}.png"
        cv2.imwrite(str(out_path), vis)

    print(f"Done. Output at {out_dir}")
    print("Each row = (target ROI | LCM only | LCM + BPN). Rows = frame 0 (sharp), 1 (sigma=1.5), 2 (sigma=3.5).")


if __name__ == "__main__":
    main()
