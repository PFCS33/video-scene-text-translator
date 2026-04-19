#!/usr/bin/env python3
"""Generate a BPN training dataset using S2-refined homographies.

For one source video + pre-saved s1_tracks.json, re-run S2 frontalization
with the alignment refiner on, and extract the corrected canonical ROIs.
Writes two things under ``--output-dir``:

  * ``corrected_track_info.json`` — per-track metadata with the refined
    homography matrices inlined (unlike ``s1_tracks.json`` this includes
    H_to_frontal / H_from_frontal so a downstream trainer can recompute
    ROIs without re-running the refiner).
  * ``track_{id:02d}_{text}/frame_{idx:06d}.png`` — one PNG per frame per
    track, warped into the track's canonical frontal rectangle using the
    corrected H_to_frontal.

Streaming: frames are read on-demand via ``VideoReader.read_frame(idx)``.
The only per-track frame held in memory is the reference canonical ROI
(used by the refiner to score each target). Peak memory is O(one frame
+ one canonical ROI) regardless of video length.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Make ``src.*`` importable when invoked from anywhere.
_THIS = Path(__file__).resolve()
_CODE_ROOT = _THIS.parent.parent  # .../code
sys.path.insert(0, str(_CODE_ROOT))

from src.config import PipelineConfig  # noqa: E402
from src.data_types import TextTrack  # noqa: E402
from src.stages.s2_frontalization import FrontalizationStage  # noqa: E402
from src.video_io import VideoReader  # noqa: E402

logger = logging.getLogger("generate_bpn_dataset")


def build_config(
    refiner_checkpoint: str,
    refiner_device: str,
    use_refiner: bool,
) -> PipelineConfig:
    """Minimal PipelineConfig for the S2-only data-gen path.

    Only the ``frontalization`` block matters here — S1, S3, S4, S5
    are not run. ``input_video`` / ``output_video`` are set to dummy
    non-empty strings so ``validate()`` passes.
    """
    cfg = PipelineConfig()
    cfg.input_video = "dummy.mp4"
    cfg.output_video = "dummy_out.mp4"
    cfg.frontalization.use_refiner = use_refiner
    cfg.frontalization.refiner_checkpoint_path = refiner_checkpoint
    cfg.frontalization.refiner_device = refiner_device
    errors = cfg.validate()
    if errors:
        raise ValueError(f"Invalid config: {'; '.join(errors)}")
    return cfg


def load_tracks(s1_tracks_path: Path) -> list[TextTrack]:
    data = json.loads(s1_tracks_path.read_text())
    tracks = [TextTrack.from_json_serializable(t) for t in data]
    logger.info("Loaded %d tracks from %s", len(tracks), s1_tracks_path)
    return tracks


def track_frame_range(track: TextTrack) -> tuple[int, int] | None:
    if not track.detections:
        return None
    return min(track.detections), max(track.detections)


def refine_and_extract_track(
    track: TextTrack,
    reader: VideoReader,
    stage: FrontalizationStage,
    output_dir: Path,
    pbar: "tqdm | None" = None,
) -> dict | None:
    """Refine one track's homographies and dump its canonical ROIs.

    Returns metadata for ``corrected_track_info.json``, or ``None`` if
    the track was skipped (missing reference frame, degenerate quad, ...).

    Flow:
      1. Run the unrefined S2 homography computation (frames=None) to
         populate ``det.H_to_frontal`` on every detection and
         ``track.canonical_size``.
      2. Read the reference frame once, warp to ``ref_canonical``.
      3. For each non-reference frame in the track's range (sorted), read
         the frame sequentially, build ``target_canonical`` via the
         unrefined H, ask the refiner for ΔH, fold it into the
         detection's H per the S2 migration rule
         (``H_to_frontal_corrected = inv(ΔH) @ H_to_frontal_unrefined``).
      4. With the corrected H now on each detection, write the canonical
         ROI PNG for every detection in the same sequential pass.
    """
    rng = track_frame_range(track)
    if rng is None:
        logger.warning("Track %d has no detections, skipping", track.track_id)
        return None
    track_start, track_end = rng

    ref_idx = track.reference_frame_idx
    if ref_idx < 0 or ref_idx not in track.detections:
        logger.warning(
            "Track %d has no valid reference frame, skipping", track.track_id
        )
        return None

    # Step 1: unrefined homography population (sets track.canonical_size).
    stage.compute_homographies(track, frames=None)
    if track.canonical_size is None:
        logger.warning(
            "Track %d has no canonical_size after S2, skipping", track.track_id
        )
        return None

    canonical_size = track.canonical_size  # (W, H)
    ref_det = track.detections[ref_idx]
    if not ref_det.homography_valid or ref_det.H_to_frontal is None:
        logger.warning(
            "Track %d reference detection has invalid homography, skipping",
            track.track_id,
        )
        return None

    # Step 2: pull the reference frame once.
    ref_frame = reader.read_frame(ref_idx)
    if ref_frame is None:
        logger.warning(
            "Track %d: failed to read reference frame %d, skipping",
            track.track_id, ref_idx,
        )
        return None
    try:
        ref_canonical = cv2.warpPerspective(
            ref_frame, ref_det.H_to_frontal, canonical_size,
        )
    except cv2.error as exc:
        logger.warning(
            "Track %d: ref canonical warp failed (%s), skipping",
            track.track_id, exc,
        )
        return None
    if ref_canonical.size == 0:
        logger.warning("Track %d: ref canonical is empty, skipping", track.track_id)
        return None

    # Step 3 + 4: one sequential sweep through the track's range —
    # refine on-the-fly, write the PNG right after refinement so we never
    # decode any frame twice.
    track_output_dir = (
        output_dir / f"track_{track.track_id:02d}_{track.source_text}"
    )
    track_output_dir.mkdir(parents=True, exist_ok=True)

    refine_total = 0
    refine_rejected = 0
    extracted = 0
    refiner = stage._refiner  # None if refiner disabled

    # Process the reference frame's PNG first, using ref_canonical we
    # already computed (no refinement on the reference itself).
    ref_png = track_output_dir / f"frame_{ref_idx:06d}.png"
    cv2.imwrite(str(ref_png), ref_canonical)
    extracted += 1
    if pbar is not None:
        pbar.update(1)

    for frame_idx in range(track_start, track_end + 1):
        if frame_idx == ref_idx:
            continue
        det = track.detections.get(frame_idx)
        if det is None or not det.homography_valid or det.H_to_frontal is None:
            continue

        frame = reader.read_frame(frame_idx)
        if frame is None:
            logger.debug(
                "Track %d: read failed at frame %d, skipping",
                track.track_id, frame_idx,
            )
            continue

        # Build target canonical under the *unrefined* H so the refiner
        # sees the same pair it would see at pipeline runtime.
        try:
            target_canonical = cv2.warpPerspective(
                frame, det.H_to_frontal, canonical_size,
            )
        except cv2.error:
            continue
        if target_canonical.size == 0:
            continue

        if refiner is not None:
            refine_total += 1
            try:
                delta_H = refiner.predict_delta_H(ref_canonical, target_canonical)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Track %d frame %d: refiner raised %s; keeping unrefined",
                    track.track_id, frame_idx, exc,
                )
                delta_H = None
            if delta_H is None:
                refine_rejected += 1
            else:
                try:
                    delta_H_inv = np.linalg.inv(delta_H)
                except np.linalg.LinAlgError:
                    refine_rejected += 1
                else:
                    det.H_to_frontal = delta_H_inv @ det.H_to_frontal
                    det.H_from_frontal = det.H_from_frontal @ delta_H

        # Extract ROI with the (possibly refined) H. Re-warp from the
        # original frame so we never re-interpolate a canonical crop.
        try:
            warped = cv2.warpPerspective(
                frame, det.H_to_frontal, canonical_size,
                flags=cv2.INTER_LINEAR,
            )
        except cv2.error:
            continue
        if warped.size == 0:
            continue

        out_png = track_output_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(out_png), warped)
        extracted += 1
        if pbar is not None:
            pbar.update(1)

    meta = {
        "track_id": track.track_id,
        "source_text": track.source_text,
        "reference_frame_idx": ref_idx,
        "canonical_size": list(canonical_size),
        "begin_frame_idx": track_start,
        "end_frame_idx": track_end,
        "extracted_count": extracted,
        "refine_total": refine_total,
        "refine_rejected": refine_rejected,
        "detections": {
            str(idx): {
                "frame_idx": det.frame_idx,
                "quad": det.quad.points.tolist(),
                "text": det.text,
                "ocr_confidence": det.ocr_confidence,
                "homography_valid": bool(det.homography_valid),
                # Full 3x3 matrices — consumers can rebuild canonical
                # ROIs without having to re-run the refiner.
                "H_to_frontal": (
                    det.H_to_frontal.tolist()
                    if det.H_to_frontal is not None else None
                ),
                "H_from_frontal": (
                    det.H_from_frontal.tolist()
                    if det.H_from_frontal is not None else None
                ),
            }
            for idx, det in sorted(track.detections.items())
        },
    }
    return meta


def process_video(
    video_path: Path,
    s1_tracks_path: Path,
    output_dir: Path,
    refiner_checkpoint: str,
    refiner_device: str,
    use_refiner: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks = load_tracks(s1_tracks_path)

    cfg = build_config(refiner_checkpoint, refiner_device, use_refiner)
    stage = FrontalizationStage(cfg)

    # Sort tracks by start frame so the reader mostly streams forward.
    tracks_sorted = sorted(
        tracks,
        key=lambda t: min(t.detections) if t.detections else 0,
    )

    all_meta: list[dict] = []
    totals = {"refine_total": 0, "refine_rejected": 0, "extracted": 0}

    # Upper bound on ROIs to extract: one per detection per track. The
    # actual count can fall short when a detection has invalid geometry
    # or a frame read fails, so the bar's total is an over-estimate —
    # tqdm is tolerant of that.
    roi_budget = sum(len(t.detections) for t in tracks_sorted)
    with VideoReader(str(video_path)) as reader:
        logger.info(
            "Video opened: %s (%d frames, %.2f fps, %dx%d)",
            video_path, reader.frame_count, reader.fps,
            reader.frame_size[0], reader.frame_size[1],
        )
        with tqdm(total=roi_budget, desc=video_path.name, unit="roi") as pbar:
            for track in tracks_sorted:
                meta = refine_and_extract_track(
                    track, reader, stage, output_dir, pbar=pbar,
                )
                if meta is None:
                    continue
                all_meta.append(meta)
                totals["refine_total"] += meta["refine_total"]
                totals["refine_rejected"] += meta["refine_rejected"]
                totals["extracted"] += meta["extracted_count"]

    corrected_json = output_dir / "corrected_track_info.json"
    corrected_json.write_text(json.dumps(all_meta, indent=2))
    logger.info(
        "Wrote %s (%d tracks, %d ROIs, refiner %d/%d rejected %.1f%%)",
        corrected_json, len(all_meta), totals["extracted"],
        totals["refine_rejected"], totals["refine_total"],
        (100.0 * totals["refine_rejected"] / totals["refine_total"])
        if totals["refine_total"] else 0.0,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, type=Path,
                   help="Source video file (.mp4).")
    p.add_argument("--s1-tracks", required=True, type=Path,
                   help="Pre-saved s1_tracks.json for this video.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Per-video output directory.")
    p.add_argument("--refiner-checkpoint", required=True, type=str,
                   help="Path to the trained alignment refiner checkpoint.")
    p.add_argument("--refiner-device", default="cuda",
                   help="Torch device for the refiner (default: cuda).")
    p.add_argument("--no-refine", action="store_true",
                   help="Skip refinement (unrefined homographies — useful "
                        "for A/B comparison against the original TPM dataset).")
    p.add_argument("--log-level", default="INFO",
                   help="Logging level (default: INFO).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.video.exists():
        logger.error("Video not found: %s", args.video)
        return 1
    if not args.s1_tracks.exists():
        logger.error("s1_tracks.json not found: %s", args.s1_tracks)
        return 1

    process_video(
        video_path=args.video,
        s1_tracks_path=args.s1_tracks,
        output_dir=args.output_dir,
        refiner_checkpoint=args.refiner_checkpoint,
        refiner_device=args.refiner_device,
        use_refiner=not args.no_refine,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
