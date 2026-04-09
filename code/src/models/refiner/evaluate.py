"""ROI alignment refiner evaluation script.

Given one or more checkpoints and a training-style config, compute:

    Synthetic metrics (ground truth known):
        - mean corner error (px at network resolution)
        - 90th / 99th percentile corner error
        - IoU of warped support mask vs. ground-truth mask
        - pre-refinement error (identity baseline) for comparison

    Real metrics (no ground truth, delta from identity baseline):
        - masked NCC on luminance (pre vs post)
        - masked Sobel-magnitude Charbonnier (pre vs post)
        - mean predicted corner displacement (sanity: nonzero but small)

And dump visualizations per checkpoint:

    - Side-by-side strips: [source | target | warp(S, ΔH) | diff] per sample
    - Canny edge overlays: red = source, green = target, yellow = aligned
      (before and after refinement)
    - Blink GIFs: 2-frame source/target alternation at 2 Hz, saved as
      animated PNG (widely supported) — catches sub-pixel drift the eye
      spots but scalar metrics miss.

Usage (from code/):

    python -m src.models.refiner.evaluate \\
        --config src/models/refiner/config_stage2.yaml \\
        --checkpoint checkpoints/refiner/stage1/refiner_best.pt \\
        --checkpoint checkpoints/refiner/stage2/refiner_best.pt \\
        --out-dir checkpoints/refiner/eval \\
        --n-vis 24
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import RefinerDataset
from .losses import luminance, masked_charbonnier, masked_ncc, sobel_magnitude
from .model import ROIRefiner
from .warp import canonical_corners, corners_to_H, warp_image, warp_validity_mask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model_from_checkpoint(
    checkpoint_path: str, device: torch.device,
) -> tuple[ROIRefiner, dict]:
    """Load a refiner checkpoint and return the model in eval mode.

    Uses the ``config`` embedded in the checkpoint to reconstruct the model
    architecture, so evaluation is robust to model shape changes between
    training and eval configs.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    mc = cfg.get("model", {})
    dc = cfg.get("data", {})
    image_size = tuple(dc.get("image_size", [64, 128]))
    model = ROIRefiner(
        base_channels=mc.get("base_channels", 32),
        dropout=mc.get("dropout", 0.2),
        image_size=image_size,
        head_init_scale=mc.get("head_init_scale", 1e-3),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    info = {
        "path": str(checkpoint_path),
        "epoch": int(ckpt.get("epoch", -1)) + 1,
        "best_metric": float(ckpt.get("best_metric", float("nan"))),
        "image_size": image_size,
    }
    return model, info


def build_val_loader(config: dict) -> DataLoader:
    """Val DataLoader that produces a balanced mix of synthetic + real pairs.

    Forces ``real_pair_fraction=0.5`` for evaluation so every checkpoint
    gets scored on both distributions under one pass.
    """
    dc = config["data"]
    ds = RefinerDataset(
        data_root=dc["data_root"],
        video_names=list(dc["val_videos"]),
        image_size=tuple(dc.get("image_size", [64, 128])),
        pairs_per_track=dc.get("pairs_per_track", 64),
        real_pair_fraction=0.5,
        syn_perturbation_px=dc.get("syn_perturbation_px", 8.0),
        photometric_strength=dc.get("photometric_strength", 1.0),
        min_track_length=dc.get("min_track_length", 2),
        seed=config.get("seed", 42),
        cache_in_ram=dc.get("cache_in_ram", False),
    )
    tc = config.get("training", {})
    num_workers = int(tc.get("num_workers", 4))
    return DataLoader(
        ds,
        batch_size=int(tc.get("batch_size", 128)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_metrics(
    model: ROIRefiner,
    loader: DataLoader,
    device: torch.device,
    image_size: tuple[int, int],
    progress: bool = True,
) -> dict[str, float]:
    """Compute synthetic + real metrics across the whole val loader.

    For each batch we run the model once, then split samples by type
    (synthetic vs real) via ``has_gt_corners``. Synthetic metrics need
    ground-truth corners; real metrics compare pre- (identity) vs post-
    (predicted) alignment.
    """
    H, W = image_size

    syn_corner_errs: list[float] = []
    syn_mask_ious: list[float] = []
    pre_syn_corner_errs: list[float] = []

    real_pre_ncc: list[float] = []
    real_post_ncc: list[float] = []
    real_pre_grad: list[float] = []
    real_post_grad: list[float] = []
    real_pred_disp: list[float] = []

    pbar = tqdm(loader, desc="eval", leave=False, disable=not progress)
    for batch in pbar:
        source = batch["source"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        delta_gt = batch["delta_corners_gt"].to(device, non_blocking=True)
        has_gt = batch["has_gt_corners"].to(device, non_blocking=True)

        pred_corners = model(source, target)
        B = source.shape[0]
        src_corners_batch = (
            canonical_corners(H, W, device=device, dtype=source.dtype)
            .unsqueeze(0).expand(B, -1, -1).contiguous()
        )
        pred_H = corners_to_H(src_corners_batch, src_corners_batch + pred_corners)

        # Weight for masked photometric metrics = warped validity.
        warped = warp_image(source, pred_H, (H, W))
        valid = warp_validity_mask(pred_H, (H, W), (H, W))

        warped_lum = luminance(warped)
        target_lum = luminance(target)
        source_lum = luminance(source)

        # Per-sample masked NCC and Sobel-magnitude Charbonnier, pre vs post.
        ones_weight = torch.ones_like(valid)
        pre_ncc_batch = masked_ncc(source_lum, target_lum, ones_weight)
        post_ncc_batch = masked_ncc(warped_lum, target_lum, valid)

        pre_sobel_s = sobel_magnitude(source_lum)
        target_sobel = sobel_magnitude(target_lum)
        post_sobel = sobel_magnitude(warped_lum)
        pre_grad_batch = masked_charbonnier(pre_sobel_s, target_sobel, ones_weight)
        post_grad_batch = masked_charbonnier(post_sobel, target_sobel, valid)

        # Corner displacement magnitude (all samples — real or synthetic).
        disp = pred_corners.abs().mean(dim=(1, 2))

        has_gt_cpu = has_gt.cpu().numpy()
        for i in range(B):
            if has_gt_cpu[i]:
                # Synthetic metrics
                err = (pred_corners[i] - delta_gt[i]).abs().mean().item()
                syn_corner_errs.append(err)
                pre_err = delta_gt[i].abs().mean().item()
                pre_syn_corner_errs.append(pre_err)

                # Mask IoU: warp validity of pred_H vs ground-truth H.
                gt_H = corners_to_H(
                    src_corners_batch[i:i + 1],
                    src_corners_batch[i:i + 1] + delta_gt[i:i + 1],
                )
                gt_mask = warp_validity_mask(gt_H, (H, W), (H, W))[0, 0] > 0.5
                pred_mask = valid[i, 0] > 0.5
                inter = (gt_mask & pred_mask).float().sum().item()
                union = (gt_mask | pred_mask).float().sum().item()
                if union > 0:
                    syn_mask_ious.append(inter / union)
            else:
                real_pre_ncc.append(pre_ncc_batch[i].item())
                real_post_ncc.append(post_ncc_batch[i].item())
                real_pre_grad.append(pre_grad_batch[i].item())
                real_post_grad.append(post_grad_batch[i].item())
                real_pred_disp.append(disp[i].item())

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    def _pct(xs: list[float], p: float) -> float:
        return float(np.percentile(xs, p)) if xs else float("nan")

    return {
        # Synthetic
        "syn_n": len(syn_corner_errs),
        "syn_corner_err_mean_px": _mean(syn_corner_errs),
        "syn_corner_err_p90_px": _pct(syn_corner_errs, 90),
        "syn_corner_err_p99_px": _pct(syn_corner_errs, 99),
        "syn_mask_iou_mean": _mean(syn_mask_ious),
        # Pre-refinement (identity baseline) on synthetic for reference.
        "syn_pre_corner_err_mean_px": _mean(pre_syn_corner_errs),
        # Real
        "real_n": len(real_pre_ncc),
        "real_ncc_pre": _mean(real_pre_ncc),
        "real_ncc_post": _mean(real_post_ncc),
        "real_ncc_gain": _mean(real_post_ncc) - _mean(real_pre_ncc),
        "real_grad_pre": _mean(real_pre_grad),
        "real_grad_post": _mean(real_post_grad),
        "real_grad_gain": _mean(real_pre_grad) - _mean(real_post_grad),
        "real_pred_disp_mean_px": _mean(real_pred_disp),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _to_u8(img: torch.Tensor) -> np.ndarray:
    """(3, H, W) float tensor in [0, 1] -> (H, W, 3) uint8 BGR."""
    arr = img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255).astype(np.uint8)
    # Dataset returns RGB; convert to BGR for OpenCV ops.
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _diff_u8(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """Per-channel absolute difference, contrast-stretched for visibility."""
    diff = (a - b).abs().clamp(0, 1)
    diff = diff * 3.0  # stretch small residuals
    return _to_u8(diff.clamp(0, 1))


def _canny_overlay(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Red = source edges, green = target edges, yellow where they coincide.

    Both inputs BGR uint8. Output BGR uint8.
    """
    gray_s = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    gray_t = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    edges_s = cv2.Canny(gray_s, 80, 160)
    edges_t = cv2.Canny(gray_t, 80, 160)
    out = np.zeros_like(source)
    out[..., 2] = edges_s  # red channel in BGR
    out[..., 1] = edges_t  # green channel
    return out


@torch.no_grad()
def dump_visualizations(
    model: ROIRefiner,
    dataset: RefinerDataset,
    device: torch.device,
    image_size: tuple[int, int],
    out_dir: Path,
    n_vis: int = 24,
) -> None:
    """Render comparison strips + edge overlays + blink GIFs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W = image_size

    # Spread indices across the dataset so multiple tracks show up.
    n = min(n_vis, len(dataset))
    if n == 0:
        logger.warning("Empty dataset; skipping visualizations")
        return
    stride = max(1, len(dataset) // n)
    indices = [(i * stride) % len(dataset) for i in range(n)]

    strips: list[np.ndarray] = []
    overlays: list[np.ndarray] = []
    blink_frames_per_sample: list[tuple[np.ndarray, np.ndarray]] = []
    manifest: list[dict[str, Any]] = []

    src_corners = canonical_corners(H, W, device=device).unsqueeze(0)

    for vis_i, ds_idx in enumerate(indices):
        sample = dataset[ds_idx]
        source = sample["source"].unsqueeze(0).to(device)  # (1, 3, H, W)
        target = sample["target"].unsqueeze(0).to(device)

        pred_corners = model(source, target)
        pred_H = corners_to_H(src_corners, src_corners + pred_corners)
        warped = warp_image(source, pred_H, (H, W))

        s_img = _to_u8(source[0])
        t_img = _to_u8(target[0])
        w_img = _to_u8(warped[0])
        d_img = _diff_u8(warped[0], target[0])

        # 4-panel strip with thin black separators
        sep = np.zeros((H, 2, 3), dtype=np.uint8)
        strip = np.concatenate([s_img, sep, t_img, sep, w_img, sep, d_img], axis=1)
        strips.append(strip)

        # Edge overlays: pre (source vs target) vs post (warped vs target)
        pre_overlay = _canny_overlay(s_img, t_img)
        post_overlay = _canny_overlay(w_img, t_img)
        overlay = np.concatenate([pre_overlay, sep, post_overlay], axis=1)
        overlays.append(overlay)

        blink_frames_per_sample.append((w_img, t_img))

        manifest.append({
            "vis_index": vis_i,
            "dataset_index": int(ds_idx),
            "sample_type": sample["sample_type"],
            "pred_corners": pred_corners[0].cpu().numpy().tolist(),
        })

    # Stack all strips vertically into one tall image per view
    row_sep = np.zeros((4, strips[0].shape[1], 3), dtype=np.uint8)
    strip_grid = np.concatenate(
        [np.concatenate([s, row_sep], axis=0) for s in strips[:-1]] + [strips[-1]],
        axis=0,
    )
    cv2.imwrite(str(out_dir / "strips.png"), strip_grid)

    overlay_grid = np.concatenate(
        [np.concatenate([o, row_sep[:, : overlays[0].shape[1]]], axis=0) for o in overlays[:-1]]
        + [overlays[-1]],
        axis=0,
    )
    cv2.imwrite(str(out_dir / "edge_overlays.png"), overlay_grid)

    # Blink GIFs per sample: 2 frames alternating at 2 Hz
    gif_dir = out_dir / "blink"
    gif_dir.mkdir(exist_ok=True)
    for i, (w_img, t_img) in enumerate(blink_frames_per_sample):
        # PIL expects RGB
        frames = [
            Image.fromarray(cv2.cvtColor(w_img, cv2.COLOR_BGR2RGB)),
            Image.fromarray(cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB)),
        ]
        frames[0].save(
            gif_dir / f"blink_{i:03d}.gif",
            save_all=True,
            append_images=frames[1:],
            duration=500,  # ms per frame -> 2 Hz
            loop=0,
        )

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Dumped %d vis samples to %s (strips.png, edge_overlays.png, blink/)",
        n, out_dir,
    )


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def evaluate_checkpoint(
    checkpoint_path: str,
    config: dict,
    out_dir: Path,
    n_vis: int,
    device: torch.device,
    progress: bool = True,
) -> dict[str, Any]:
    model, info = build_model_from_checkpoint(checkpoint_path, device)
    logger.info(
        "Loaded %s (epoch %d, best_metric=%.4f)",
        info["path"], info["epoch"], info["best_metric"],
    )
    loader = build_val_loader(config)

    metrics = compute_metrics(
        model, loader, device, image_size=info["image_size"], progress=progress,
    )
    logger.info("Metrics: %s", json.dumps(metrics, indent=2))

    # Visualizations use a fresh dataset instance with a fixed real_pair
    # fraction so the indices are deterministic and each checkpoint is
    # scored on the exact same samples.
    vis_dataset = RefinerDataset(
        data_root=config["data"]["data_root"],
        video_names=list(config["data"]["val_videos"]),
        image_size=tuple(config["data"].get("image_size", [64, 128])),
        pairs_per_track=config["data"].get("pairs_per_track", 64),
        real_pair_fraction=1.0,  # always real pairs for visualization
        syn_perturbation_px=config["data"].get("syn_perturbation_px", 8.0),
        photometric_strength=0.0,  # disable aug so vis shows true content
        min_track_length=config["data"].get("min_track_length", 2),
        seed=config.get("seed", 42),
        cache_in_ram=config["data"].get("cache_in_ram", False),
    )
    dump_visualizations(
        model, vis_dataset, device, info["image_size"], out_dir, n_vis=n_vis,
    )

    return {"info": info, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ROI alignment refiner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="Checkpoint path — may be passed multiple times")
    parser.add_argument("--out-dir", default="checkpoints/refiner/eval")
    parser.add_argument("--n-vis", type=int, default=24)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(message)s",
    )

    config = load_config(args.config)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {}
    for ckpt_path in args.checkpoint:
        ckpt_name = Path(ckpt_path).parent.name + "_" + Path(ckpt_path).stem
        ckpt_out = out_root / ckpt_name
        result = evaluate_checkpoint(
            ckpt_path, config, ckpt_out, args.n_vis, device,
            progress=not args.no_progress,
        )
        all_results[ckpt_name] = result

    # Write a combined comparison report
    report_path = out_root / "comparison.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    logger.info("Comparison report: %s", report_path)

    # Print a short side-by-side summary table
    print("\n=== Evaluation Summary ===")
    names = list(all_results.keys())
    headline_keys = [
        "syn_corner_err_mean_px",
        "syn_corner_err_p90_px",
        "syn_mask_iou_mean",
        "real_ncc_pre",
        "real_ncc_post",
        "real_ncc_gain",
        "real_grad_pre",
        "real_grad_post",
        "real_grad_gain",
        "real_pred_disp_mean_px",
    ]
    print(f"{'metric':<28}" + "".join(f"{n:>22}" for n in names))
    for key in headline_keys:
        row = f"{key:<28}"
        for name in names:
            val = all_results[name]["metrics"].get(key, float("nan"))
            row += f"{val:>22.4f}"
        print(row)


if __name__ == "__main__":
    main()
