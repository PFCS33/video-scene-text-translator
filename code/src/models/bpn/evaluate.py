"""BPN evaluation and visualization.

Produces:
1. Quantitative metrics (reconstruction MSE, parameter stats)
2. Visual comparisons: reference | target | predicted blur | difference

Usage:
    python -m src.models.bpn.evaluate --config src/models/bpn/config.yaml --checkpoint checkpoints/bpn/bpn_stage1_best.pt
    python -m src.models.bpn.evaluate --config src/models/bpn/config.yaml --checkpoint checkpoints/bpn/bpn_stage1_best.pt --output-dir eval_output/
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .blur import DifferentiableBlur
from .dataset import BPNDataset
from .model import BPN


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate_metrics(
    model: BPN,
    loader: DataLoader,
    blur_module: DifferentiableBlur,
    device: torch.device,
) -> dict:
    """Compute quantitative evaluation metrics."""
    model.eval()
    all_mse = []
    all_params = {"sigma_x": [], "sigma_y": [], "rho": [], "w": []}

    for batch in loader:
        images = batch["images"].to(device)
        ref = batch["ref_image"].to(device)
        neighbors = batch["neighbor_images"].to(device)

        pred = model(images)
        B, N, C, H, W = neighbors.shape

        for i in range(N):
            blurred = blur_module(
                ref, pred["sigma_x"][:, i], pred["sigma_y"][:, i],
                pred["rho"][:, i], pred["w"][:, i],
            )
            mse = ((blurred - neighbors[:, i]) ** 2).mean(dim=(1, 2, 3))
            all_mse.extend(mse.cpu().numpy().tolist())

        for k in all_params:
            all_params[k].extend(pred[k].cpu().numpy().reshape(-1).tolist())

    metrics = {
        "reconstruction_mse": {
            "mean": float(np.mean(all_mse)),
            "std": float(np.std(all_mse)),
            "median": float(np.median(all_mse)),
        },
        "param_stats": {},
    }
    for k, vals in all_params.items():
        arr = np.array(vals)
        metrics["param_stats"][k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    return metrics


def _select_samples_from_distinct_tracks(
    dataset, max_samples: int, seed: int | None = None,
) -> list[int]:
    """Pick dataset indices such that each comes from a different track.

    Uses the dataset's `sample_track_ids` parallel list (populated at build
    time) to group sample indices by track, shuffles tracks, then picks one
    random sample from each.
    """
    import random as _random
    rng = _random.Random(seed)  # seed=None -> system entropy, different each run

    track_to_indices: dict[str, list[int]] = {}
    for i, tid in enumerate(dataset.sample_track_ids):
        track_to_indices.setdefault(tid, []).append(i)

    track_ids = list(track_to_indices.keys())
    rng.shuffle(track_ids)
    track_ids = track_ids[:max_samples]

    return [rng.choice(track_to_indices[t]) for t in track_ids]


@torch.no_grad()
def generate_visualizations(
    model: BPN,
    loader: DataLoader,
    blur_module: DifferentiableBlur,
    device: torch.device,
    output_dir: Path,
    max_samples: int = 20,
    targets_per_sample: int = 16,
):
    """Generate side-by-side comparison images.

    Each sample is drawn from a different track. For each sample we visualize
    up to `targets_per_sample` consecutive target frames (or fewer if the
    track is shorter). The model is run in sliding chunks of n_neighbors to
    cover all targets, since the network's input window is fixed.

    Layout per sample: rows of [ref | target_i | predicted_i | diff_i].
    """
    import cv2

    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = loader.dataset
    n_neighbors = model.n_neighbors
    selected_indices = _select_samples_from_distinct_tracks(dataset, max_samples)
    print(f"Selected {len(selected_indices)} samples from {len(selected_indices)} distinct tracks")

    for sample_idx, ds_idx in enumerate(selected_indices):
        ref, targets = dataset.get_track_window(ds_idx, targets_per_sample)
        T = targets.shape[0]
        if T == 0:
            continue

        ref_dev = ref.unsqueeze(0).to(device)         # (1, 3, H, W)
        targets_dev = targets.to(device)               # (T, 3, H, W)

        # Run model in sliding chunks of n_neighbors. For each target index t,
        # we want a prediction; we use the chunk that contains it. To minimize
        # forward passes, do non-overlapping chunks tiling [0, T).
        pred_sigma_x = torch.zeros(T, device=device)
        pred_sigma_y = torch.zeros(T, device=device)
        pred_rho = torch.zeros(T, device=device)
        pred_w = torch.zeros(T, device=device)

        for chunk_start in range(0, T, n_neighbors):
            chunk_end = min(chunk_start + n_neighbors, T)
            chunk_len = chunk_end - chunk_start

            # Pad chunk to n_neighbors by repeating last target
            chunk = targets_dev[chunk_start:chunk_end]
            if chunk_len < n_neighbors:
                pad = chunk[-1:].expand(n_neighbors - chunk_len, -1, -1, -1)
                chunk = torch.cat([chunk, pad], dim=0)

            # Build network input: ref + n_neighbors targets, all concatenated
            stacked = torch.cat([ref_dev[0], *chunk], dim=0).unsqueeze(0)  # (1, 3*(N+1), H, W)
            out = model(stacked)
            pred_sigma_x[chunk_start:chunk_end] = out["sigma_x"][0, :chunk_len]
            pred_sigma_y[chunk_start:chunk_end] = out["sigma_y"][0, :chunk_len]
            pred_rho[chunk_start:chunk_end] = out["rho"][0, :chunk_len]
            pred_w[chunk_start:chunk_end] = out["w"][0, :chunk_len]

        # Build visualization rows
        ref_np = _tensor_to_bgr(ref_dev[0])
        H, W = ref_np.shape[:2]
        rows = []
        for i in range(T):
            blurred = blur_module(
                ref_dev,
                pred_sigma_x[i:i+1],
                pred_sigma_y[i:i+1],
                pred_rho[i:i+1],
                pred_w[i:i+1],
            )
            pred_np = _tensor_to_bgr(blurred[0])
            target_np = _tensor_to_bgr(targets_dev[i])
            diff_np = np.abs(target_np.astype(float) - pred_np.astype(float))
            diff_np = (diff_np * 3).clip(0, 255).astype(np.uint8)

            label_h = 20
            ref_labeled = _add_label(ref_np, "Reference", label_h)
            target_labeled = _add_label(target_np, f"Target {i}", label_h)
            pred_labeled = _add_label(pred_np, f"Predicted {i}", label_h)
            diff_labeled = _add_label(diff_np, f"Diff {i} (3x)", label_h)

            row = np.concatenate(
                [ref_labeled, target_labeled, pred_labeled, diff_labeled],
                axis=1,
            )
            rows.append(row)

        vis = np.concatenate(rows, axis=0)
        scale = max(1, 256 // H)
        vis = cv2.resize(vis, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_NEAREST)

        out_path = output_dir / f"sample_{sample_idx:04d}.png"
        cv2.imwrite(str(out_path), vis)

    print(f"Saved {len(selected_indices)} visualizations to {output_dir}")


def plot_training_log(log_path: str, output_dir: Path):
    """Plot training curves from JSON log using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping training curve plot")
        return

    with open(log_path) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    train_total = [h["train"]["total"] for h in history]
    val_total = [h["val"]["total"] for h in history]
    train_recon = [h["train"]["recon"] for h in history]
    val_recon = [h["val"]["recon"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_total, label="Train Total")
    axes[0].plot(epochs, val_total, label="Val Total")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Total Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_recon, label="Train Recon")
    axes[1].plot(epochs, val_recon, label="Val Recon")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Reconstruction Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "training_curves.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"Training curves saved to {out_path}")


def _tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) float tensor [0,1] to BGR uint8 numpy array."""
    img = (t.cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img[:, :, ::-1].copy()  # RGB -> BGR


def _add_label(img: np.ndarray, text: str, label_h: int = 20) -> np.ndarray:
    """Add text label above image."""
    import cv2
    H, W = img.shape[:2]
    label = np.zeros((label_h, W, 3), dtype=np.uint8)
    cv2.putText(label, text, (2, label_h - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([label, img], axis=0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate BPN model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="eval_output/bpn")
    parser.add_argument("--max-vis", type=int, default=20,
                        help="Max visualization samples (one per track)")
    parser.add_argument("--targets-per-sample", type=int, default=16,
                        help="Number of consecutive target frames per sample")
    parser.add_argument("--training-log", type=str, default=None,
                        help="Path to training log JSON for curve plotting")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dc = config["data"]
    n_neighbors = dc.get("n_neighbors", 3)

    # Load model
    model = BPN(n_neighbors=n_neighbors, pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    # Validation data
    val_ds = BPNDataset(
        data_root=dc["data_root"],
        n_neighbors=n_neighbors,
        image_size=tuple(dc.get("image_size", [64, 128])),
        video_indices=dc.get("video_indices_val", [8, 9]),
        max_tracks_per_video=dc.get("max_tracks_per_video_val"),
    )
    val_loader = DataLoader(val_ds, batch_size=dc.get("batch_size", 32),
                            shuffle=False, num_workers=2)

    blur_module = DifferentiableBlur(
        kernel_size=config.get("blur_kernel_size", 21)
    ).to(device)

    # Quantitative evaluation
    print("Computing metrics...")
    metrics = evaluate_metrics(model, val_loader, blur_module, device)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    print(f"  Reconstruction MSE: {metrics['reconstruction_mse']['mean']:.6f} "
          f"+/- {metrics['reconstruction_mse']['std']:.6f}")
    for k, v in metrics["param_stats"].items():
        print(f"  {k}: mean={v['mean']:.4f}, std={v['std']:.4f}, "
              f"range=[{v['min']:.4f}, {v['max']:.4f}]")

    # Visual evaluation
    print("Generating visualizations...")
    generate_visualizations(model, val_loader, blur_module, device,
                            output_dir / "vis", max_samples=args.max_vis,
                            targets_per_sample=args.targets_per_sample)

    # Training curves
    if args.training_log:
        plot_training_log(args.training_log, output_dir)


if __name__ == "__main__":
    main()
