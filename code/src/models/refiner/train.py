"""ROI alignment refiner training script.

Mirrors src/models/bpn/train.py: YAML config, seeded runs, AdamW with warmup
+ cosine LR, periodic + best-val checkpoints, ``--resume`` support.

Training schedule ramps ``real_pair_fraction`` across epochs via the schedule
in config — no hard Stage 1 / Stage 2 split. A batch can mix both types, and
``losses.RefinerLoss`` routes Type A vs Type B losses per-sample. The
headline validation metric is mean corner error (px) on synthetic samples.

Usage:
    python -m src.models.refiner.train --config src/models/refiner/config.yaml
    python -m src.models.refiner.train --config src/models/refiner/config.yaml \\
        --resume checkpoints/refiner/refiner_last.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import RefinerDataset
from .losses import RefinerLoss, RefinerLossWeights
from .model import ROIRefiner

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Config + seeding helpers
# ----------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Per-worker RNG init for DataLoader.

    Without this, forked workers inherit the parent's random state and
    different workers produce identical sample sequences — a known torch
    DataLoader gotcha. This fixes the per-worker Python/numpy RNG; torch's
    own RNG already gets per-worker seeding from DataLoader internals.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def real_pair_fraction_from_schedule(
    epoch: int, schedule: dict[int, float],
) -> float:
    """Piecewise linear interpolation of ``real_pair_fraction`` over epochs.

    ``schedule`` is a dict ``{epoch: value}``. For epochs before the first
    milestone the first value is held; after the last milestone the last
    value is held; in between, linearly interpolate between surrounding pairs.
    """
    if not schedule:
        return 0.0
    milestones = sorted(schedule.keys())
    if epoch <= milestones[0]:
        return float(schedule[milestones[0]])
    if epoch >= milestones[-1]:
        return float(schedule[milestones[-1]])
    for i in range(len(milestones) - 1):
        a, b = milestones[i], milestones[i + 1]
        if a <= epoch <= b:
            t = (epoch - a) / max(1, b - a)
            return float(schedule[a]) + t * (float(schedule[b]) - float(schedule[a]))
    return float(schedule[milestones[-1]])


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def create_dataloaders(
    config: dict, seed: int,
) -> tuple[DataLoader, DataLoader]:
    dc = config["data"]
    tc = config["training"]

    train_videos = list(dc["train_videos"])
    val_videos = list(dc["val_videos"])
    overlap = set(train_videos) & set(val_videos)
    if overlap and not dc.get("allow_train_val_overlap", False):
        raise ValueError(
            f"train and val videos overlap: {sorted(overlap)}. "
            f"Set data.allow_train_val_overlap=true to bypass (test-only)."
        )

    common: dict[str, Any] = dict(
        data_root=dc["data_root"],
        image_size=tuple(dc.get("image_size", [64, 128])),
        pairs_per_track=dc.get("pairs_per_track", 64),
        syn_perturbation_px=dc.get("syn_perturbation_px", 8.0),
        photometric_strength=dc.get("photometric_strength", 1.0),
        min_track_length=dc.get("min_track_length", 2),
        cache_in_ram=dc.get("cache_in_ram", False),
        seed=seed,
    )
    train_ds = RefinerDataset(video_names=train_videos, **common)
    val_ds = RefinerDataset(video_names=val_videos, **common)

    batch_size = tc.get("batch_size", 128)
    num_workers = tc.get("num_workers", 8)
    persistent = num_workers > 0
    loader_kwargs: dict[str, Any] = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent,
        worker_init_fn=seed_worker,
    )
    if persistent:
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


# ----------------------------------------------------------------------
# Train / validate loops
# ----------------------------------------------------------------------

def train_one_epoch(
    model: ROIRefiner,
    loader: DataLoader,
    criterion: RefinerLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    progress: bool = True,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    n_batches = 0
    pbar = tqdm(loader, desc="train", leave=False, disable=not progress)
    for batch in pbar:
        source = batch["source"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        delta_gt = batch["delta_corners_gt"].to(device, non_blocking=True)
        has_gt = batch["has_gt_corners"].to(device, non_blocking=True)

        pred = model(source, target)
        losses = criterion(source, target, pred, delta_gt, has_gt)

        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{losses['total'].item():.4f}")

    return {k: v / max(1, n_batches) for k, v in totals.items()}


@torch.no_grad()
def validate(
    model: ROIRefiner,
    loader: DataLoader,
    criterion: RefinerLoss,
    device: torch.device,
    progress: bool = True,
) -> dict[str, float]:
    """Validate and compute the headline ``corner_err_px`` metric.

    ``corner_err_px`` = mean L1 of (pred_corners - gt_corners) over all
    Type A samples in the val set, in network pixel units. This is the
    metric used for best-checkpoint selection.
    """
    model.eval()
    totals: dict[str, float] = {}
    n_batches = 0
    corner_err_sum = 0.0
    corner_err_n = 0
    pbar = tqdm(loader, desc="val", leave=False, disable=not progress)
    for batch in pbar:
        source = batch["source"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        delta_gt = batch["delta_corners_gt"].to(device, non_blocking=True)
        has_gt = batch["has_gt_corners"].to(device, non_blocking=True)

        pred = model(source, target)
        losses = criterion(source, target, pred, delta_gt, has_gt)

        if has_gt.any():
            err = (pred - delta_gt).abs().mean(dim=(1, 2))  # (B,)
            corner_err_sum += (err * has_gt.float()).sum().item()
            corner_err_n += int(has_gt.sum().item())

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{losses['total'].item():.4f}")

    result = {k: v / max(1, n_batches) for k, v in totals.items()}
    result["corner_err_px"] = corner_err_sum / max(1, corner_err_n)
    return result


# ----------------------------------------------------------------------
# Checkpoint I/O
# ----------------------------------------------------------------------

def save_checkpoint(
    model: ROIRefiner,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_metric: float,
    path: Path,
    config: dict,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "config": config,
        },
        path,
    )


# ----------------------------------------------------------------------
# Main training driver
# ----------------------------------------------------------------------

def train(config: dict) -> dict[str, Any]:
    """Run the full training loop. Returns the final history dict."""
    seed = config.get("seed", 42)
    set_seed(seed)
    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )

    logger.info("=== ROI Refiner Training ===")
    logger.info("Device: %s", device)

    train_loader, val_loader = create_dataloaders(config, seed)
    logger.info(
        "Train: %d samples (%d tracks) | Val: %d samples (%d tracks)",
        len(train_loader.dataset),
        len(train_loader.dataset.track_paths),  # type: ignore[attr-defined]
        len(val_loader.dataset),
        len(val_loader.dataset.track_paths),    # type: ignore[attr-defined]
    )

    mc = config.get("model", {})
    dc = config["data"]
    image_size = tuple(dc.get("image_size", [64, 128]))
    model = ROIRefiner(
        base_channels=mc.get("base_channels", 32),
        dropout=mc.get("dropout", 0.2),
        image_size=image_size,
        head_init_scale=mc.get("head_init_scale", 1e-3),
    ).to(device)
    logger.info("Model: %d parameters", model.num_parameters())

    lc = config.get("loss", {})
    weights = RefinerLossWeights(
        corner=lc.get("corner", 1.0),
        recon=lc.get("recon", 0.25),
        ncc=lc.get("ncc", 1.0),
        grad=lc.get("grad", 1.0),
        reg=lc.get("reg", 0.01),
    )
    criterion = RefinerLoss(
        image_size=image_size,
        weights=weights,
        border_frac=lc.get("border_frac", 0.1),
        edge_value=lc.get("edge_value", 0.1),
        charbonnier_eps=lc.get("charbonnier_eps", 1e-3),
    ).to(device)

    tc = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc.get("lr", 1e-4),
        weight_decay=tc.get("weight_decay", 1e-4),
    )

    epochs = tc.get("epochs", 50)
    warmup = max(1, tc.get("warmup_epochs", 2))
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup), eta_min=tc.get("eta_min", 1e-6),
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup],
    )

    schedule_raw = tc.get("real_pair_schedule", {0: 0.0})
    schedule = {int(k): float(v) for k, v in schedule_raw.items()}

    start_epoch = 0
    best_metric = float("inf")
    resume_path = config.get("resume")
    init_from = config.get("init_from")
    if resume_path and init_from:
        raise ValueError("use either 'resume' or 'init_from', not both")
    if resume_path and os.path.exists(resume_path):
        # Full resume: model + optimizer + scheduler + epoch counter + best
        # metric. Use for mid-training restarts.
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_metric = float(ckpt.get("best_metric", float("inf")))
        logger.info("Resumed from %s (epoch %d)", resume_path, start_epoch)
    elif init_from and os.path.exists(init_from):
        # Weights-only init: load just the model parameters and start with
        # a fresh optimizer, scheduler, and epoch counter. Use for stage
        # transitions where the new config's LR / epoch count / loss mix
        # should take effect cleanly.
        ckpt = torch.load(init_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            "Initialized model weights from %s (fresh optimizer/scheduler)",
            init_from,
        )

    ckpt_cfg = config.get("checkpoint", {})
    ckpt_dir = Path(ckpt_cfg.get("out_dir", "checkpoints/refiner"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_every = int(ckpt_cfg.get("save_every_epochs", 5))

    progress = config.get("progress", True)
    history: list[dict[str, Any]] = []
    log_path = ckpt_dir / "train_log.json"

    for epoch in range(start_epoch, epochs):
        rp_frac = real_pair_fraction_from_schedule(epoch, schedule)
        train_loader.dataset.real_pair_fraction = rp_frac  # type: ignore[attr-defined]
        val_loader.dataset.real_pair_fraction = rp_frac    # type: ignore[attr-defined]

        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=tc.get("grad_clip", 1.0),
            progress=progress,
        )
        val_losses = validate(model, val_loader, criterion, device, progress=progress)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %d/%d | real_frac=%.2f | train total=%.4f "
            "(corner=%.4f recon=%.4f ncc=%.4f grad=%.4f) | "
            "val total=%.4f corner_err=%.3f px | lr=%.2e",
            epoch + 1, epochs, rp_frac,
            train_losses["total"], train_losses["corner"], train_losses["recon"],
            train_losses["ncc"], train_losses["grad"],
            val_losses["total"], val_losses["corner_err_px"], current_lr,
        )

        history.append({
            "epoch": epoch + 1,
            "real_pair_fraction": rp_frac,
            "train": train_losses,
            "val": val_losses,
            "lr": current_lr,
        })
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2, default=float)

        metric = val_losses["corner_err_px"]
        if metric < best_metric:
            best_metric = metric
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_metric,
                ckpt_dir / "refiner_best.pt", config,
            )
            logger.info("  -> new best corner_err_px: %.4f", best_metric)

        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_metric,
                ckpt_dir / f"refiner_epoch{epoch + 1}.pt", config,
            )

    save_checkpoint(
        model, optimizer, scheduler, epochs - 1, best_metric,
        ckpt_dir / "refiner_last.pt", config,
    )
    logger.info("Training complete. Best corner_err_px: %.4f", best_metric)

    return {"history": history, "best_metric": best_metric}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ROI alignment refiner")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume", default=None,
        help="Full resume: load model + optimizer + scheduler + epoch "
             "from a checkpoint. Use for mid-training restarts.",
    )
    parser.add_argument(
        "--init-from", default=None,
        help="Weights-only init: load only model parameters from a "
             "checkpoint. Fresh optimizer/scheduler/epoch. Use for stage "
             "transitions (e.g. Stage 2 fine-tune from Stage 1 best).",
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config['training']['epochs']")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = load_config(args.config)
    if args.resume:
        config["resume"] = args.resume
    if args.init_from:
        config["init_from"] = args.init_from
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.device is not None:
        config["device"] = args.device
    if args.no_progress:
        config["progress"] = False

    train(config)


if __name__ == "__main__":
    main()
