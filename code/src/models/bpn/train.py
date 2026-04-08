"""BPN training script.

Supports two training stages from STRIVE:
- Stage 1: Supervised with synthetic blur (known ground-truth parameters)
- Stage 2: Self-supervised on real video data (reconstruction + temporal loss)

Usage:
    python -m src.models.bpn.train --config src/models/bpn/config.yaml
    python -m src.models.bpn.train --config src/models/bpn/config.yaml --stage 2 --resume checkpoints/bpn_stage1_best.pt
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .blur import DifferentiableBlur
from .dataset import BPNDataset, create_dataloaders
from .losses import BPNLoss
from .model import BPN


def load_config(path: str) -> dict:
    """Load YAML config file."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SyntheticAugmentor:
    """Generate synthetic blur pairs for Stage 1 training.

    Takes real ROI sequences and applies random blur to create
    (reference, blurred_neighbors) pairs with known parameters.
    """

    def __init__(self, blur_module: DifferentiableBlur, device: torch.device):
        self.blur = blur_module
        self.device = device

    def augment(self, batch: dict, n_neighbors: int) -> tuple[dict, dict]:
        """Apply random blur to reference to create synthetic training pairs.

        Args:
            batch: dict from BPNDataset with ref_image, neighbor_images
            n_neighbors: N

        Returns:
            augmented_batch: modified batch with synthetically blurred neighbors
            gt_params: ground truth blur parameters
        """
        ref = batch["ref_image"].to(self.device)  # (B, 3, H, W)
        B = ref.shape[0]

        # Distribution chosen to match real video blur differences between
        # consecutive frames. Wide ranges (the original paper's [0.5, 4.0]
        # for sigma) train the model in a regime that doesn't appear in
        # Stage 2 self-supervised data, leaving Stage 2 to relearn from a
        # weak signal.
        gt_sigma_x = torch.empty(B, n_neighbors, device=self.device).uniform_(0.3, 1.8)
        gt_sigma_y = torch.empty(B, n_neighbors, device=self.device).uniform_(0.3, 1.8)
        gt_rho = torch.empty(B, n_neighbors, device=self.device).uniform_(-math.pi, math.pi)
        gt_w = torch.empty(B, n_neighbors, device=self.device).uniform_(-0.4, 0.4)

        # Generate blurred versions of reference as synthetic neighbors
        synth_neighbors = []
        for i in range(n_neighbors):
            blurred = self.blur(ref, gt_sigma_x[:, i], gt_sigma_y[:, i],
                                gt_rho[:, i], gt_w[:, i])
            synth_neighbors.append(blurred)
        synth_neighbors = torch.stack(synth_neighbors, dim=1)  # (B, N, 3, H, W)

        # Rebuild concatenated input
        imgs_list = [ref]
        for i in range(n_neighbors):
            imgs_list.append(synth_neighbors[:, i])
        synth_images = torch.cat(imgs_list, dim=1)  # (B, 3*(N+1), H, W)

        augmented = {
            "images": synth_images,
            "ref_image": ref,
            "neighbor_images": synth_neighbors,
        }
        gt_params = {
            "sigma_x": gt_sigma_x,
            "sigma_y": gt_sigma_y,
            "rho": gt_rho,
            "w": gt_w,
        }
        return augmented, gt_params


def train_one_epoch(
    model: BPN,
    loader: DataLoader,
    criterion: BPNLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    stage: int,
    augmentor: SyntheticAugmentor | None = None,
) -> dict[str, float]:
    model.train()
    total_losses = {}
    n_batches = 0

    pbar = tqdm(loader, desc="Train", leave=True)
    for batch in pbar:
        if stage == 1 and augmentor is not None:
            # Stage 1: synthetic blur pairs
            batch, gt_params = augmentor.augment(batch, model.n_neighbors)
            images = batch["images"]
            ref = batch["ref_image"]
            neighbors = batch["neighbor_images"]
        else:
            # Stage 2: real data, no gt params
            images = batch["images"].to(device)
            ref = batch["ref_image"].to(device)
            neighbors = batch["neighbor_images"].to(device)
            gt_params = None

        pred_params = model(images)
        losses = criterion(pred_params, ref, neighbors, gt_params)

        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        n_batches += 1

        pbar.set_postfix(loss=f"{losses['total'].item():.4f}",
                         recon=f"{losses['recon'].item():.4f}")

    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def validate(
    model: BPN,
    loader: DataLoader,
    criterion: BPNLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_losses = {}
    n_batches = 0

    pbar = tqdm(loader, desc="Val", leave=True)
    for batch in pbar:
        images = batch["images"].to(device)
        ref = batch["ref_image"].to(device)
        neighbors = batch["neighbor_images"].to(device)

        pred_params = model(images)
        losses = criterion(pred_params, ref, neighbors, gt_params=None)

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        n_batches += 1

        pbar.set_postfix(loss=f"{losses['total'].item():.4f}")

    if n_batches == 0:
        return {k: 0.0 for k in ["total", "recon", "temporal", "psi"]}
    return {k: v / n_batches for k, v in total_losses.items()}


def train(config: dict):
    """Main training loop."""
    # Setup
    seed = config.get("seed", 42)
    set_seed(seed)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    stage = config.get("stage", 1)

    print(f"=== BPN Training Stage {stage} ===")
    print(f"Device: {device}")

    # Data
    dc = config["data"]
    train_loader, val_loader = create_dataloaders(
        data_root=dc["data_root"],
        n_neighbors=dc.get("n_neighbors", 3),
        image_size=tuple(dc.get("image_size", [64, 128])),
        video_indices_train=dc.get("video_indices_train"),
        video_indices_val=dc.get("video_indices_val"),
        max_tracks_per_video_train=dc.get("max_tracks_per_video_train"),
        max_tracks_per_video_val=dc.get("max_tracks_per_video_val"),
        batch_size=dc.get("batch_size", 32),
        num_workers=dc.get("num_workers", 4),
        seed=seed,
        cache_in_ram=dc.get("cache_in_ram", False),
    )
    print(f"Train samples: {len(train_loader.dataset)}, "
          f"Val samples: {len(val_loader.dataset)}")

    # Model
    mc = config.get("model", {})
    n_neighbors = dc.get("n_neighbors", 3)
    model = BPN(
        n_neighbors=n_neighbors,
        pretrained=mc.get("pretrained", True),
    ).to(device)

    # Loss
    blur_module = DifferentiableBlur(
        kernel_size=config.get("blur_kernel_size", 21)
    ).to(device)

    lc = config.get("loss", {})
    criterion = BPNLoss(
        blur_module=blur_module,
        lambda_psi=lc.get("lambda_psi", 1.0),
        lambda_recon=lc.get("lambda_recon", 1.0),
        lambda_temporal=lc.get("lambda_temporal", 0.5),
        use_psi_loss=(stage == 1),
    )

    # Optimizer
    oc = config.get("optimizer", {})
    lr = oc.get("lr_stage1", 0.0005) if stage == 1 else oc.get("lr_stage2", 0.0003)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                  weight_decay=oc.get("weight_decay", 1e-5))

    # LR scheduler: linear warmup + cosine annealing
    sc = config.get("scheduler", {})
    epochs = config.get("epochs", 100)
    warmup_epochs = sc.get("warmup_epochs", 2)

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs),
        eta_min=sc.get("eta_min", 1e-6),
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0 / max(1, warmup_epochs),
        end_factor=1.0, total_iters=warmup_epochs,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    resume_path = config.get("resume")
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and stage == ckpt.get("stage"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from {resume_path} (epoch {start_epoch})")

    # Synthetic augmentor for Stage 1
    augmentor = SyntheticAugmentor(blur_module, device) if stage == 1 else None

    # Checkpoint directory
    ckpt_dir = Path(config.get("checkpoint_dir", "checkpoints/bpn"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training log
    log_path = ckpt_dir / f"train_log_stage{stage}.json"
    history = []

    epochs = config.get("epochs", 100)
    for epoch in range(start_epoch, epochs):
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device, stage, augmentor
        )
        val_losses = validate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train loss: {train_losses['total']:.6f} "
              f"(recon={train_losses['recon']:.6f}, "
              f"temp={train_losses['temporal']:.6f}) | "
              f"Val loss: {val_losses['total']:.6f} | "
              f"LR: {current_lr:.2e}")

        history.append({
            "epoch": epoch + 1,
            "train": train_losses,
            "val": val_losses,
            "lr": current_lr,
        })

        # Save best model
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            save_checkpoint(model, optimizer, epoch, stage, best_val_loss,
                            ckpt_dir / f"bpn_stage{stage}_best.pt")
            print(f"  -> New best val loss: {best_val_loss:.6f}")

        # Periodic save
        if (epoch + 1) % config.get("save_every", 10) == 0:
            save_checkpoint(model, optimizer, epoch, stage, best_val_loss,
                            ckpt_dir / f"bpn_stage{stage}_epoch{epoch+1}.pt")

    # Save final
    save_checkpoint(model, optimizer, epochs - 1, stage, best_val_loss,
                    ckpt_dir / f"bpn_stage{stage}_final.pt")

    # Save training log
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. Log saved to {log_path}")


def save_checkpoint(model, optimizer, epoch, stage, best_val_loss, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "stage": stage,
        "best_val_loss": best_val_loss,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train BPN model")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--stage", type=int, default=None,
                        help="Training stage (1=synthetic, 2=self-supervised)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.stage is not None:
        config["stage"] = args.stage
    if args.resume is not None:
        config["resume"] = args.resume

    train(config)


if __name__ == "__main__":
    main()
