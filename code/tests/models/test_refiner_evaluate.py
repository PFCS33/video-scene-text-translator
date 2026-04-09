"""Unit tests for ``src.models.refiner.evaluate``.

Uses a tiny fake dataset + checkpoint under ``tmp_path`` so tests stay
independent of ``/workspace/tpm_dataset/`` and previous training runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.models.refiner.evaluate import (
    build_model_from_checkpoint,
    build_val_loader,
    compute_metrics,
    dump_visualizations,
    evaluate_checkpoint,
)
from src.models.refiner.model import ROIRefiner

# ---------------------------------------------------------------------------
# Fake dataset + checkpoint fixtures
# ---------------------------------------------------------------------------


def _make_fake_video(root: Path, name: str, n_frames: int = 8) -> None:
    tdir = root / name / "track_00_FOO"
    tdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(hash(name) & 0xFFFF)
    for i in range(n_frames):
        arr = rng.integers(0, 256, (80, 160, 3), dtype=np.uint8)
        # Bright vertical bars so there are real edges for Sobel/Canny.
        arr[:, 20::32] = 255
        Image.fromarray(arr).save(tdir / f"frame_{i:06d}.png")


@pytest.fixture
def fake_eval_setup(tmp_path: Path) -> tuple[dict, str]:
    """Return (config, checkpoint_path) for evaluation tests."""
    _make_fake_video(tmp_path, "vid_train")
    _make_fake_video(tmp_path, "vid_val")

    config = {
        "seed": 0,
        "device": "cpu",
        "data": {
            "data_root": str(tmp_path),
            "train_videos": ["vid_train"],
            "val_videos": ["vid_val"],
            "image_size": [64, 128],
            "pairs_per_track": 4,
            "syn_perturbation_px": 6.0,
            "photometric_strength": 0.0,
            "min_track_length": 2,
            "cache_in_ram": False,
        },
        "training": {
            "batch_size": 4,
            "num_workers": 0,
        },
        "model": {
            "base_channels": 16,
            "dropout": 0.0,
            "head_init_scale": 1.0e-3,
        },
    }

    # Hand-craft a checkpoint with a small untrained model. The point is
    # for evaluate() to accept it, not for the metrics to be good.
    model = ROIRefiner(
        base_channels=16, dropout=0.0, image_size=(64, 128), head_init_scale=1e-3,
    )
    ckpt_path = tmp_path / "fake_refiner.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "epoch": 0,
            "best_metric": 0.5,
            "config": config,
        },
        ckpt_path,
    )
    return config, str(ckpt_path)


# ---------------------------------------------------------------------------
# build_model_from_checkpoint
# ---------------------------------------------------------------------------


def test_build_model_from_checkpoint(fake_eval_setup):
    _, ckpt_path = fake_eval_setup
    device = torch.device("cpu")
    model, info = build_model_from_checkpoint(ckpt_path, device)
    assert isinstance(model, ROIRefiner)
    assert not model.training  # eval mode
    assert info["epoch"] == 1
    assert info["best_metric"] == 0.5
    assert info["image_size"] == (64, 128)


def test_build_model_reconstructs_base_channels(fake_eval_setup):
    """Checkpoint embeds model hyperparams; eval must honor them."""
    _, ckpt_path = fake_eval_setup
    device = torch.device("cpu")
    model, _ = build_model_from_checkpoint(ckpt_path, device)
    # We used base_channels=16 -> first conv has 16 output channels.
    first_conv = model.backbone[0][0]
    assert first_conv.out_channels == 16


# ---------------------------------------------------------------------------
# build_val_loader
# ---------------------------------------------------------------------------


def test_build_val_loader(fake_eval_setup):
    config, _ = fake_eval_setup
    loader = build_val_loader(config)
    batch = next(iter(loader))
    assert batch["source"].shape[1:] == (3, 64, 128)
    assert batch["target"].shape[1:] == (3, 64, 128)
    # Forced mix — should produce a mixture of sample types over the whole
    # dataset.
    types = []
    for b in loader:
        types.extend(b["sample_type"])
    assert "syn" in types
    # Real is stochastic; with fraction=0.5 and small set it should usually
    # appear, but don't hard-fail on it.


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_returns_expected_keys(fake_eval_setup):
    config, ckpt_path = fake_eval_setup
    device = torch.device("cpu")
    model, info = build_model_from_checkpoint(ckpt_path, device)
    loader = build_val_loader(config)
    metrics = compute_metrics(
        model, loader, device, info["image_size"], progress=False,
    )
    expected = {
        "syn_n", "syn_corner_err_mean_px", "syn_corner_err_p90_px",
        "syn_corner_err_p99_px", "syn_mask_iou_mean",
        "syn_pre_corner_err_mean_px",
        "real_n", "real_ncc_pre", "real_ncc_post", "real_ncc_gain",
        "real_grad_pre", "real_grad_post", "real_grad_gain",
        "real_pred_disp_mean_px",
    }
    assert expected.issubset(metrics.keys())
    # All numeric values should be finite or NaN — never inf
    for k, v in metrics.items():
        if isinstance(v, float):
            assert not np.isinf(v), f"{k} is inf"


def test_compute_metrics_untrained_model_near_identity_gives_small_disp(fake_eval_setup):
    """Refiner with near-identity init should predict tiny displacements.

    Pins the combined init-scale + metric path: if anything makes the
    untrained model suddenly predict large warps, this breaks.
    """
    config, ckpt_path = fake_eval_setup
    device = torch.device("cpu")
    model, info = build_model_from_checkpoint(ckpt_path, device)
    loader = build_val_loader(config)
    metrics = compute_metrics(
        model, loader, device, info["image_size"], progress=False,
    )
    if metrics["real_n"] > 0:
        assert metrics["real_pred_disp_mean_px"] < 1.0, (
            f"untrained model predicted large displacement: "
            f"{metrics['real_pred_disp_mean_px']}"
        )


# ---------------------------------------------------------------------------
# dump_visualizations
# ---------------------------------------------------------------------------


def test_dump_visualizations_writes_expected_files(fake_eval_setup, tmp_path: Path):
    from src.models.refiner.dataset import RefinerDataset

    config, ckpt_path = fake_eval_setup
    device = torch.device("cpu")
    model, info = build_model_from_checkpoint(ckpt_path, device)

    vis_ds = RefinerDataset(
        data_root=config["data"]["data_root"],
        video_names=config["data"]["val_videos"],
        image_size=tuple(config["data"]["image_size"]),
        pairs_per_track=4,
        real_pair_fraction=1.0,
        photometric_strength=0.0,
    )
    out_dir = tmp_path / "vis"
    dump_visualizations(
        model, vis_ds, device, info["image_size"], out_dir, n_vis=4,
    )
    assert (out_dir / "strips.png").exists()
    assert (out_dir / "edge_overlays.png").exists()
    assert (out_dir / "manifest.json").exists()
    # Blink GIFs: one per sample.
    gifs = list((out_dir / "blink").glob("*.gif"))
    assert len(gifs) == 4


# ---------------------------------------------------------------------------
# evaluate_checkpoint end-to-end
# ---------------------------------------------------------------------------


def test_evaluate_checkpoint_end_to_end(fake_eval_setup, tmp_path: Path):
    config, ckpt_path = fake_eval_setup
    out_dir = tmp_path / "eval_out"
    result = evaluate_checkpoint(
        ckpt_path, config, out_dir, n_vis=3, device=torch.device("cpu"),
        progress=False,
    )
    assert "info" in result
    assert "metrics" in result
    assert (out_dir / "strips.png").exists()
