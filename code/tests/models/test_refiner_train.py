"""Smoke + overfitting tests for ``src.models.refiner.train``.

The overfitting test is the minimum "everything connects" signal for Step
1.6: if the network can't drive loss toward zero on 8 synthetic samples in
a few dozen steps, something is fundamentally wrong — data, losses, warp,
or gradients.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.models.refiner.train import (
    real_pair_fraction_from_schedule,
    seed_worker,
    set_seed,
    train,
)

# ---------------------------------------------------------------------------
# Schedule interpolation
# ---------------------------------------------------------------------------

def test_schedule_empty_returns_zero():
    assert real_pair_fraction_from_schedule(0, {}) == 0.0
    assert real_pair_fraction_from_schedule(100, {}) == 0.0


def test_schedule_holds_before_first_and_after_last():
    sched = {5: 0.2, 20: 0.8}
    assert real_pair_fraction_from_schedule(0, sched) == 0.2
    assert real_pair_fraction_from_schedule(4, sched) == 0.2
    assert real_pair_fraction_from_schedule(25, sched) == 0.8


def test_schedule_linearly_interpolates():
    sched = {0: 0.0, 10: 1.0}
    assert real_pair_fraction_from_schedule(0, sched) == 0.0
    assert real_pair_fraction_from_schedule(5, sched) == pytest.approx(0.5)
    assert real_pair_fraction_from_schedule(10, sched) == 1.0


def test_schedule_piecewise_linear():
    sched = {0: 0.0, 10: 0.2, 20: 0.5, 35: 0.8}
    # In the 10->20 segment, t=5 -> halfway -> 0.35
    assert real_pair_fraction_from_schedule(15, sched) == pytest.approx(0.35)
    # In the 20->35 segment, t=0 -> 0.5
    assert real_pair_fraction_from_schedule(20, sched) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------

def test_set_seed_makes_rng_deterministic():
    set_seed(123)
    a = torch.rand(4).tolist()
    b = np.random.rand(4).tolist()
    set_seed(123)
    c = torch.rand(4).tolist()
    d = np.random.rand(4).tolist()
    assert a == c
    assert b == d


def test_seed_worker_does_not_crash():
    # Just verify the function is callable with an arbitrary id — it reads
    # torch.initial_seed() which is always defined in the main process.
    seed_worker(0)


# ---------------------------------------------------------------------------
# Fake dataset fixture
# ---------------------------------------------------------------------------

def _make_tiny_dataset(root: Path, n_frames: int = 8, h: int = 80, w: int = 160) -> None:
    """Create a single video with a single track of ``n_frames`` distinct frames.

    Each frame gets a simple geometric pattern so synthetic warps produce a
    learnable signal.
    """
    tdir = root / "vid0" / "track_00_FOO"
    tdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        base = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        # Draw a few bright vertical bars so the image has clear edges —
        # pure random noise would give weak gradient signal.
        base[:, 20::32] = 255
        base[10:-10, :] = np.clip(base[10:-10, :] + 40, 0, 255)
        Image.fromarray(base).save(tdir / f"frame_{i:06d}.png")


@pytest.fixture
def tiny_config(tmp_path: Path) -> dict:
    """Minimal training config pointing at a tiny on-disk dataset.

    Train and val both read from the same ``vid0`` — overfitting tests want
    to verify the model can *memorize* the training distribution, not
    generalize to held-out content. The ``allow_train_val_overlap`` flag
    bypasses the production safety check.
    """
    _make_tiny_dataset(tmp_path)
    return {
        "seed": 0,
        "device": "cpu",
        "progress": False,
        "data": {
            "data_root": str(tmp_path),
            "train_videos": ["vid0"],
            "val_videos": ["vid0"],
            "allow_train_val_overlap": True,
            "image_size": [64, 128],
            "pairs_per_track": 8,
            "syn_perturbation_px": 6.0,
            "photometric_strength": 0.0,
            "min_track_length": 2,
            "cache_in_ram": True,
        },
        "model": {
            "base_channels": 32,
            "dropout": 0.0,
            "head_init_scale": 1.0e-3,
        },
        "loss": {
            "corner": 1.0,
            "recon": 0.0,  # pure corner regression — tighter signal
            "ncc": 0.0,
            "grad": 0.0,
            "reg": 0.0,
        },
        "training": {
            "batch_size": 8,
            "num_workers": 0,
            "epochs": 150,
            "lr": 3.0e-3,
            "weight_decay": 0.0,
            "eta_min": 1.0e-4,
            "warmup_epochs": 5,
            "grad_clip": 1.0,
            "real_pair_schedule": {0: 0.0},  # synthetic only
        },
        "checkpoint": {
            "out_dir": str(tmp_path / "ckpts"),
            "save_every_epochs": 1000,
        },
    }


# ---------------------------------------------------------------------------
# Smoke + overfitting
# ---------------------------------------------------------------------------

def test_train_runs_end_to_end(tiny_config: dict):
    """Two-epoch smoke test: train() completes without errors and writes
    a checkpoint + training log."""
    tiny_config["training"]["epochs"] = 2
    result = train(tiny_config)
    assert len(result["history"]) == 2
    ckpt_dir = Path(tiny_config["checkpoint"]["out_dir"])
    assert (ckpt_dir / "refiner_last.pt").exists()
    assert (ckpt_dir / "train_log.json").exists()


def test_train_overfits_tiny_synthetic_set(tiny_config: dict):
    """Headline: with only 8 synthetic samples, training corner loss should
    drop sharply from init to end of training.

    This pins every piece of the pipeline: dataset -> model -> losses ->
    differentiable warp -> optimizer. If any link breaks silently the
    training loss can't descend and this test fails loudly.

    Note: we assert on *training* loss, not validation. With only 8 source
    images and fresh random warps each __getitem__, val measures
    generalization to unseen warps — which needs much more data/epochs.
    Overfitting (training loss → 0) is the right signal for "the pipeline
    works end-to-end."
    """
    result = train(tiny_config)
    history = result["history"]
    first_train = history[0]["train"]["corner"]
    last_train = history[-1]["train"]["corner"]
    first_val = history[0]["val"]["corner_err_px"]
    last_val = history[-1]["val"]["corner_err_px"]

    # Training loss must descend substantially — proves gradients flow.
    assert last_train < 1.0, (
        f"expected train corner loss < 1.0 px after 150 epochs, "
        f"got first={first_train:.3f}, last={last_train:.3f}"
    )
    assert last_train < 0.4 * first_train, (
        f"training did not meaningfully reduce train corner loss: "
        f"first={first_train:.3f}, last={last_train:.3f}"
    )
    # Validation should also show a clear learning signal, even if noisier.
    assert last_val < 0.7 * first_val, (
        f"no generalization signal: val first={first_val:.3f}, last={last_val:.3f}"
    )


def test_train_val_overlap_guard_rejects_by_default(tiny_config: dict):
    """The production safety check must fire unless explicitly bypassed."""
    tiny_config["data"]["allow_train_val_overlap"] = False
    tiny_config["training"]["epochs"] = 1
    with pytest.raises(ValueError, match="train and val videos overlap"):
        train(tiny_config)


def test_train_resume_continues_from_checkpoint(tiny_config: dict):
    """Resume path: a second run with --resume continues training and
    inherits the best_metric from the prior run."""
    tiny_config["training"]["epochs"] = 3
    result = train(tiny_config)
    first_best = result["best_metric"]

    ckpt_dir = Path(tiny_config["checkpoint"]["out_dir"])
    resume_path = ckpt_dir / "refiner_last.pt"
    assert resume_path.exists()

    tiny_config["resume"] = str(resume_path)
    tiny_config["training"]["epochs"] = 5
    result2 = train(tiny_config)
    # Should have run exactly 2 more epochs (3 -> 5)
    assert len(result2["history"]) == 2
    # Best metric should only have improved or stayed the same
    assert result2["best_metric"] <= first_best + 1e-6


def test_train_init_from_loads_weights_only(tiny_config: dict):
    """--init-from path: only model weights are loaded; optimizer, scheduler,
    epoch counter, and best metric are fresh.

    Pins the Stage 1 -> Stage 2 transition contract: the new config's LR
    and epoch count must take effect cleanly.
    """
    tiny_config["training"]["epochs"] = 3
    result = train(tiny_config)
    ckpt_dir = Path(tiny_config["checkpoint"]["out_dir"])
    first_ckpt = ckpt_dir / "refiner_last.pt"
    assert first_ckpt.exists()

    # Second run: init from the first checkpoint, new out_dir, new epochs.
    second_out = ckpt_dir.parent / "stage2_ckpts"
    tiny_config["init_from"] = str(first_ckpt)
    tiny_config["checkpoint"]["out_dir"] = str(second_out)
    tiny_config["training"]["epochs"] = 4  # full 4 epochs from scratch
    result2 = train(tiny_config)
    # Fresh epoch counter -> full 4 epochs, not (4 - 3 = 1).
    assert len(result2["history"]) == 4, (
        f"expected 4 epochs with init_from, got {len(result2['history'])}"
    )
    # Best metric should be recomputed from scratch (not inherited).
    assert result2["best_metric"] != float("inf")
    # The new model weights should start close to the first run's weights —
    # so the first epoch's loss shouldn't be catastrophically different from
    # the first run's final training loss.
    last_first = result["history"][-1]["train"]["total"]
    first_second = result2["history"][0]["train"]["total"]
    assert first_second < 3.0 * last_first, (
        f"init_from didn't carry weights: first run ended at {last_first:.3f}, "
        f"second run started at {first_second:.3f}"
    )


def test_train_rejects_both_resume_and_init_from(tiny_config: dict):
    tiny_config["resume"] = "/nonexistent1.pt"
    tiny_config["init_from"] = "/nonexistent2.pt"
    tiny_config["training"]["epochs"] = 1
    with pytest.raises(ValueError, match="either 'resume' or 'init_from'"):
        train(tiny_config)
