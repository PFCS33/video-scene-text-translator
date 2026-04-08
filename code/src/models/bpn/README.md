File	Purpose
init.py	Package exports
model.py	BPN network — ResNet18 backbone with modified first conv (3*(N+1) channels) + GAP + 2 FC layers → 4N blur parameters (sigma_x, sigma_y, rho, w) per neighbor
blur.py	Differentiable oriented 2D Gaussian blur — builds rotated anisotropic kernel from predicted params, applies via grouped conv2d. Model: I_out = (1+w)*I_ref - w*(I_ref * G)
dataset.py	BPNDataset — sliding window over track frames. Supports video_indices and max_tracks_per_video for easy subsetting
losses.py	BPNLoss — L_psi (parameter regression, Stage 1), L_R (reconstruction MSE), L_T (temporal consistency)
train.py	Training loop — Stage 1 (synthetic blur with known GT params) + Stage 2 (self-supervised on real data). Adam optimizer, cosine LR, gradient clipping, checkpointing
evaluate.py	Evaluation — quantitative metrics (MSE, param stats) + visual comparisons (ref | target | predicted | diff) + training curve plots
config.yaml	Full training config
config_test.yaml	Quick-test config (video 1 train, video 2 val, 20 tracks, 3 epochs)

# Usage

cd code

# Quick test (20 tracks, 3 epochs)
.venv/bin/python -m src.models.bpn.train --config src/models/bpn/config_test.yaml

# Full Stage 1 training (synthetic blur supervision)
.venv/bin/python -m src.models.bpn.train --config src/models/bpn/config.yaml --stage 1

# Stage 2 fine-tuning (self-supervised on real data)
.venv/bin/python -m src.models.bpn.train --config src/models/bpn/config.yaml --stage 2 --resume checkpoints/bpn/bpn_stage1_best.pt

# Evaluation + visualization
.venv/bin/python -m src.models.bpn.evaluate --config src/models/bpn/config.yaml --checkpoint checkpoints/bpn/bpn_stage1_best.pt
