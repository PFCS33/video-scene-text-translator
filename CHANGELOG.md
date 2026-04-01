# Changelog

## 2026-04-01 — CoTracker Integration (experiment/cotracker)

- Add `flow_fill_strategy` config option: `gaps_only` (original) vs `full_propagation` (overwrite all OCR quads with optical-flow-tracked quads from reference frame)
- Integrate Meta CoTracker3 as a new `optical_flow_method` option, replacing pairwise Farneback/LK with batch point tracking (~25x faster, smoother trajectories)
- Add timing logs for OCR detection and optical flow steps in Stage 1
- Add `third_party/install_cotracker.sh` for cloning and installing CoTracker
