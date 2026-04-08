# Session: LaMa Inpainter — 2026-04-08

## Completed
- Analyzed S4 pipeline regression: SRNet inpainting artifacts degrade LCM lighting ratios (washed-out results on textured backgrounds)
- Confirmed from STRIVE paper that SRNet IS required by the paper's TPM for background inpainting in LCM
- Researched 5 scene text removal alternatives (CTRNet, ViTEraser, GaRNet, FETNet, PERT) + LaMa general inpainter
- Evaluated RS-STE (CVPR 2025) — confirmed it has NO background branch (unified architecture), cannot replace SRNet's inpainting role
- Selected LaMa as the best replacement: Apache 2.0, 206MB TorchScript, pip-compatible, broad community
- Wrote implementation plan in `plan.md` on branch `feat/lama-inpainter`

## Current State
- Branch `feat/lama-inpainter` created from latest master (includes all BPN/S4/test-reorg work)
- `plan.md` contains the 8-step implementation plan
- No code changes yet — implementation to be done on remote GPU machine

## Next Steps
1. Implement plan steps 1-8 on remote (extend ABC, create LaMa wrapper, update stage.py, tests)
2. Download `big-lama.pt` via install script on remote
3. Run e2e comparison: LaMa vs SRNet on the Scrabble tile video
4. If LaMa produces better LCM ratios, make it the default in `adv.yaml`

## Decisions Made
- **LaMa over GaRNet**: LaMa has better texture synthesis (Fourier convolutions), broader training data (Places365), pip-installable, 8k+ stars. GaRNet is text-specific but smaller community
- **Extend ABC with optional mask** (not auto-generate inside wrapper): User preferred principled interface extension. S4 generates mask via Otsu thresholding, passes it. SRNet ignores; LaMa uses.
- **Direct TorchScript loading** (not pip package): Consistent with SRNet/BPN checkpoint pattern. `torch.jit.load()` + install script.
- **Upscale small ROIs to 256px**: LaMa trained at 256x256, our ROIs can be 64px. Upscale before, downscale after.

## Open Questions
- What expansion ratio works best with LaMa vs SRNet? Needs e2e comparison
- Does LCM + LaMa + BPN together improve or hurt quality? BPN may still over-blur
- Consider disabling BPN entirely if LaMa produces clean enough backgrounds
