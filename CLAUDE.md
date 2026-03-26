# Cross-Language Scene Text Replacement in Video
CMPT 743 Visual Computing Lab II final project (SFU). Replace scene text in video frames across languages, preserving font style, perspective, and lighting consistency. Team: Hebin Yao, Yunshan Feng, Liliana Lopez.

## Workflow

### Session
1. At session start, run /load-context to load context from the previous session.
   Older session history is in docs/sessions/ — read when you need context beyond the last session.
2. At session end when I ask, run /session-summary to archive the session.

### Development
When starting a new feature:
1. Create a feature branch: feat/, fix/, chore/
2. Run /architect if the project has no docs/architecture.md yet.
3. Run /plan to brainstorm and write plan.md for this feature. Wait for approval.
   If plan.md already exists for this feature, load it and continue from where it left off.
4. If a design decision needs research, delegate to @researcher.

When implementing:
5. For scoped module work, delegate to @coder with the specific plan step.
6. If @coder reports unresolved test failures, delegate to @debugger with the error output.
7. After completing each plan step, mark it as [x] in plan.md Progress and note any changes.

When wrapping up:
8. Delegate to @reviewer for code review.
9. Commit changes — the commit skill will propose atomic splits for approval.
10. When merging a feature branch to main, check if docs/architecture.md needs updating to reflect what was actually built.

## Commands
dev:    python scripts/run_pipeline.py --input <video> --output <out> --source-lang en --target-lang es
test:   cd code && python -m pytest tests/ -v
lint:   ruff check code/
build:  (N/A — not a distributable package)

## Stack
- Python 3.11 (conda env: `vc_final`)
- OpenCV (cv2) — core CV operations, homography, optical flow
- NumPy — array operations
- PyYAML — config loading
- Pillow — image I/O
- EasyOCR — scene text detection (not yet installed)
- googletrans 4.0.0-rc1 — translation API (not yet installed)
- pytest + pytest-cov — testing (105 tests)
- ruff — linting and formatting

## Key Directories
- `code/src/` — Pipeline implementation (5 stages)
- `code/src/stages/` — S1 detection, S2 frontalization, S3 text editing, S4 propagation, S5 revert
- `code/src/models/` — Stage A model interface (BaseTextEditor ABC) + backends
- `code/src/utils/` — Geometry, image processing, optical flow utilities
- `code/config/` — default.yaml (all pipeline parameters)
- `code/tests/` — 105 unit + integration tests
- `code/scripts/` — CLI entry point (run_pipeline.py)
- `_refs/` — Pipeline diagram, milestone report
- `docs/` — Architecture docs, session summaries

## Conventions
- All configurable values live in `config/default.yaml`, never hardcoded
- Stages communicate via `TextTrack` dataclass — the central data structure flowing through S1→S5
- Stage A models implement `BaseTextEditor` ABC — swap backends via `text_editor.backend` in config
- Lazy initialization for expensive resources (EasyOCR, translator) — never import at module level
- Detections keyed by `frame_idx` (dict, not list) for O(1) lookup
- Activate conda before any command: `eval "$(/opt/miniconda3/bin/conda shell.bash hook)" && conda activate vc_final`
- Domain-specific rules auto-load from .claude/rules/ when working in matching paths

## Gotchas
- Never import easyocr or googletrans at module level — they're lazy-loaded and may not be installed
- Always activate conda env before running tests or pipeline
- All frames loaded into memory — will break on long videos (>500 frames)
- googletrans is unofficial and may fail silently — always verify translation output
- config weight arrays must sum to ~1.0 (detection: 4 weights, reference: 2 weights) — validation catches this

## Git
- Never push directly to main
- Commit format: type(scope): description — e.g., feat(stageb): add histogram matching
- Run tests and lint before committing: `cd code && python -m pytest tests/ -v && ruff check code/`

## Reference Docs
- For architecture decisions, see docs/architecture.md
- Pipeline diagram: _refs/pipeline.png
- Milestone report: _refs/report.pdf
