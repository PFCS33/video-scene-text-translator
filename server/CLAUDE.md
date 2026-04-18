# server/ — FastAPI backend for the live demo

Nested CLAUDE.md — auto-loads when working in `server/`. Root CLAUDE.md covers
pipeline conventions; this file only covers what's specific to the web
backend. Don't duplicate the root.

## Scope
FastAPI app that wraps the existing `VideoPipeline` (from `code/src/`) for
browser-based use: one video upload → one pipeline run → live SSE progress →
MP4 download. Single `ThreadPoolExecutor(max_workers=1)`, in-memory job
dict, no persistence across restarts. Same-origin static SPA mount; no CORS.

## Run
```bash
# activate env first (see root CLAUDE.md)
eval "$(/opt/miniconda3/bin/conda shell.bash hook)" && conda activate vc_final

# dev (uvicorn + Vite in parallel, Vite proxies /api to :8000)
./server/scripts/dev.sh

# prod/demo (after ./server/scripts/build_frontend.sh)
python -m uvicorn server.app.main:app --host 0.0.0.0 --port 8000

# tests
cd server && python -m pytest tests/ -v        # default: 82 unit/integ tests, no GPU
cd server && python -m pytest tests/ -v -m gpu # real pipeline + AnyText2, 3 tests
```

## Architecture summary
- `JobManager` (`jobs.py`) owns the job registry + the single worker thread.
- One `ThreadPoolExecutor(max_workers=1)` — a second `POST /api/jobs` while
  another is queued/running raises `ConcurrentJobError` → 409. (D4)
- Per-job `asyncio.Queue` populated from the worker thread via
  `loop.call_soon_threadsafe(queue.put_nowait, event)`. The queue lives on
  the event loop; only the loop reads from it.
- `PipelineRunner` is a `Protocol` (`jobs.py`) — the real impl is
  `run_pipeline_job` in `pipeline_runner.py`. Tests inject a stub callable.
- **D16 terminal-state-flip invariant:** the `emit` closure in
  `JobManager._run_job` mutates `record.status` to `succeeded` / `failed`
  *before* calling `_emit_threadsafe`, so a client that races from SSE
  `DoneEvent` to `GET /status` never observes a stale `"running"`.
- `main.py` uses `@asynccontextmanager lifespan` to: sweep stale job dirs
  (>2h TTL), build the `JobManager` (captures the running loop), stash it
  on `app.state`, and override `get_manager` via `app.dependency_overrides`.
- Static SPA mount at `/` happens **after** `app.include_router(router)` —
  order matters; `StaticFiles` will shadow `/api/*` if mounted first.

## Key files
| file | role |
|------|------|
| `app/main.py` | FastAPI app, `lifespan` hook, static SPA mount |
| `app/routes.py` | 6 HTTP endpoints under `APIRouter(prefix="/api")` |
| `app/jobs.py` | `JobManager`, `PipelineRunner` protocol, `_JobRecord`, `ConcurrentJobError`, `UnknownJobError` |
| `app/pipeline_runner.py` | lazy-imports `VideoPipeline`; attaches `logging.Handler`; adapts `progress_callback`; ffmpeg transcode for browser-safe MP4 |
| `app/storage.py` | `storage_root()` (env-overridable), `uploads_dir`, `outputs_dir`, `cleanup_job`, `sweep_old_jobs` |
| `app/schemas.py` | Pydantic v2 models; SSE events with `extra="forbid"` |
| `app/languages.py` | curated 7-language list (D12) |

## Conventions
- `APIRouter(prefix="/api")` — every HTTP endpoint lives under `/api/*`.
  The bare `/` path is reserved for the SPA bundle. `GET /api/health` is the
  one liveness endpoint on `main.py` directly.
- Dependency injection via `Depends(get_manager)`. Tests swap in a
  stub-backed `JobManager` with `app.dependency_overrides[get_manager]`.
  The default `get_manager()` raises — wiring mistakes fail loudly, not
  silently.
- Pydantic v2 only: `BaseModel`, `Field`, `ConfigDict`, `model_validate`,
  `model_dump`, `field_validator`. No `parse_obj` / `dict()` / v1 `Config`.
- SSE events are sent via `sse_starlette.EventSourceResponse`:
  `yield {"event": event.type, "data": event.model_dump_json()}`. The
  event name is always the `type` field literal (`stage_start`,
  `stage_complete`, `log`, `done`, `error`).
- Lazy-import heavy pipeline deps (`torch`, `paddle`, `cv2`,
  `src.pipeline`, `src.config`) *inside* function bodies, never at module
  scope. Keeps `from server.app import *` + `import server.app.pipeline_runner`
  cheap and lets non-GPU tests import freely.
- `SERVER_STORAGE_ROOT` env var overrides `<repo>/server/storage/`.
  Re-read on every `storage_root()` call; tests use `monkeypatch.setenv`
  for isolation. `storage/` is gitignored.

## Gotchas
- `JobManager.__init__` calls `asyncio.get_running_loop()` to capture the
  loop for `call_soon_threadsafe`. Must be constructed from inside a
  running loop — i.e. FastAPI `lifespan` or a `@pytest.mark.asyncio` test
  (we use `asyncio_mode = auto` in `pytest.ini` so every test gets one).
  Never build it at module import time.
- `sse-starlette`'s `AppStatus.should_exit_event` is a module-global
  `anyio.Event` that binds lazily to the first event loop it sees. Multiple
  `TestClient` instances in the same session hit it. `test_integration.py`
  has an autouse fixture that resets it between tests — mirror that
  pattern if you add more SSE-heavy tests.
- Static mount order: `app.mount("/", StaticFiles(...))` **must** come
  after `app.include_router(router)`. If you flip them, `/api/*` returns
  404s because `StaticFiles` owns `/` and everything below it. See
  `_mount_spa()` in `main.py`.
- The progress-event terminal race (D16): the `emit` closure in
  `JobManager._run_job` flips `record.status` to its terminal value
  **before** enqueueing `DoneEvent` / `ErrorEvent`. A client that reads
  the terminal event and immediately polls `/status` must see the
  matching final state. Don't refactor this into "enqueue first, mutate
  later" — the test suite asserts on the order.
- The pipeline's `VideoWriter` fourcc is `FMP4` (MPEG-4 Part 2), which
  Chrome and Firefox refuse to play in `<video>`.
  `pipeline_runner._transcode_to_browser_safe` shells out to
  `ffmpeg -c:v libx264 -pix_fmt yuv420p -movflags +faststart` after
  `VideoPipeline.run()` returns, then atomic-swaps the file. `ffmpeg`
  must be on PATH (R3).

## Test strategy
- **Default tests** (82): fully mocked pipeline — stub `PipelineRunner`
  callable, no GPU, no AnyText2, no network. `pytest.ini` sets
  `addopts = -m "not gpu"` so these run on any box.
- **`gpu`-marked tests** (3, in `test_integration.py`): real pipeline
  against real AnyText2 at `text_editor.server_url` from `adv.yaml`.
  Requires CUDA + AnyText2 reachable. Skip guard inside each test
  catches missing deps. Run with `pytest -m gpu`.
- SSE tests: `TestClient.stream("GET", "/api/jobs/{id}/events")` +
  tiny line-based frame parser (`test_api.py::_parse_sse_stream`).
  No `httpx.ASGITransport` needed — path 2 from the Step 7 plan worked
  out of the box.
- Storage tests use `tmp_path` + `monkeypatch.setenv(SERVER_STORAGE_ROOT)`
  for isolation — don't write to the real `server/storage/`.

## Do NOT
- Don't add CORS middleware. Same-origin by design (D9). Dev mode uses
  Vite's proxy; prod serves the SPA from FastAPI itself.
- Don't hold the `JobManager` in a module-level variable. It lives on
  `app.state.job_manager`, built in `lifespan`, because `__init__` needs
  a running loop. A module-level singleton will crash at import.
- Don't import `VideoPipeline` / `PipelineConfig` at the top of
  `pipeline_runner.py`. Keep the lazy imports inside functions so the
  FastAPI boot path stays free of `torch` / `paddle` / `cv2`.
- Don't return 500 for business-logic errors. Map:
  `ConcurrentJobError` → 409 (with `active_job_id` in the detail body for
  R8 rejoin UX), `UnknownJobError` → 404, unsupported language → 400,
  oversize upload → 413.
- Don't add new endpoints outside `APIRouter(prefix="/api")`. The root
  path is owned by the SPA static mount.

## References
- `/workspace/video-scene-text-translator/plan.md` — decisions D1–D19, risks R1–R10
- `/workspace/video-scene-text-translator/docs/architecture.md` — Web Application section
- Root `/workspace/video-scene-text-translator/CLAUDE.md` — pipeline conventions, conda env
