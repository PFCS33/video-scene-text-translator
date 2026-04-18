"""HTTP routes wiring JobManager + PipelineRunner + storage into FastAPI.

Six endpoints (plan.md API surface):

    POST   /api/jobs                 — upload + kick off pipeline
    GET    /api/jobs/{id}/status     — snapshot of job state
    GET    /api/jobs/{id}/events     — SSE stream of pipeline events (D5)
    GET    /api/jobs/{id}/output     — download the final MP4
    DELETE /api/jobs/{id}            — remove a terminal job + its files (D14)
    GET    /api/languages            — curated dropdown list (D12)

``GET /api/health`` stays in ``main.py``.

Design notes
------------

* ``JobManager`` is injected via FastAPI's dependency system (``get_manager``).
  ``main.py`` overrides the provider in its ``lifespan`` hook so tests can
  swap in a manager backed by a stub runner without touching the real
  pipeline (see ``server/tests/test_api.py``).

* The upload is streamed in 1 MiB chunks; we allocate the job's UUID *before*
  writing to disk so the upload lands under its final
  ``storage.uploads_dir(job_id)`` path. ``JobManager.submit`` accepts the
  pre-allocated id via its ``job_id`` kwarg — this avoids the
  "rename storage dirs after submit" dance that would otherwise be needed.

* Uploads are capped at ``MAX_UPLOAD_BYTES`` (R2) with a running byte
  counter; over-limit requests get a 413 and their partial file is cleaned
  up before returning. The constant is a module attribute so tests can
  monkeypatch it to small values.

* Concurrency: if ``JobManager.submit`` raises ``ConcurrentJobError``, we
  respond ``409`` with ``{"error": "concurrent_job", "active_job_id": <id>}``
  in the detail body so the frontend can render "rejoin existing run"
  (R8).
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from . import storage
from .jobs import ConcurrentJobError, JobManager, UnknownJobError
from .languages import SUPPORTED_LANGUAGES, Language, is_supported
from .schemas import JobCreateResponse, JobStatus

logger = logging.getLogger(__name__)

# 200 MiB upload cap (plan.md R2). Module-level so tests can monkeypatch.
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

# Read chunk size for the multipart upload streaming loop. 1 MiB is a
# standard sweet spot between memcpy overhead and syscall count.
_UPLOAD_CHUNK = 1024 * 1024


router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Dependency providers — swappable in tests via app.dependency_overrides.
# ---------------------------------------------------------------------------


def get_manager() -> JobManager:
    """Return the application-lifetime ``JobManager`` instance.

    ``main.py``'s lifespan hook overrides this provider so it returns
    ``app.state.job_manager``. The default raises to make wiring mistakes
    immediately obvious rather than surfacing as an ``AttributeError``
    deep inside a request.
    """
    raise RuntimeError(
        "get_manager must be overridden — main.py's lifespan hook wires "
        "app.dependency_overrides[get_manager] to app.state.job_manager"
    )


# ---------------------------------------------------------------------------
# GET /api/languages
# ---------------------------------------------------------------------------


@router.get("/languages", response_model=list[Language])
def list_languages() -> list[Language]:
    """Curated source/target languages for the dropdown (plan.md D12)."""
    return SUPPORTED_LANGUAGES


# ---------------------------------------------------------------------------
# POST /api/jobs — multipart upload + kick off pipeline
# ---------------------------------------------------------------------------


@router.post("/jobs", response_model=JobCreateResponse)
async def create_job(
    video: Annotated[UploadFile, File(description="input video (mp4/mov/...)")],
    source_lang: Annotated[str, Form()],
    target_lang: Annotated[str, Form()],
    manager: Annotated[JobManager, Depends(get_manager)],
) -> JobCreateResponse:
    """Accept a video upload, stream it to storage, and submit the pipeline."""
    # Validate language codes up front — saves writing the upload to disk if
    # the request is already malformed.
    if not is_supported(source_lang):
        raise HTTPException(
            status_code=400, detail=f"unsupported source_lang: {source_lang}"
        )
    if not is_supported(target_lang):
        raise HTTPException(
            status_code=400, detail=f"unsupported target_lang: {target_lang}"
        )

    # Allocate the job_id first so the upload lands under its final storage
    # path. JobManager.submit() accepts an explicit job_id kwarg precisely
    # for this flow.
    job_id = str(uuid.uuid4())

    uploads = storage.uploads_dir(job_id)
    # `Path(...).name` strips any directory components — belt-and-braces
    # protection against path-traversal shenanigans in the client-supplied
    # filename. Fallback name keeps things sane if the client omits it.
    safe_name = Path(video.filename or "upload.mp4").name
    upload_path = uploads / safe_name

    total = 0
    try:
        with upload_path.open("wb") as fh:
            while chunk := await video.read(_UPLOAD_CHUNK):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    # Close + clean before raising; FastAPI's exception
                    # handler will never see the half-written file.
                    fh.close()
                    storage.cleanup_job(job_id)
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"upload exceeds {MAX_UPLOAD_BYTES} bytes"
                        ),
                    )
                fh.write(chunk)
    except HTTPException:
        raise
    except Exception:
        # Unexpected I/O error — nuke the partial upload so we don't leak
        # empty per-job directories.
        storage.cleanup_job(job_id)
        raise

    output_path = storage.outputs_dir(job_id) / "out.mp4"

    try:
        await manager.submit(
            job_id=job_id,
            source_lang=source_lang,
            target_lang=target_lang,
            input_path=upload_path,
            output_path=output_path,
        )
    except ConcurrentJobError as exc:
        # Roll back the storage we just allocated for this would-be job.
        storage.cleanup_job(job_id)
        raise HTTPException(
            status_code=409,
            detail={
                "error": "concurrent_job",
                "active_job_id": exc.args[0] if exc.args else None,
            },
        ) from exc

    return JobCreateResponse(job_id=job_id)


# ---------------------------------------------------------------------------
# GET /api/jobs/{job_id}/status
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}/status", response_model=JobStatus)
def get_job_status(
    job_id: str,
    manager: Annotated[JobManager, Depends(get_manager)],
) -> JobStatus:
    try:
        return manager.get_status(job_id)
    except UnknownJobError as exc:
        raise HTTPException(
            status_code=404, detail=f"unknown job: {job_id}"
        ) from exc


# ---------------------------------------------------------------------------
# GET /api/jobs/{job_id}/events — SSE stream (plan.md D5, D16)
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}/events")
async def stream_events(
    job_id: str,
    manager: Annotated[JobManager, Depends(get_manager)],
):
    """Subscribe to a job's SSE stream.

    The client `EventSource` reconnects automatically; events during the
    gap are lost (plan.md D16). Callers that need exact fidelity should
    poll ``/status`` after reconnect to resync.
    """
    try:
        manager.get_status(job_id)  # existence check
    except UnknownJobError as exc:
        raise HTTPException(
            status_code=404, detail=f"unknown job: {job_id}"
        ) from exc

    async def event_gen():
        async for event in manager.subscribe(job_id):
            yield {
                "event": event.type,
                "data": event.model_dump_json(),
            }

    return EventSourceResponse(event_gen())


# ---------------------------------------------------------------------------
# GET /api/jobs/{job_id}/output — download finished MP4
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}/output")
def download_output(
    job_id: str,
    manager: Annotated[JobManager, Depends(get_manager)],
):
    try:
        status = manager.get_status(job_id)
    except UnknownJobError as exc:
        raise HTTPException(
            status_code=404, detail=f"unknown job: {job_id}"
        ) from exc

    if not status.output_available:
        raise HTTPException(
            status_code=404,
            detail=(
                f"output for job {job_id} not available "
                f"(status={status.status})"
            ),
        )

    output_path = storage.outputs_dir(job_id) / "out.mp4"
    if not output_path.exists():
        # Sanity check — `output_available` already verified existence via
        # JobStatus, but a concurrent cleanup could have nuked it in the
        # microseconds between the two checks.
        raise HTTPException(status_code=404, detail="output file missing")

    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename="out.mp4",
        # `FileResponse(filename=...)` already sets Content-Disposition to
        # `attachment; filename="out.mp4"`, so we don't need to add it
        # manually.
    )


# ---------------------------------------------------------------------------
# DELETE /api/jobs/{job_id}
# ---------------------------------------------------------------------------


@router.delete("/jobs/{job_id}")
async def delete_job(
    job_id: str,
    manager: Annotated[JobManager, Depends(get_manager)],
):
    """Remove a terminal job from the registry and its storage.

    Returns 409 if the job is still running (MVP has no cancellation —
    see plan.md D14).
    """
    try:
        await manager.delete(job_id)
    except UnknownJobError as exc:
        raise HTTPException(
            status_code=404, detail=f"unknown job: {job_id}"
        ) from exc
    except ConcurrentJobError as exc:
        raise HTTPException(
            status_code=409,
            detail="job is still running and cannot be deleted",
        ) from exc

    storage.cleanup_job(job_id)
    return {"deleted": job_id, "ts": time.time()}
