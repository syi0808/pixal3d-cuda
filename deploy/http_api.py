"""Bounded HTTP transport, exposed only behind Modal Proxy Token authentication."""

import asyncio
import json
import logging
from pathlib import Path
import tempfile
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException
from starlette.responses import FileResponse, JSONResponse


class DailyLimitExceeded(Exception):
    def __init__(self, retry_after):
        self.retry_after = retry_after
        super().__init__("Daily new-job admission limit reached; retry after the UTC day changes")


class JobOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    seed: int = Field(default=42, ge=0, le=2 ** 63 - 1)
    geometry_resolution: Literal[1024, 1536] = 1024
    texture_size: Literal[512, 1024, 2048, 4096] = 4096
    target_faces: int = Field(default=1000000, ge=4, le=1_000_000)
    camera: Literal["auto", "fixed"] = "auto"
    fov_degrees: float = Field(default=49.13, ge=1, le=150)
    remove_background: bool = False
    reference: bool = False
    attempt: str = Field(default="", max_length=64, pattern=r"^[A-Za-z0-9_.-]*$")


class BodyLimitMiddleware:
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        length = dict(scope["headers"]).get(b"content-length")
        if length is not None:
            try:
                size = int(length)
                if size < 0:
                    raise ValueError
            except ValueError:
                return await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(scope, receive, send)
            if size > self.limit:
                return await JSONResponse({"detail": "Request body limit exceeded"}, status_code=413)(scope, receive, send)
        consumed = 0

        async def limited_receive():
            nonlocal consumed
            message = await receive()
            consumed += len(message.get("body", b""))
            if consumed > self.limit:
                scope.setdefault("state", {})["body_limit_exceeded"] = True
                raise MultiPartException("Request body limit exceeded")
            return message

        return await self.app(scope, limited_receive, send)


def public_job(record):
    visible = {name: record[name] for name in ("job_id", "status", "updated_at", "error", "bytes", "sha256") if name in record}
    visible["status_url"] = f"/api/jobs/{record['job_id']}"
    visible["glb_url"] = f"/api/jobs/{record['job_id']}/result.glb"
    visible["report_url"] = f"/api/jobs/{record['job_id']}/report.json"
    return visible


def create_app(backend, settings):
    """Create the application; production authentication is enforced at Modal's edge."""
    app = FastAPI(title="Pixal3D CUDA", description="Official Pixal3D: asynchronous native 1024/1536 PBR generation")
    body_limit = settings.max_total_image_bytes + settings.max_transforms_bytes + 128 * 1024
    app.add_middleware(BodyLimitMiddleware, limit=body_limit)
    storage_lock = asyncio.Lock()

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request, error):
        if getattr(request.state, "body_limit_exceeded", False):
            return JSONResponse({"detail": "Request body limit exceeded"}, status_code=413)
        return JSONResponse({"detail": error.detail}, status_code=error.status_code, headers=error.headers)

    @app.get("/api/health")
    async def health():
        from pixalcuda.upstream_sources import lock
        from pixalcuda.optimizations import OptimizationOptions
        from dataclasses import asdict
        return {"status": "ok", "runtime": "official", "environment": settings.environment, "geometry_resolution": 1024,
                "supported_geometry_resolutions": [1024, 1536],
                "upstream_revision": lock()["sources"]["pixal3d"]["revision"], "low_vram": settings.low_vram,
                "attention_backend": settings.attention_backend, "gpu": settings.gpu, "max_tokens": settings.max_tokens,
                "background_removal_default": False, "rmbg_enabled": settings.allow_rmbg,
                "skip_weight_init": settings.skip_weight_init, "inpaint_workers": settings.inpaint_workers,
                "gpu_cpu_cores": settings.gpu_cpu_cores,
                "optimizations": asdict(OptimizationOptions(settings.share_dino, settings.dino_cache_mb,
                                                            settings.sparse_projection, settings.conditioning_cache_mb)),
                "max_views": settings.max_views, "max_new_jobs_per_utc_day": settings.api_max_jobs_per_day}

    @app.post("/api/jobs", status_code=202, openapi_extra={
        "requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
            "type": "object", "properties": {
                "image": {"type": "string", "format": "binary", "description": "One unposed image; mutually exclusive with images"},
                "images": {"type": "array", "items": {"type": "string", "format": "binary"}, "description": "Ordered images mapped by transforms.image_index"},
                "transforms": {"type": "string", "description": "Camera JSON text, required with images"},
                "options": {"type": "string", "description": "Generation options as JSON text", "default": "{}"},
            },
        }}}},
    })
    async def submit(request: Request):
        async with request.form(max_files=settings.max_views + 1, max_fields=2,
                                max_part_size=settings.max_transforms_bytes) as form:
            if any(name not in ("image", "images", "transforms", "options") for name in form):
                raise HTTPException(422, "Unknown multipart field")
            singles, mapped = form.getlist("image"), form.getlist("images")
            if ((not singles) == (not mapped) or len(singles) > 1
                    or len(mapped) > settings.max_views):
                raise HTTPException(422, "Supply one image, or mapped images with one camera each, not both")
            uploads = mapped or singles
            if any(not isinstance(upload, UploadFile) for upload in uploads):
                raise HTTPException(422, "Images must be multipart file uploads")
            if any(len(form.getlist(name)) > 1 for name in ("options", "transforms")):
                raise HTTPException(422, "options and transforms must each be submitted once")
            for name in ("options", "transforms"):
                value = form.get(name)
                if value is not None and (not isinstance(value, str) or len(value.encode()) > settings.max_transforms_bytes):
                    raise HTTPException(422, f"{name} must be a bounded JSON text field, not a file")
            try:
                options = JobOptions.model_validate(json.loads(form.get("options", "{}")))
            except (ValueError, ValidationError):
                raise HTTPException(422, "Invalid options JSON or unsupported generation settings") from None
            try:
                settings.validate_background_removal(options.remove_background)
            except ValueError as error:
                raise HTTPException(422, str(error)) from None
            transforms = form.get("transforms")
            if bool(mapped) != (transforms is not None):
                raise HTTPException(422, "Mapped images require transforms; single image must not include transforms")
            if mapped and options.model_fields_set.intersection(("camera", "fov_degrees")):
                raise HTTPException(422, "Mapped cameras supply their own FOV; omit single-image camera options")
            if mapped:
                try:
                    transforms = json.loads(transforms)
                except ValueError:
                    raise HTTPException(422, "Invalid camera JSON") from None
            with tempfile.TemporaryDirectory(prefix="pixal-upload-") as temporary:
                images, total = [], 0
                for index, upload in enumerate(uploads):
                    destination = Path(temporary) / f"image-{index}.png"
                    size = 0
                    with destination.open("wb") as target:
                        while chunk := await upload.read(1024 * 1024):
                            size += len(chunk)
                            total += len(chunk)
                            if size > settings.max_upload_bytes or total > settings.max_total_image_bytes:
                                raise HTTPException(413, "Per-image or aggregate image limit exceeded")
                            target.write(chunk)
                    images.append(destination)
                try:
                    async with storage_lock:
                        record = await run_in_threadpool(backend.submit, images, transforms, options.model_dump())
                except DailyLimitExceeded as error:
                    raise HTTPException(429, str(error), headers={"Retry-After": str(error.retry_after)}) from None
                except (ValueError, OSError, Image.DecompressionBombError) as error:
                    raise HTTPException(422, str(error)) from None
                except Exception:
                    logging.getLogger(__name__).exception("HTTP job submission failed")
                    raise HTTPException(503, "Job submission failed; inspect Modal logs before retrying") from None
            return JSONResponse(public_job(record), status_code=202,
                                headers={"Location": f"/api/jobs/{record['job_id']}"})

    async def find_job(job_id):
        if len(job_id) != 64 or any(character not in "0123456789abcdef" for character in job_id):
            raise HTTPException(404, "Unknown job")
        record = await run_in_threadpool(backend.status, job_id)
        if record is None:
            raise HTTPException(404, "Unknown job")
        return record

    @app.get("/api/jobs/{job_id}")
    async def status(job_id: str):
        return public_job(await find_job(job_id))

    async def artifact(job_id, filename, media_type):
        record = await find_job(job_id)
        allowed = ("complete", "failed") if filename == "report.json" else ("complete",)
        if record["status"] not in allowed:
            raise HTTPException(409, f"Result is not ready: {record['status']}")
        temporary = tempfile.TemporaryDirectory(prefix="pixal-download-")
        destination = Path(temporary.name) / filename
        try:
            async with storage_lock:
                await run_in_threadpool(backend.copy_artifact, job_id, filename, destination)
        except FileNotFoundError:
            temporary.cleanup()
            if record["status"] == "failed":
                raise HTTPException(404, "No runtime report was produced for this failed job") from None
            raise HTTPException(410, "Artifact is no longer retained") from None
        except BaseException:
            temporary.cleanup()
            raise
        return FileResponse(destination, media_type=media_type, filename=filename,
                            background=BackgroundTask(temporary.cleanup))

    @app.get("/api/jobs/{job_id}/result.glb")
    async def result(job_id: str):
        return await artifact(job_id, "result.glb", "model/gltf-binary")

    @app.get("/api/jobs/{job_id}/report.json")
    async def report(job_id: str):
        return await artifact(job_id, "report.json", "application/json")

    schema = app.openapi()
    schema.setdefault("components", {}).setdefault("securitySchemes", {}).update({
        "ModalKey": {"type": "apiKey", "in": "header", "name": "Modal-Key"},
        "ModalSecret": {"type": "apiKey", "in": "header", "name": "Modal-Secret"},
    })
    schema["security"] = [{"ModalKey": [], "ModalSecret": []}]
    return app
