"""Single-environment Modal workers and a proxy-authenticated HTTP API."""

import hashlib
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

import modal

from deploy.settings import load_settings


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_ROOT = Path("/weights")
JOBS_ROOT = Path("/jobs")
SETTINGS = load_settings()
GPU = SETTINGS.gpu

app = modal.App(SETTINGS.app_name)
weights = modal.Volume.from_name(SETTINGS.weights_volume, create_if_missing=True, version=2, environment_name=SETTINGS.environment)
jobs = modal.Volume.from_name(SETTINGS.jobs_volume, create_if_missing=True, version=2, environment_name=SETTINGS.environment)
records = modal.Dict.from_name(SETTINGS.jobs_dict, create_if_missing=True, environment_name=SETTINGS.environment)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libglib2.0-0", "libgl1")
    .pip_install("torch==2.6.0", "torchvision==0.21.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install_from_pyproject(str(ROOT / "pyproject.toml"), optional_dependencies=["preprocess", "camera", "server"])
    .env({"PYTHONPATH": "/opt/pixal3d:/opt/pixal3d/deploy", "HF_HOME": "/weights/hf", "U2NET_HOME": "/weights/u2net", "PIXAL_GPU": GPU,
          "MODAL_ENVIRONMENT": SETTINGS.environment, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(ROOT / "pixalcuda", "/opt/pixal3d/pixalcuda", copy=True, ignore=["__pycache__", "*.pyc"])
    .add_local_file(ROOT / "deploy" / "modal_app.py", "/opt/pixal3d/deploy/modal_app.py", copy=True)
    .add_local_file(ROOT / "deploy" / "settings.py", "/opt/pixal3d/deploy/settings.py", copy=True)
    .add_local_file(ROOT / "deploy" / "http_api.py", "/opt/pixal3d/deploy/http_api.py", copy=True)
    .add_local_file(ROOT / "deploy" / "modal.toml", "/opt/pixal3d/deploy/modal.toml", copy=True)
    .add_local_file(ROOT / "pyproject.toml", "/opt/pixal3d/pyproject.toml", copy=True)
)
offline = image.env({"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
cpu_options = dict(image=offline, cpu=(SETTINGS.cpu_cores, SETTINGS.cpu_cores), memory=SETTINGS.cpu_memory_mib,
                   timeout=SETTINGS.cpu_timeout_seconds, retries=0,
                   min_containers=SETTINGS.min_containers, max_containers=1, buffer_containers=0,
                   scaledown_window=SETTINGS.scaledown_window,
                   include_source=False)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_identity():
    from pixalcuda.weights import sha256_file

    return {**{path.name: sha256_file(path) for path in sorted((ROOT / "pixalcuda").glob("*.py"))},
            **{f"deploy/{name}": sha256_file(ROOT / "deploy" / name) for name in ("modal_app.py", "settings.py", "http_api.py")},
            "deploy/modal.toml": sha256_file(ROOT / "deploy" / "modal.toml"),
            "pyproject.toml": sha256_file(ROOT / "pyproject.toml")}


def job_path(job_id):
    if len(job_id) != 64 or any(character not in "0123456789abcdef" for character in job_id):
        raise ValueError("Invalid job ID")
    return JOBS_ROOT / job_id


def mark(job_id, status, **extra):
    value = {"job_id": job_id, "status": status, "updated_at": time.time(), **extra}
    from pixalcuda.runtime import atomic_json

    atomic_json(job_path(job_id) / "status.json", value)
    jobs.commit()
    records[job_id] = value
    return value


def validate_image_files(files, multiview):
    from PIL import Image

    if not 1 <= len(files) <= SETTINGS.max_views:
        raise ValueError(f"Operational limit is 1-{SETTINGS.max_views} mapped images")
    sizes = [path.stat().st_size for path, _ in files]
    if any(size > SETTINGS.max_upload_bytes for size in sizes) or sum(sizes) > SETTINGS.max_total_image_bytes:
        raise ValueError("Per-image or aggregate upload limit exceeded")
    for path, _ in files:
        with Image.open(path) as source:
            if source.format not in ("PNG", "JPEG", "WEBP") or source.width * source.height > SETTINGS.max_image_pixels:
                raise ValueError("Unsupported image format or image pixel limit exceeded")
            if getattr(source, "is_animated", False):
                raise ValueError("Animated images are not supported")
            if multiview and source.getexif().get(274, 1) != 1:
                raise ValueError("Mapped images must already be in calibrated orientation; remove EXIF rotation first")
            source.load()


def inference_limits(geometry_resolution=512, conditioning_cache_mb=128):
    from pixalcuda.ops import Limits

    if type(geometry_resolution) is not int or geometry_resolution not in (512, 1024):
        raise ValueError("Geometry resolution must be 512 or 1024")
    if type(conditioning_cache_mb) is not int or not 0 <= conditioning_cache_mb <= 2048:
        raise ValueError("Conditioning cache must be an integer between 0 and 2048 MiB")
    options = {"conditioning_cache_mb": conditioning_cache_mb,
               "attention_queries": SETTINGS.attention_queries, "cross_attention_queries": SETTINGS.cross_attention_queries,
               "ffn_rows": SETTINGS.ffn_rows, "sparse_rows": SETTINGS.sparse_rows}
    if geometry_resolution == 1024:
        options.update(max_tokens=SETTINGS.geometry_1024_max_tokens, max_voxels=SETTINGS.geometry_1024_max_voxels)
    return Limits(**options)


def validate_request_inputs(request, directory):
    from pixalcuda.cameras import uploaded_camera_metadata, view_filename

    inference_limits(request["options"]["geometry_resolution"], request["options"]["conditioning_cache_mb"])
    inputs = request["inputs"]
    if not request["multiview"] and len(inputs) != 1:
        raise ValueError("Single-image mode requires exactly one image")
    if not request["multiview"] and request["cameras"] is not None:
        raise ValueError("Single-image mode cannot carry mapped cameras")
    names = [view_filename(index) if request["multiview"] else "input.png" for index in range(len(inputs))]
    if [record["name"] for record in inputs] != names:
        raise ValueError("Uploaded filenames must match their canonical image indices")
    if request["multiview"]:
        cameras = request["cameras"]
        if len(json.dumps(cameras).encode()) > SETTINGS.max_transforms_bytes:
            raise ValueError("Camera JSON limit exceeded")
        if uploaded_camera_metadata(cameras, len(inputs)) != cameras:
            raise ValueError("Camera mapping must be normalized in image order")
    validate_image_files([(directory / "upload" / name, name) for name in names], request["multiview"])


@app.function(image=image, cpu=(SETTINGS.cpu_cores, SETTINGS.cpu_cores), memory=SETTINGS.prefetch_memory_mib,
              timeout=SETTINGS.prefetch_timeout_seconds, retries=0,
              min_containers=SETTINGS.min_containers, max_containers=1, scaledown_window=SETTINGS.scaledown_window,
              include_source=False, volumes={"/weights": weights},
              secrets=[modal.Secret.from_name(SETTINGS.hf_secret, required_keys=["HF_TOKEN"], environment_name=SETTINGS.environment)])
def prefetch(multiview=False, camera=True, rembg=True):
    from pixalcuda.runtime import atomic_json
    from pixalcuda.weights import ModelStore

    store = ModelStore(multiview=multiview)
    store.download(camera=camera)
    if rembg:
        from rembg import new_session
        new_session("u2net", providers=["CPUExecutionProvider"])
    marker = {"revision": store.revision, "camera": camera and not multiview, "rembg": rembg}
    atomic_json(WEIGHTS_ROOT / ("ready-mv.json" if multiview else "ready-sv.json"), marker)
    weights.commit()
    return marker


@app.function(**cpu_options, volumes={"/weights": weights.with_mount_options(read_only=True), "/jobs": jobs})
def prepare_job(job_id, request):
    from pixalcuda.multiview import prepare_views, read_views
    from pixalcuda.preprocess import prepare_foreground
    from pixalcuda.runtime import atomic_json
    from pixalcuda.weights import sha256_file

    if fingerprint(request) != job_id or request["source"] != source_identity() or request["gpu"] != GPU:
        raise ValueError("Request/source/GPU mismatch; submit with the deployed source and PIXAL_GPU")
    directory = job_path(job_id)
    jobs.reload()
    weights.reload()
    if not records.put(job_id, {"status": "preparing", "job_id": job_id}, skip_if_exists=True):
        if records[job_id]["status"] != "submitted" or not records.put(f"prepare:{job_id}", True, skip_if_exists=True):
            return records[job_id]
    try:
        directory.mkdir(parents=True, exist_ok=True)
        mark(job_id, "preparing")
        validate_request_inputs(request, directory)
        options = request["options"]
        marker = json.loads((WEIGHTS_ROOT / ("ready-mv.json" if request["multiview"] else "ready-sv.json")).read_text())
        if (marker["revision"] != request["revision"] or (options["remove_background"] and not marker["rembg"])
                or (not request["multiview"] and options["camera"] == "auto" and not marker["camera"])):
            raise ValueError("Run CPU prefetch for this mode/camera/background-removal setting first")
        for record in request["inputs"]:
            if sha256_file(directory / "upload" / record["name"]) != record["sha256"]:
                raise ValueError("Uploaded image hash mismatch")
        prepared = directory / "prepared"
        prepared.mkdir(exist_ok=True)
        if request["multiview"]:
            atomic_json(directory / "upload" / "transforms.json", request["cameras"])
            views = prepare_views(read_views(directory / "upload"), options["remove_background"])
            for index, view in enumerate(views.images):
                view.save(prepared / f"view{index:02d}.png")
            atomic_json(prepared / "transforms.json", request["cameras"])
        else:
            prepare_foreground(directory / "upload" / "input.png", options["remove_background"]).save(prepared / "input.png")
        atomic_json(directory / "request.json", request)
        mark(job_id, "queued_gpu")
        call = infer_job.spawn(job_id)
        return {"job_id": job_id, "status": "queued_gpu", "call_id": call.object_id}
    except BaseException as error:
        mark(job_id, "failed", error=f"{type(error).__name__}: {error}")
        raise


@app.function(image=offline, gpu=GPU, cpu=(SETTINGS.cpu_cores, SETTINGS.cpu_cores), memory=SETTINGS.gpu_memory_mib,
              timeout=SETTINGS.gpu_timeout_seconds, retries=0,
              min_containers=SETTINGS.min_containers, max_containers=SETTINGS.max_gpu_containers,
              buffer_containers=0, scaledown_window=SETTINGS.scaledown_window,
              include_source=False, volumes={"/weights": weights.with_mount_options(read_only=True), "/jobs": jobs})
def infer_job(job_id):
    from pixalcuda.pipeline import Pipeline

    directory = job_path(job_id)
    jobs.reload()
    weights.reload()
    try:
        request = json.loads((directory / "request.json").read_text())
        if fingerprint(request) != job_id or request["source"] != source_identity() or request["gpu"] != GPU:
            raise ValueError("Deployed GPU worker no longer matches this request")
        mark(job_id, "running_gpu")
        options = {**request["options"], "remove_background": False, "defer_export": True}
        limits = inference_limits(options["geometry_resolution"], options.pop("conditioning_cache_mb"))
        if request["multiview"]:
            options["views_dir"] = directory / "prepared"
        Pipeline("cuda", limits=limits).generate(None if request["multiview"] else directory / "prepared" / "input.png",
                                  directory / "run", **options)
        mark(job_id, "queued_export")
        call = export_job.spawn(job_id)
        return {"job_id": job_id, "status": "queued_export", "call_id": call.object_id}
    except BaseException as error:
        mark(job_id, "failed", error=f"{type(error).__name__}: {error}")
        raise


@app.function(**cpu_options, volumes={"/jobs": jobs})
def export_job(job_id):
    from pixalcuda.pipeline import finish_export
    from pixalcuda.weights import sha256_file

    directory = job_path(job_id)
    jobs.reload()
    try:
        mark(job_id, "exporting_cpu")
        result = finish_export(directory / "run")
        return mark(job_id, "complete", output=f"{job_id}/run/result.glb", sha256=sha256_file(result), bytes=result.stat().st_size)
    except BaseException as error:
        mark(job_id, "failed", error=f"{type(error).__name__}: {error}")
        raise


def build_request(image_path, views_dir, seed, texture_size, target_faces, camera, fov_degrees, no_rembg, attempt,
                  *, images=None, transforms=None, geometry_resolution=512, conditioning_cache_mb=128):
    from pixalcuda.cameras import uploaded_camera_metadata, view_filename
    from pixalcuda.multiview import read_mapped_views, read_views
    from pixalcuda.weights import MULTIVIEW_REVISION, PIXAL_REVISION, sha256_file

    inference_limits(geometry_resolution, conditioning_cache_mb)
    if sum((bool(image_path), bool(views_dir), images is not None)) != 1:
        raise ValueError("Supply exactly one of --image-path, --views-dir or --images")
    if (images is not None) != bool(transforms):
        raise ValueError("--images and --transforms must be supplied together")
    multiview = bool(views_dir) or images is not None
    if texture_size not in (512, 1024, 2048) or target_faces < 4 or camera not in ("auto", "fixed"):
        raise ValueError("Invalid texture size, target faces or camera")
    if not 1 <= fov_degrees <= 150:
        raise ValueError("FOV must be between 1 and 150 degrees")
    if multiview and (camera != "auto" or fov_degrees != 49.13):
        raise ValueError("Mapped cameras supply the FOV; camera/FOV overrides are not allowed")
    cameras = None
    if multiview:
        metadata_path = Path(transforms) if images is not None else Path(views_dir) / "transforms.json"
        if metadata_path.stat().st_size > SETTINGS.max_transforms_bytes:
            raise ValueError("Camera JSON limit exceeded")
        manifest = read_mapped_views(images, json.loads(metadata_path.read_text())) if images is not None else read_views(views_dir)
        files = [(path, view_filename(index)) for index, path in enumerate(manifest.paths)]
        cameras = uploaded_camera_metadata({"mesh_scale": manifest.rig.mesh_scale, "frames": [
            {"image_index": index, "camera_angle_x": float(manifest.rig.camera_angle_x[0, index]),
             "transform_matrix": manifest.rig.transform_matrix[0, index].tolist()}
            for index in range(len(files))]}, len(files))
    else:
        files = [(Path(image_path).expanduser().resolve(), "input.png")]
    validate_image_files(files, multiview)
    request = {"schema": 3, "source": source_identity(), "gpu": GPU, "multiview": multiview,
               "revision": MULTIVIEW_REVISION if multiview else PIXAL_REVISION, "cameras": cameras,
               "inputs": [{"name": name, "sha256": sha256_file(path)} for path, name in files], "attempt": attempt,
               "options": {"seed": seed, "texture_size": texture_size, "target_faces": target_faces,
                           "geometry_resolution": geometry_resolution, "conditioning_cache_mb": conditioning_cache_mb,
                           "camera": camera, "fov_degrees": fov_degrees, "remove_background": not no_rembg}}
    return request, files


def reserve_api_quota(job_id, now=None):
    from deploy.http_api import DailyLimitExceeded

    now = now or datetime.now(timezone.utc)
    day = now.date().isoformat()
    for slot in range(SETTINGS.api_max_jobs_per_day):
        key = f"api-quota:{day}:{slot}"
        if records.put(key, job_id, skip_if_exists=True) or records[key] == job_id:
            return
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    raise DailyLimitExceeded(max(1, math.ceil((tomorrow - now).total_seconds())))


class HttpBackend:
    def submit(self, images, transforms, options):
        with tempfile.TemporaryDirectory(prefix="pixal-camera-") as temporary:
            metadata_path = Path(temporary) / "cameras.json"
            if transforms is not None:
                metadata_path.write_text(json.dumps(transforms))
            request, files = build_request(
                "" if transforms is not None else images[0], "", options["seed"], options["texture_size"],
                options["target_faces"], options["camera"], options["fov_degrees"],
                not options["remove_background"], options["attempt"],
                images=images if transforms is not None else None,
                transforms=metadata_path if transforms is not None else None,
                geometry_resolution=options["geometry_resolution"], conditioning_cache_mb=options["conditioning_cache_mb"],
            )
        job_id = fingerprint(request)
        previous = records.get(job_id)
        if previous is not None:
            return previous
        reserve_api_quota(job_id)
        jobs.reload()
        directory = job_path(job_id)
        upload = directory / "upload"
        upload.mkdir(parents=True, exist_ok=True)
        for path, name in files:
            shutil.copyfile(path, upload / name)
        jobs.commit()
        record = {"job_id": job_id, "status": "submitted", "updated_at": time.time()}
        if records.put(job_id, record, skip_if_exists=True):
            try:
                prepare_job.spawn(job_id, request)
            except BaseException:
                records[job_id] = {**record, "status": "failed", "error": "Dispatch failed; inspect Modal logs before a new attempt"}
                raise
        return records[job_id]

    def status(self, job_id):
        return records.get(job_id)

    def copy_artifact(self, job_id, filename, destination):
        if filename not in ("result.glb", "report.json"):
            raise ValueError("Unsupported artifact")
        jobs.reload()
        shutil.copyfile(job_path(job_id) / "run" / filename, destination)


@app.function(image=offline, cpu=(0.5, 1), memory=SETTINGS.api_memory_mib, timeout=120,
              min_containers=0, max_containers=1, buffer_containers=0, scaledown_window=SETTINGS.scaledown_window,
              include_source=False, volumes={"/jobs": jobs})
@modal.concurrent(max_inputs=4)
@modal.asgi_app(requires_proxy_auth=True)
def api():
    from deploy.http_api import create_app

    return create_app(HttpBackend(), SETTINGS)


def run_client(action: str = "generate", image_path: str = "", views_dir: str = "", multiview: bool = False,
         seed: int = 42, texture_size: int = 1024, target_faces: int = 100000, camera: str = "auto",
         fov_degrees: float = 49.13, no_rembg: bool = False, attempt: str = "", job_id: str = "",
         output_dir: str = "outputs/modal", wait_seconds: int = 3600, images=None, transforms=None,
         geometry_resolution: int = 512, conditioning_cache_mb: int = 128):
    if action == "plan":
        print(json.dumps({"scope": "Local declarations only; no remote diff, deployment or spend cap",
                          "environment": SETTINGS.environment,
                          "resources": asdict(SETTINGS)}, indent=2))
        return
    if action == "url":
        print(modal.Function.from_name(SETTINGS.app_name, "api", environment_name=SETTINGS.environment).get_web_url())
        return
    if action == "prefetch":
        function = modal.Function.from_name(SETTINGS.app_name, "prefetch", environment_name=SETTINGS.environment)
        print(function.remote(multiview=multiview, camera=camera == "auto", rembg=not no_rembg))
        return
    if action == "generate":
        request, files = build_request(image_path, views_dir, seed, texture_size, target_faces, camera,
                                      fov_degrees, no_rembg, attempt, images=images, transforms=transforms,
                                      geometry_resolution=geometry_resolution, conditioning_cache_mb=conditioning_cache_mb)
        job_id = fingerprint(request)
        if job_id not in records:
            with jobs.batch_upload(force=True) as batch:
                for path, name in files:
                    batch.put_file(str(path), f"/{job_id}/upload/{name}")
            print(modal.Function.from_name(SETTINGS.app_name, "prepare_job", environment_name=SETTINGS.environment).remote(job_id, request))
    elif action != "status":
        raise ValueError("Action must be plan, url, prefetch, generate or status")
    job_path(job_id)
    print(f"Job: {job_id}", flush=True)
    deadline = time.monotonic() + wait_seconds
    previous = None
    while True:
        status = records.get(job_id)
        if status is None:
            raise ValueError("Unknown job; submit generate first")
        if status["status"] != previous:
            print(json.dumps(status), flush=True)
            previous = status["status"]
        if status["status"] == "failed":
            raise RuntimeError(status.get("error", "Worker failed"))
        if status["status"] == "complete":
            destination = Path(output_dir).expanduser() / job_id
            destination.mkdir(parents=True, exist_ok=True)
            for filename in ("result.glb", "report.json", "manifest.json"):
                with (destination / filename).open("wb") as target:
                    for chunk in jobs.read_file(f"/{job_id}/run/{filename}"):
                        target.write(chunk)
            print(f"Saved: {destination.resolve() / 'result.glb'}")
            return
        if time.monotonic() >= deadline:
            print("Stopped local polling; remote job may still be running. Use --action status --job-id to reconnect.")
            return
        time.sleep(5)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("plan", "url", "prefetch", "generate", "status"), default="generate")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--image-path", default="")
    inputs.add_argument("--views-dir", default="")
    inputs.add_argument("--images", nargs="+")
    parser.add_argument("--transforms")
    parser.add_argument("--multiview", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--geometry-resolution", type=int, choices=(512, 1024), default=512)
    parser.add_argument("--conditioning-cache-mb", type=int, default=128, help="Per-flow projection cache budget, 0-2048 MiB")
    parser.add_argument("--texture-size", type=int, choices=(512, 1024, 2048), default=1024)
    parser.add_argument("--target-faces", type=int, default=100000)
    parser.add_argument("--camera", choices=("auto", "fixed"), default="auto")
    parser.add_argument("--fov-degrees", type=float, default=49.13)
    parser.add_argument("--no-rembg", action="store_true")
    parser.add_argument("--attempt", default="")
    parser.add_argument("--job-id", default="")
    parser.add_argument("--output-dir", default="outputs/modal")
    parser.add_argument("--wait-seconds", type=int, default=3600)
    run_client(**vars(parser.parse_args()))
