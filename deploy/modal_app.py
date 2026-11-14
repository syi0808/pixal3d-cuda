"""Official native CUDA generation and proxy-authenticated API in Modal main."""

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
from pixalcuda.upstream_sources import NATIVE_PACKAGES


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_ROOT = Path("/weights")
JOBS_ROOT = Path("/jobs")
SETTINGS = load_settings()
GPU = SETTINGS.gpu

app = modal.App(SETTINGS.app_name)
weights = modal.Volume.from_name(SETTINGS.weights_volume, create_if_missing=True, version=2, environment_name=SETTINGS.environment)
jobs = modal.Volume.from_name(SETTINGS.jobs_volume, create_if_missing=True, version=2, environment_name=SETTINGS.environment)
records = modal.Dict.from_name(SETTINGS.jobs_dict, create_if_missing=True, environment_name=SETTINGS.environment)
native_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential", "ninja-build", "libglib2.0-0", "libgl1", "libgl1-mesa-dev", "libegl1-mesa-dev", "libjpeg-dev")
    .pip_install("torch==2.6.0", "torchvision==0.21.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install_from_pyproject(str(ROOT / "pyproject.toml"), optional_dependencies=["upstream", "camera", "server"])
    .env({"PYTHONPATH": "/opt/pixal3d:/opt/pixal3d/deploy", "HF_HOME": "/weights/hf", "PIXAL_GPU": GPU,
          "PIXAL_UPSTREAM_ROOT": "/opt/upstream", "CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.9",
          "CC": "gcc", "CXX": "g++",
          "GIT_LFS_SKIP_SMUDGE": "1",
          "ATTN_BACKEND": "sdpa", "SPARSE_ATTN_BACKEND": "sdpa", "SPARSE_CONV_BACKEND": "flex_gemm",
          "MODAL_ENVIRONMENT": SETTINGS.environment, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_file(ROOT / "pixalcuda" / "upstream_sources.py", "/opt/pixal-build/pixalcuda/upstream_sources.py", copy=True)
    .add_local_file(ROOT / "pixalcuda" / "upstream-lock.json", "/opt/pixal-build/pixalcuda/upstream-lock.json", copy=True)
    .run_commands("PYTHONPATH=/opt/pixal-build python -m pixalcuda.upstream_sources --root /opt/upstream")
)
for native_package in NATIVE_PACKAGES:
    native_image = native_image.run_commands(
        f"PYTHONPATH=/opt/pixal-build python -m pixalcuda.upstream_sources --root /opt/upstream --build-package {native_package} --cuda-arch 8.9")
image = (
    native_image
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
            "pyproject.toml": sha256_file(ROOT / "pyproject.toml"),
            "upstream-lock.json": sha256_file(ROOT / "pixalcuda" / "upstream-lock.json")}


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


def inference_options(geometry_resolution=1024, reference=False):
    from pixalcuda.pipeline import validate_options
    from pixalcuda.optimizations import OptimizationOptions

    validate_options(geometry_resolution)
    if type(reference) is not bool:
        raise ValueError("reference must be boolean")
    optimizations = OptimizationOptions.disabled() if reference else OptimizationOptions(
        SETTINGS.share_dino, SETTINGS.dino_cache_mb, SETTINGS.sparse_projection, SETTINGS.conditioning_cache_mb)
    return {"low_vram": SETTINGS.low_vram, "attention_backend": SETTINGS.attention_backend,
            "max_tokens": SETTINGS.max_tokens, "timeout_seconds": SETTINGS.gpu_timeout_seconds - 30,
            "optimizations": optimizations, "save_reference": SETTINGS.save_reference,
            "skip_weight_init": SETTINGS.skip_weight_init and not reference,
            "inpaint_workers": 1 if reference else SETTINGS.inpaint_workers}


def validate_request_inputs(request, directory):
    from pixalcuda.cameras import uploaded_camera_metadata, view_filename

    inference_options(request["options"]["geometry_resolution"], request["options"]["reference"])
    SETTINGS.validate_background_removal(request["options"]["remove_background"])
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
def prefetch(multiview=False, camera=True, rembg=False):
    from pixalcuda.runtime import atomic_json
    from pixalcuda.upstream_weights import UpstreamStore
    from pixalcuda.upstream_sources import lock

    SETTINGS.validate_background_removal(rembg)
    store = UpstreamStore(multiview=multiview)
    store.resolve(camera=camera, rembg=rembg)
    marker = {"runtime": "official", "lock": lock(), "revision": store.revision, "camera": camera and not multiview, "rembg": rembg}
    atomic_json(WEIGHTS_ROOT / ("ready-mv.json" if multiview else "ready-sv.json"), marker)
    weights.commit()
    return marker


@app.function(**cpu_options, volumes={"/weights": weights.with_mount_options(read_only=True), "/jobs": jobs})
def prepare_job(job_id, request):
    from pixalcuda.runtime import atomic_json
    from pixalcuda.upstream_sources import lock
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
        if (marker.get("runtime") != "official" or marker.get("lock") != lock()
                or marker["revision"] != request["revision"] or (options["remove_background"] and not marker["rembg"])
                or (not request["multiview"] and options["camera"] == "auto" and not marker["camera"])):
            raise ValueError("Run CPU prefetch for this mode/camera/background-removal setting first")
        for record in request["inputs"]:
            if sha256_file(directory / "upload" / record["name"]) != record["sha256"]:
                raise ValueError("Uploaded image hash mismatch")
        prepared = directory / "prepared"
        prepared.mkdir(exist_ok=True)
        for record in request["inputs"]:
            shutil.copyfile(directory / "upload" / record["name"], prepared / record["name"])
        if request["multiview"]:
            atomic_json(prepared / "transforms.json", request["cameras"])
        atomic_json(directory / "request.json", request)
        mark(job_id, "queued_gpu")
        call = infer_job.spawn(job_id)
        return {"job_id": job_id, "status": "queued_gpu", "call_id": call.object_id}
    except BaseException as error:
        mark(job_id, "failed", error=f"{type(error).__name__}: {error}")
        raise


@app.function(image=offline, gpu=GPU, cpu=(SETTINGS.gpu_cpu_cores, SETTINGS.gpu_cpu_cores), memory=SETTINGS.gpu_memory_mib,
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
        options = {**request["options"]}
        SETTINGS.validate_background_removal(options["remove_background"])
        limits = inference_options(options["geometry_resolution"], options.pop("reference"))
        if request["multiview"]:
            options["views_dir"] = directory / "prepared"
        result = Pipeline("cuda", **limits).generate(None if request["multiview"] else directory / "prepared" / "input.png",
                                  directory / "run", **options)
        from pixalcuda.weights import sha256_file
        return mark(job_id, "complete", output=f"{job_id}/run/result.glb", sha256=sha256_file(result), bytes=result.stat().st_size)
    except BaseException as error:
        mark(job_id, "failed", error=f"{type(error).__name__}: {error}")
        raise


def build_request(image_path, views_dir, seed, texture_size, target_faces, camera, fov_degrees, no_rembg, attempt,
                  *, images=None, transforms=None, geometry_resolution=1024, reference=False):
    from pixalcuda.cameras import uploaded_camera_metadata, view_filename
    from pixalcuda.multiview import read_mapped_views, read_views
    from pixalcuda.pipeline import validate_options
    from pixalcuda.weights import MULTIVIEW_REVISION, PIXAL_REVISION, sha256_file

    inference_options(geometry_resolution, reference)
    if type(no_rembg) is not bool:
        raise ValueError("no_rembg must be boolean")
    SETTINGS.validate_background_removal(not no_rembg)
    validate_options(geometry_resolution, texture_size, target_faces)
    if sum((bool(image_path), bool(views_dir), images is not None)) != 1:
        raise ValueError("Supply exactly one of --image-path, --views-dir or --images")
    if (images is not None) != bool(transforms):
        raise ValueError("--images and --transforms must be supplied together")
    multiview = bool(views_dir) or images is not None
    if texture_size not in (512, 1024, 2048, 4096) or not 4 <= target_faces <= 1000000 or camera not in ("auto", "fixed"):
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
    request = {"schema": 4, "runtime": "official", "source": source_identity(), "gpu": GPU, "multiview": multiview,
               "revision": MULTIVIEW_REVISION if multiview else PIXAL_REVISION, "cameras": cameras,
               "inputs": [{"name": name, "sha256": sha256_file(path)} for path, name in files], "attempt": attempt,
               "options": {"seed": seed, "texture_size": texture_size, "target_faces": target_faces,
                           "geometry_resolution": geometry_resolution, "reference": reference,
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
                geometry_resolution=options["geometry_resolution"], reference=options["reference"],
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
         seed: int = 42, texture_size: int = 4096, target_faces: int = 1000000, camera: str = "auto",
         fov_degrees: float = 49.13, no_rembg: bool = True, attempt: str = "", job_id: str = "",
         output_dir: str = "outputs/modal", wait_seconds: int = 3600, images=None, transforms=None,
         geometry_resolution: int = 1024, reference: bool = False):
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
                                      geometry_resolution=geometry_resolution, reference=reference)
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
    parser.add_argument("--geometry-resolution", type=int, choices=(1024, 1536), default=1024)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--texture-size", type=int, choices=(512, 1024, 2048, 4096), default=4096)
    parser.add_argument("--target-faces", type=int, default=1000000)
    parser.add_argument("--camera", choices=("auto", "fixed"), default="auto")
    parser.add_argument("--fov-degrees", type=float, default=49.13)
    background = parser.add_mutually_exclusive_group()
    background.add_argument("--remove-background", dest="no_rembg", action="store_false", help="Requires server allow_rmbg and separate license review")
    background.add_argument("--no-rembg", dest="no_rembg", action="store_true")
    parser.set_defaults(no_rembg=True)
    parser.add_argument("--attempt", default="")
    parser.add_argument("--job-id", default="")
    parser.add_argument("--output-dir", default="outputs/modal")
    parser.add_argument("--wait-seconds", type=int, default=3600)
    run_client(**vars(parser.parse_args()))
