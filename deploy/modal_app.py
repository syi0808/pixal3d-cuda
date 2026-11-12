"""Authenticated, scale-to-zero Modal workers; no public HTTP endpoint."""

import hashlib
import json
import os
from pathlib import Path
import time

import modal


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_ROOT = Path("/weights")
JOBS_ROOT = Path("/jobs")
GPU = os.environ.get("PIXAL_GPU", "T4")
if GPU not in ("T4", "L4"):
    raise ValueError("PIXAL_GPU must be T4 or L4; no automatic expensive GPU fallback")

app = modal.App("pixal3d-cuda")
weights = modal.Volume.from_name("pixal3d-weights", create_if_missing=True, version=2)
jobs = modal.Volume.from_name("pixal3d-jobs", create_if_missing=True, version=2)
records = modal.Dict.from_name("pixal3d-jobs", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libglib2.0-0", "libgl1")
    .pip_install("torch==2.6.0", "torchvision==0.21.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install_from_pyproject(str(ROOT / "pyproject.toml"), optional_dependencies=["preprocess", "camera"])
    .env({"PYTHONPATH": "/opt/pixal3d:/opt/pixal3d/deploy", "HF_HOME": "/weights/hf", "U2NET_HOME": "/weights/u2net", "PIXAL_GPU": GPU,
          "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(ROOT / "pixalcuda", "/opt/pixal3d/pixalcuda", copy=True, ignore=["__pycache__", "*.pyc"])
    .add_local_dir(ROOT / "deploy", "/opt/pixal3d/deploy", copy=True, ignore=["__pycache__", "*.pyc"])
    .add_local_file(ROOT / "pyproject.toml", "/opt/pixal3d/pyproject.toml", copy=True)
)
offline = image.env({"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
cpu_options = dict(image=offline, cpu=(2, 2), memory=(4096, 16384), timeout=1800, retries=0,
                   min_containers=0, max_containers=1, buffer_containers=0, scaledown_window=2,
                   include_source=False)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_identity():
    from pixalcuda.weights import sha256_file

    return {**{path.name: sha256_file(path) for path in sorted((ROOT / "pixalcuda").glob("*.py"))},
            "deploy/modal_app.py": sha256_file(__file__), "pyproject.toml": sha256_file(ROOT / "pyproject.toml")}


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


@app.function(image=image, cpu=(2, 2), memory=(2048, 8192), timeout=3600, retries=0,
              min_containers=0, max_containers=1, scaledown_window=2, include_source=False,
              volumes={"/weights": weights}, secrets=[modal.Secret.from_name("pixal3d-hf")])
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
        return records[job_id]
    try:
        directory.mkdir(parents=True, exist_ok=True)
        mark(job_id, "preparing")
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


@app.function(image=offline, gpu=GPU, cpu=(2, 2), memory=(8192, 16384), timeout=1800, retries=0,
              min_containers=0, max_containers=1, buffer_containers=0, scaledown_window=2,
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
        if request["multiview"]:
            options["views_dir"] = directory / "prepared"
        Pipeline("cuda").generate(None if request["multiview"] else directory / "prepared" / "input.png",
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


def build_request(image_path, views_dir, seed, texture_size, target_faces, camera, fov_degrees, no_rembg, attempt):
    from pixalcuda.multiview import read_views
    from pixalcuda.weights import MULTIVIEW_REVISION, PIXAL_REVISION, sha256_file

    if bool(image_path) == bool(views_dir):
        raise ValueError("Supply exactly one of --image-path or --views-dir")
    if texture_size not in (512, 1024, 2048) or target_faces < 4 or camera not in ("auto", "fixed"):
        raise ValueError("Invalid texture size, target faces or camera")
    if not 1 <= fov_degrees <= 150:
        raise ValueError("FOV must be between 1 and 150 degrees")
    if views_dir and (camera != "auto" or fov_degrees != 49.13):
        raise ValueError("Multi-view uses transforms.json, not camera/FOV overrides")
    cameras = None
    if views_dir:
        manifest = read_views(views_dir)
        files = [(path, f"view{index:02d}.png") for index, path in enumerate(manifest.paths)]
        cameras = {"mesh_scale": manifest.rig.mesh_scale, "frames": [
            {"file_path": name, "camera_angle_x": float(manifest.rig.camera_angle_x[0, index]),
             "transform_matrix": manifest.rig.transform_matrix[0, index].tolist()}
            for index, (_, name) in enumerate(files)]}
    else:
        files = [(Path(image_path).expanduser().resolve(), "input.png")]
    if sum(path.stat().st_size for path, _ in files) > 64 * 1024 ** 2:
        raise ValueError("Combined upload limit is 64 MiB")
    request = {"schema": 1, "source": source_identity(), "gpu": GPU, "multiview": bool(views_dir),
               "revision": MULTIVIEW_REVISION if views_dir else PIXAL_REVISION, "cameras": cameras,
               "inputs": [{"name": name, "sha256": sha256_file(path)} for path, name in files], "attempt": attempt,
               "options": {"seed": seed, "texture_size": texture_size, "target_faces": target_faces,
                           "camera": camera, "fov_degrees": fov_degrees, "remove_background": not no_rembg}}
    return request, files


def run_client(action: str = "generate", image_path: str = "", views_dir: str = "", multiview: bool = False,
         seed: int = 42, texture_size: int = 1024, target_faces: int = 100000, camera: str = "auto",
         fov_degrees: float = 49.13, no_rembg: bool = False, attempt: str = "", job_id: str = "",
         output_dir: str = "outputs/modal", wait_seconds: int = 3600):
    if action == "prefetch":
        function = modal.Function.from_name("pixal3d-cuda", "prefetch")
        print(function.remote(multiview=multiview, camera=camera == "auto", rembg=not no_rembg))
        return
    if action == "generate":
        request, files = build_request(image_path, views_dir, seed, texture_size, target_faces, camera, fov_degrees, no_rembg, attempt)
        job_id = fingerprint(request)
        if job_id not in records:
            with jobs.batch_upload(force=True) as batch:
                for path, name in files:
                    batch.put_file(str(path), f"/{job_id}/upload/{name}")
            print(modal.Function.from_name("pixal3d-cuda", "prepare_job").remote(job_id, request))
    elif action != "status":
        raise ValueError("Action must be prefetch, generate or status")
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
    parser.add_argument("--action", choices=("prefetch", "generate", "status"), default="generate")
    parser.add_argument("--image-path", default="")
    parser.add_argument("--views-dir", default="")
    parser.add_argument("--multiview", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
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
