"""Isolated worker delegating neural inference and remesh/PBR export to upstream."""

from contextlib import contextmanager
import gc
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import psutil
import torch

from .pipeline import validate_pbr_glb
from .runtime import atomic_json
from .upstream_sources import lock, verify_source
from .upstream_weights import local_pipeline_config


def require_hardware():
    if not torch.cuda.is_available():
        raise RuntimeError("Official Pixal3D needs CUDA and its native extensions")
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        raise RuntimeError("The unmodified BF16 baseline requires SM80+; T4/SM75 needs a separately validated port. No legacy or GPU fallback is applied.")
    if capability not in ((8, 0), (8, 6), (8, 9)):
        raise RuntimeError("The locked native build targets SM80/86/89; use a matching architecture build")
    if torch.__version__.split("+")[0] != "2.6.0":
        raise RuntimeError("The locked native baseline requires torch==2.6.0 with matching compiled extensions")
    return {"gpu": torch.cuda.get_device_name(), "capability": list(capability),
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory, "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda}


class NativeReport:
    def __init__(self, directory, identity):
        self.path = directory / "report.json"
        self.data = {"schema": "pixalcuda.upstream.v1", "identity": identity, "status": "running", "stages": []}

    def write(self):
        atomic_json(self.path, self.data)

    @contextmanager
    def stage(self, name):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            with torch.profiler.record_function(f"pixalcuda.upstream/{name}"):
                yield
        finally:
            torch.cuda.synchronize()
            self.data["stages"].append({"name": name, "elapsed_seconds": time.perf_counter() - started,
                                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                                        "rss_bytes_after": psutil.Process().memory_info().rss})
            self.write()


def load_entry(root, multiview):
    filename = verify_source("pixal3d", root) / ("inference_mv.py" if multiview else "inference.py")
    specification = importlib.util.spec_from_file_location("pixalcuda_official_entry", filename)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def pin_naf_loader(root, weights):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor

    directory = verify_source("naf", root)
    def load(extractor):
        if extractor.naf_model is None:
            model = torch.hub.load(str(directory), "naf", pretrained=False, device="cpu", source="local")
            model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
            extractor.naf_model = model.to(next(extractor.model.parameters()).device).eval().requires_grad_(False)
    DinoV3ProjFeatureExtractor._load_naf = load


def single_input(entry, pipeline, identity, bundle, directory):
    with Image.open(identity["inputs"][0]["path"]) as source:
        if identity["remove_background"]:
            image = pipeline.preprocess_image(source.copy())
        else:
            rgba = np.asarray(source.convert("RGBA"), dtype=np.float32)
            image = Image.fromarray((rgba[..., :3] * (rgba[..., 3:] / 255)).astype(np.uint8))
    destination = directory / "input-preprocessed.png"
    image.save(destination)
    if identity["camera"] == "fixed":
        angle = math.radians(identity["fov_degrees"])
        distance = entry.distance_from_fov(angle, torch.tensor([-1.0, 0.0, 0.0]), torch.tensor([0, 511]), 1.0, 512)["distance_from_x"]
        camera = {"camera_angle_x": angle, "distance": distance, "mesh_scale": 1.0}
    else:
        model = entry.load_moge_model(device="cuda", model_name=bundle["moge"])
        camera = entry.get_camera_params_wild_moge(str(destination), model, device="cuda")
        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return image, camera


def mapped_input(entry, pipeline, identity, directory):
    inputs = directory / "mapped-inputs"
    inputs.mkdir()
    metadata = identity["cameras"]
    for frame, record in zip(metadata["frames"], identity["inputs"]):
        with Image.open(record["path"]) as source:
            source.save(inputs / frame["file_path"], format="PNG")
    atomic_json(inputs / "transforms.json", metadata)
    remover = entry.make_rembg(pipeline) if identity["remove_background"] else lambda image: image.convert("RGBA")
    views = entry.load_views(str(inputs), rembg=remover)
    entry.check_main_view(views)
    return views


@torch.inference_mode()
def execute(request, directory, entry, exporter, report):
    identity, bundle = request["identity"], request["bundle"]
    for config in entry.IMAGE_COND_CONFIGS.values():
        config["model_name"] = bundle["dino"]
    with report.stage("load_pipeline"):
        local = local_pipeline_config(bundle, directory / "pipeline-config")
        pipeline = entry.init_pipeline(model_path=str(local), low_vram=identity["low_vram"])
    report.data["conditioning"] = {name: {key: config[key] for key in ("image_size", "grid_resolution", "naf_target_size") if key in config}
                                   for name, config in entry.IMAGE_COND_CONFIGS.items()}
    with report.stage("preprocess_camera"):
        prepared = mapped_input(entry, pipeline, identity, directory) if identity["multiview"] else single_input(entry, pipeline, identity, bundle, directory)
    options = {"seed": identity["seed"], "return_latent": True,
               "pipeline_type": f"{identity['geometry_resolution']}_cascade", "max_num_tokens": identity["max_tokens"]}
    with report.stage("denoise_decode"):
        if identity["multiview"]:
            meshes, latents = pipeline.run_mv(prepared, **options)
        else:
            image, camera = prepared
            report.data["camera"] = camera
            meshes, latents = pipeline.run(image, camera_params=camera, preprocess_image=False, **options)
    actual_resolution = latents[2]
    report.data.update(actual_geometry_resolution=actual_resolution,
                       upstream_resolution_fallback=actual_resolution != identity["geometry_resolution"])
    mesh = meshes[0]
    if identity["save_reference"]:
        with report.stage("save_reference"):
            reference = {"schema": "pixalcuda.native-reference.v1", "identity": identity,
                         "actual_geometry_resolution": actual_resolution,
                         "shape": {"features": latents[0].feats.cpu(), "coordinates": latents[0].coords.cpu()},
                         "texture": {"features": latents[1].feats.cpu(), "coordinates": latents[1].coords.cpu()},
                         "mesh": {name: getattr(mesh, name).cpu() for name in ("vertices", "faces", "attrs", "coords")}}
            torch.save(reference, directory / "reference.pt")
            del reference
            report.data["private_reference"] = "reference.pt"
    with report.stage("remesh_pbr_export"):
        glb = exporter(vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
                       attr_layout=pipeline.pbr_attr_layout, grid_size=actual_resolution,
                       aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], decimation_target=identity["target_faces"],
                       texture_size=identity["texture_size"], remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True)
        glb.apply_transform(np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64))
        temporary = directory / "result.pending.glb"
        glb.export(str(temporary), extension_webp=True)
        validation = validate_pbr_glb(temporary)
        temporary.replace(directory / "result.glb")
    report.data.update(status="complete", validation=validation, postprocess_device="cuda")
    report.write()
    return directory / "result.glb"


def run(request_path):
    request_path = Path(request_path)
    request = json.loads(request_path.read_text())
    report = NativeReport(request_path.parent, request["identity"])
    try:
        if request["identity"]["upstream_lock"] != lock():
            raise ValueError("Worker lock differs from the submitted request")
        report.data.update(require_hardware())
        for name in lock()["sources"]:
            verify_source(name, request["upstream_root"])
        entry = load_entry(request["upstream_root"], request["identity"]["multiview"])
        pin_naf_loader(request["upstream_root"], request["bundle"]["naf_weights"])
        return execute(request, request_path.parent, entry, entry.o_voxel.postprocess.to_glb, report)
    except BaseException as error:
        report.data.update(status="failed", error=f"{type(error).__name__}: {error}")
        report.write()
        raise


if __name__ == "__main__":
    if sys.argv[1:] == ["--check-hardware"]:
        print(json.dumps(require_hardware()))
    else:
        run(sys.argv[1])
