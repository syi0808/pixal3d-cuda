"""Opt-in full-path parity and CPU/native export instrumentation, not a backend."""

from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from functools import wraps
import gc
import hashlib
import json
from pathlib import Path
import time

import psutil
import torch

from .background import deferred_background_removal
from .optimizations import OptimizationOptions, OptimizationSession, Patches
from .pipeline import validate_pbr_glb
from .profiling import PrefixProbe, PrefixTrace, alias_loaded_backbones, fixed_splitk, tensor_digest
from .runtime import atomic_json
from .upstream_weights import local_pipeline_config


@contextmanager
def frozen_autotuning(persistent_type, triton_type):
    patches = Patches()

    def reject(*arguments, **options):
        raise RuntimeError("Autotuning is frozen; a new kernel/configuration key was requested")

    try:
        patches.set(persistent_type, "_benchmark", reject)
        patches.set(triton_type, "_bench", reject)
        yield
    finally:
        patches.close()


class Timings:
    def __init__(self, synchronize=False):
        self.synchronize = synchronize
        self.records = {}

    @contextmanager
    def measure(self, name):
        if self.synchronize:
            torch.cuda.synchronize()
        process = psutil.Process()
        cpu_before = process.cpu_times()
        started = time.perf_counter()
        try:
            yield
        finally:
            if self.synchronize:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            cpu_after = process.cpu_times()
            record = self.records.setdefault(name, {"calls": 0, "inclusive_wall_seconds": 0.0,
                                                     "process_cpu_seconds": 0.0})
            record["calls"] += 1
            record["inclusive_wall_seconds"] += elapsed
            record["process_cpu_seconds"] += cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system

    def wrap(self, name, original):
        @wraps(original)
        def forward(*arguments, **options):
            with self.measure(name):
                return original(*arguments, **options)
        return forward


class LoadProbe:
    def __init__(self, entry):
        self.entry = entry
        self.timings = Timings()
        self.patches = Patches()
        self.image_label = "image"
        self.extractor_labels = {}
        self.initializer_depth = 0

    def __enter__(self):
        import safetensors.torch
        from pixal3d import models
        from pixal3d.trainers.flow_matching.mixins import image_conditioned_proj

        try:
            original_model = models.from_pretrained

            def model(path, **options):
                label = "model." + Path(path).name
                config = json.loads(Path(str(path) + ".json").read_text())
                kind = getattr(models, config["name"])
                patches = Patches()
                try:
                    patches.set(kind, "__init__", self.timings.wrap(label + ".construct", kind.__init__))
                    patches.set(kind, "load_state_dict", self.timings.wrap(label + ".apply_checkpoint", kind.load_state_dict))
                    patches.set(safetensors.torch, "load_file", self.timings.wrap(label + ".checkpoint_mapping", safetensors.torch.load_file))
                    with self.timings.measure(label + ".total"):
                        return original_model(path, **options)
                finally:
                    patches.close()
            self.patches.set(models, "from_pretrained", model)
            original_builder = self.entry.build_image_cond_model

            def build(config):
                previous = self.image_label
                name = next(name for name, value in self.entry.IMAGE_COND_CONFIGS.items() if value == config)
                self.image_label = "image." + name
                try:
                    with self.timings.measure(self.image_label + ".build"):
                        result = original_builder(config)
                        self.extractor_labels[id(result)] = name
                        return result
                finally:
                    self.image_label = previous
            self.patches.set(self.entry, "build_image_cond_model", build)
            backbone = image_conditioned_proj.DINOv3ViTModel
            original_dino = backbone.from_pretrained

            def dino(*arguments, **options):
                with self.timings.measure(self.image_label + ".dino_load"):
                    return original_dino(*arguments, **options)
            self.patches.set(backbone, "from_pretrained", staticmethod(dino))
            extractor = image_conditioned_proj.DinoV3ProjFeatureExtractor
            original_naf = extractor._load_naf

            def naf(model):
                name = self.extractor_labels[id(model)]
                with self.timings.measure("image." + name + ".naf_load"):
                    return original_naf(model)
            self.patches.set(extractor, "_load_naf", naf)
            for name in ("kaiming_uniform_", "kaiming_normal_", "xavier_uniform_", "xavier_normal_", "trunc_normal_",
                         "normal_", "uniform_", "zeros_", "ones_", "constant_"):
                self.patches.set(torch.nn.init, name, self.initializer(name, getattr(torch.nn.init, name)))
            return self
        except BaseException:
            self.patches.close()
            raise

    def initializer(self, name, original):
        @wraps(original)
        def initialize(*arguments, **options):
            outer = self.initializer_depth == 0
            self.initializer_depth += 1
            try:
                with self.timings.measure("initialization." + name) if outer else nullcontext():
                    return original(*arguments, **options)
            finally:
                self.initializer_depth -= 1
        return initialize

    def __exit__(self, *error):
        self.patches.close()


class ExportProbe:
    def __init__(self, exporter):
        self.exporter = exporter
        self.timings = Timings(synchronize=True)
        self.patches = Patches()
        self.coarse = []

    def __enter__(self):
        import cumesh
        import cv2
        import nvdiffrast.torch as rasterizer

        try:
            for name in ("fill_holes", "simplify", "uv_unwrap", "compute_charts", "compute_vertex_normals"):
                self.patches.set(cumesh.CuMesh, name, self.timings.wrap("cumesh." + name, getattr(cumesh.CuMesh, name)))
            for name in ("add_mesh", "compute_charts", "pack_charts", "get_mesh"):
                self.patches.set(cumesh.Atlas, name, self.timings.wrap("xatlas." + name, getattr(cumesh.Atlas, name)))
            self.patches.set(cv2, "inpaint", self.timings.wrap("cv2.inpaint", cv2.inpaint))
            for name in ("rasterize", "interpolate"):
                self.patches.set(rasterizer, name, self.timings.wrap("nvdiffrast." + name, getattr(rasterizer, name)))
            namespace = self.exporter.__globals__
            self.original_tqdm = namespace["tqdm"]
            self.original_sampling = namespace["grid_sample_3d"]
            namespace["grid_sample_3d"] = self.timings.wrap("volume.grid_sample_3d", self.original_sampling)
            probe = self

            class Progress:
                def __init__(self, total, desc):
                    self.name = desc
                    torch.cuda.synchronize()
                    self.started = time.perf_counter()

                def update(self, amount):
                    torch.cuda.synchronize()
                    probe.coarse.append({"phase": self.name, "seconds": time.perf_counter() - self.started})

                def set_description(self, desc):
                    self.name = desc
                    torch.cuda.synchronize()
                    self.started = time.perf_counter()

                def close(self):
                    pass

            namespace["tqdm"] = Progress
            return self
        except BaseException:
            self.patches.close()
            raise

    def __exit__(self, *error):
        namespace = self.exporter.__globals__
        namespace["tqdm"] = self.original_tqdm
        namespace["grid_sample_3d"] = self.original_sampling
        self.patches.close()


def backend_settings():
    import os
    from flex_gemm.kernels.triton import config

    return {"cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "flex_gemm_allow_tf32": config.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


@torch.inference_mode()
def execute_full_profile(entry, bundle, image, camera, directory, report):
    from flex_gemm.kernels.triton import config
    from flex_gemm.utils.autotuner import PersistentCacheAutoTuner, TritonPersistentCacheAutotuner, get_autotune_cache
    from pixal3d.modules.sparse import SparseTensor
    from pixal3d.trainers.flow_matching.mixins import image_conditioned_proj

    directory = Path(directory)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    config.allow_tf32 = False
    report.data["backend_settings"] = backend_settings()
    with report.stage("load_reference_pipeline"):
        with LoadProbe(entry) as load_probe, deferred_background_removal():
            local = local_pipeline_config(bundle, directory / "reference-pipeline-config", rembg=False)
            pipeline = entry.init_pipeline(model_path=str(local), low_vram=True)
    report.data["load_reference_components"] = load_probe.timings.records
    profiles = []
    reference_count = None
    kernel_manifest = None
    report.data["profiles"] = profiles
    for name, options in (("reference_first", OptimizationOptions.disabled()),
                          ("reference_repeat", OptimizationOptions.disabled()), ("optimized", OptimizationOptions())):
        trace = PrefixTrace(directory / name, directory / "reference_first" if profiles else None,
                            max_snapshot_bytes=3 * 1024**3)
        freeze = frozen_autotuning(PersistentCacheAutoTuner, TritonPersistentCacheAutotuner) if profiles else nullcontext()
        with report.stage(name):
            with alias_loaded_backbones(pipeline, options.share_dino), fixed_splitk(PersistentCacheAutoTuner) as choices, freeze:
                with OptimizationSession(options, image_conditioned_proj.DINOv3ViTModel,
                                         image_conditioned_proj.DinoV3ProjFeatureExtractor, SparseTensor) as session:
                    session.install(pipeline, False)
                    with PrefixProbe(pipeline, trace, kernel_profile=bool(profiles), full=True) as probe:
                        meshes, latents = pipeline.run(image, camera_params=camera, seed=42, preprocess_image=False,
                                                      return_latent=True, pipeline_type="1024_cascade", max_num_tokens=49152)
                        trace.emit("final.shape_latent", latents[0])
                        trace.emit("final.texture_latent", latents[1])
                        if latents[2] != 1024:
                            raise RuntimeError("Unexpected resolution change during full native parity probe")
                        if not profiles:
                            mesh = meshes[0]
                            frozen = {key: getattr(mesh, key).detach().cpu() for key in ("vertices", "faces", "attrs", "coords")}
                            frozen["layout"] = {key: (value.start, value.stop) for key, value in pipeline.pbr_attr_layout.items()}
                            torch.save(frozen, directory / "frozen-mesh.pt")
                            report.data["frozen_mesh_digests"] = {key: tensor_digest(value) for key, value in frozen.items() if isinstance(value, torch.Tensor)}
                            del frozen, mesh
                        del meshes, latents
            summary = trace.finish(reference_count)
            if reference_count is None:
                reference_count = len(trace.events)
            manifest = get_autotune_cache()
            if kernel_manifest is None:
                kernel_manifest = json.loads(json.dumps(manifest))
            manifest_equal = manifest == kernel_manifest
            if not manifest_equal:
                raise RuntimeError("Native kernel cache changed despite frozen autotuning")
            atomic_json(directory / (name + "-autotune.json"), manifest)
            profiles.append({"profile": name, "options": asdict(options), "splitk_choices": choices,
                             "events": len(trace.events), "first_failure": summary["first_failure"],
                             "first_nonexact": summary["first_nonexact"], "passed": summary["passed"],
                             "first_activation_failure": summary["first_activation_failure"],
                             "first_activation_nonexact": summary["first_activation_nonexact"],
                             "snapshot_payload_bytes": trace.snapshot_bytes, "kernel_cache_equal": manifest_equal,
                             "kernel_summaries": probe.kernel_summaries,
                             "inclusive_timings_with_trace_overhead": probe.measurements,
                             "optimization_counters": session.report()})
            report.write()
        gc.collect()
        torch.cuda.empty_cache()
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    exports = []
    report.data["frozen_exports"] = exports
    for index in range(2):
        with report.stage(f"frozen_export_{index}"):
            frozen = torch.load(directory / "frozen-mesh.pt", map_location="cpu", weights_only=True)
            digests = {key: tensor_digest(value) for key, value in frozen.items() if isinstance(value, torch.Tensor)}
            if digests != report.data["frozen_mesh_digests"]:
                raise RuntimeError("Frozen export inputs changed")
            layout = {key: slice(*value) for key, value in frozen.pop("layout").items()}
            with ExportProbe(entry.o_voxel.postprocess.to_glb) as probe:
                glb = entry.o_voxel.postprocess.to_glb(vertices=frozen["vertices"].cuda(), faces=frozen["faces"].cuda(),
                    attr_volume=frozen["attrs"].cuda(), coords=frozen["coords"].cuda(), attr_layout=layout,
                    grid_size=1024, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], decimation_target=1000000,
                    texture_size=4096, remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True)
                import numpy as np

                glb.apply_transform(np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64))
                destination = directory / f"frozen-export-{index}.glb"
                with probe.timings.measure("glb.serialize_webp"):
                    glb.export(str(destination), extension_webp=True)
            exports.append({"index": index, "input_digests": digests, "coarse_phases": probe.coarse,
                            "nested_component_timings": probe.timings.records,
                            "validation": validate_pbr_glb(destination), "triangles": len(glb.faces)})
            del frozen, glb
            report.write()
        gc.collect()
        torch.cuda.empty_cache()
    with report.stage("load_optimized_pipeline"):
        with LoadProbe(entry) as load_probe:
            with OptimizationSession(OptimizationOptions(), image_conditioned_proj.DINOv3ViTModel,
                                     image_conditioned_proj.DinoV3ProjFeatureExtractor, SparseTensor) as session:
                with deferred_background_removal():
                    local = local_pipeline_config(bundle, directory / "optimized-pipeline-config", rembg=False)
                    pipeline = entry.init_pipeline(model_path=str(local), low_vram=True)
        report.data["load_optimized_components"] = load_probe.timings.records
        report.data["load_optimized_reuse_counters"] = session.report()
        del pipeline
    report.data["backend_settings_after"] = backend_settings()
    report.data["kernel_manifest_sha256"] = hashlib.sha256(json.dumps(kernel_manifest, sort_keys=True).encode()).hexdigest()
    return profiles
