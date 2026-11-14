"""Bounded L4 coverage/parity validation and frozen-chart CPU/inpaint trials."""

from contextlib import contextmanager, nullcontext
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .background import deferred_background_removal
from .full_profiling import LoadProbe, ExportProbe, backend_settings, frozen_autotuning
from .initialization import covered_initialization
from .inpainting import inpaint_batch
from .optimizations import OptimizationOptions, OptimizationSession, Patches
from .pipeline import validate_pbr_glb
from .profiling import PrefixProbe, PrefixTrace, fixed_splitk, tensor_digest
from .runtime import atomic_json
from .upstream_weights import local_pipeline_config


@contextmanager
def capture_export_inputs(directory):
    import cumesh
    import cv2

    patches, charts, images, masks = Patches(), [], [], []
    options = {}
    original_add = cumesh.Atlas.add_mesh
    original_inpaint = cv2.inpaint

    def add(atlas, vertices, faces, *arguments, **keywords):
        if arguments or keywords:
            raise ValueError("Unexpected atlas input contract")
        charts.append({"vertices": vertices.clone(), "faces": faces.clone()})
        return original_add(atlas, vertices, faces)

    def paint(image, mask, radius, method):
        images.append(image.copy())
        masks.append(mask.copy())
        return original_inpaint(image, mask, radius, method)

    def capture_options(name, original):
        def forward(atlas, **keywords):
            options[name] = dict(keywords)
            return original(atlas, **keywords)
        return forward
    try:
        patches.set(cumesh.Atlas, "add_mesh", add)
        for name in ("compute_charts", "pack_charts"):
            patches.set(cumesh.Atlas, name, capture_options(name, getattr(cumesh.Atlas, name)))
        patches.set(cv2, "inpaint", paint)
        yield
        if len(images) != 4 or not all(np.array_equal(masks[0], mask) for mask in masks):
            raise ValueError("Unexpected four-channel inpaint contract")
        torch.save({"charts": charts, "options": options}, Path(directory) / "charts.pt")
        np.savez_compressed(Path(directory) / "inpaint.npz", color=images[0], metallic=images[1],
                            roughness=images[2], alpha=images[3], mask=masks[0])
    finally:
        patches.close()


def controlled_backend():
    from flex_gemm.kernels.triton import config

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    config.allow_tf32 = False
    return backend_settings()


@torch.inference_mode()
def validate_initialization(entry, bundle, image, camera, directory, report):
    from flex_gemm.utils.autotuner import PersistentCacheAutoTuner, TritonPersistentCacheAutotuner, get_autotune_cache
    from pixal3d.modules.sparse import SparseTensor
    from pixal3d.trainers.flow_matching.mixins import image_conditioned_proj

    report.data["backend_settings"] = controlled_backend()
    results, count, manifest = [], None, None
    report.data["initialization_trials"] = results
    for name, skip in (("full_init", False), ("covered_no_init", True)):
        with report.stage("load_" + name):
            with LoadProbe(entry) as load_probe, covered_initialization(skip) as coverage:
                with OptimizationSession(OptimizationOptions(), image_conditioned_proj.DINOv3ViTModel,
                                         image_conditioned_proj.DinoV3ProjFeatureExtractor, SparseTensor):
                    with deferred_background_removal():
                        local = local_pipeline_config(bundle, Path(directory) / (name + "-config"), rembg=False)
                        pipeline = entry.init_pipeline(model_path=str(local), low_vram=True)
        trace = PrefixTrace(Path(directory) / name, Path(directory) / "full_init" if results else None,
                            max_snapshot_bytes=3 * 1024**3)
        with report.stage("neural_" + name):
            freeze = frozen_autotuning(PersistentCacheAutoTuner, TritonPersistentCacheAutotuner) if results else nullcontext()
            with fixed_splitk(PersistentCacheAutoTuner), freeze:
                with OptimizationSession(OptimizationOptions(), image_conditioned_proj.DINOv3ViTModel,
                                         image_conditioned_proj.DinoV3ProjFeatureExtractor, SparseTensor) as session:
                    session.install(pipeline, False)
                    with PrefixProbe(pipeline, trace, full=True) as probe:
                        meshes, latents = pipeline.run(image, camera_params=camera, seed=42, preprocess_image=False,
                                                      return_latent=True, pipeline_type="1024_cascade", max_num_tokens=49152)
                        trace.emit("final.shape_latent", latents[0])
                        trace.emit("final.texture_latent", latents[1])
                        if latents[2] != 1024:
                            raise RuntimeError("Unexpected geometry fallback")
                        if not results:
                            mesh = meshes[0]
                            frozen = {key: getattr(mesh, key).detach().cpu() for key in ("vertices", "faces", "attrs", "coords")}
                            frozen["layout"] = {key: (value.start, value.stop) for key, value in pipeline.pbr_attr_layout.items()}
                            torch.save(frozen, Path(directory) / "mesh.pt")
                            del frozen, mesh
                        del meshes, latents
            summary = trace.finish(count)
            count = len(trace.events)
            current = get_autotune_cache()
            if manifest is None:
                manifest = json.loads(json.dumps(current))
            elif manifest != current:
                raise RuntimeError("Kernel manifest changed during no-init validation")
            results.append({"name": name, "skip_weight_init": skip, "coverage": coverage,
                            "load_components": load_probe.timings.records, "trace": summary})
            report.write()
            del pipeline, probe
        gc.collect()
        torch.cuda.empty_cache()
    if not results[-1]["trace"]["passed"] or results[-1]["trace"]["first_nonexact"] is not None:
        raise RuntimeError("No-init failed strict full-path equivalence")
    frozen = torch.load(Path(directory) / "mesh.pt", map_location="cpu", weights_only=True)
    layout = {key: slice(*value) for key, value in frozen.pop("layout").items()}
    with report.stage("capture_export"):
        with ExportProbe(entry.o_voxel.postprocess.to_glb) as probe, capture_export_inputs(directory):
            glb = entry.o_voxel.postprocess.to_glb(vertices=frozen["vertices"].cuda(), faces=frozen["faces"].cuda(),
                attr_volume=frozen["attrs"].cuda(), coords=frozen["coords"].cuda(), attr_layout=layout,
                grid_size=1024, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], decimation_target=1000000,
                texture_size=4096, remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True)
            glb.apply_transform(np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64))
            glb.export(str(Path(directory) / "baseline.glb"), extension_webp=True)
        report.data["baseline_export"] = {"validation": validate_pbr_glb(Path(directory) / "baseline.glb"),
                                         "components": probe.timings.records, "coarse_phases": probe.coarse}
    del frozen, glb, probe
    gc.collect()
    torch.cuda.empty_cache()


def oriented_faces(faces):
    faces = np.asarray(faces)
    offsets = faces.argmin(axis=1)
    rotated = np.take_along_axis(faces, (offsets[:, None] + np.arange(3)) % 3, axis=1)
    return rotated[np.lexsort((rotated[:, 2], rotated[:, 1], rotated[:, 0]))]


def benchmark_cpu(directory, output, cores, report):
    import cv2
    import cumesh
    import platform

    directory, output = Path(directory), Path(output)
    fixtures = torch.load(directory / "charts.pt", map_location="cpu", weights_only=True)
    with np.load(directory / "inpaint.npz") as arrays:
        arguments = [arrays[key] for key in ("color", "metallic", "roughness", "alpha", "mask")]
    report.data["fixture_hashes"] = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                    for name in ("charts.pt", "inpaint.npz")}
    report.data["cpu_cores"] = cores
    report.data["opencv"] = cv2.__version__
    report.data["platform"] = platform.platform()
    report.data["charts"] = len(fixtures["charts"])
    trials, paints = [], []
    report.data.update(chart_trials=trials, inpaint_trials=paints)
    for repeat in range(2):
        with report.stage(f"atlas_{repeat}"):
            atlas = cumesh.Atlas()
            for chart in fixtures["charts"]:
                atlas.add_mesh(chart["vertices"], chart["faces"])
            started = time.perf_counter()
            atlas.compute_charts(**fixtures["options"]["compute_charts"])
            compute_seconds = time.perf_counter() - started
            started = time.perf_counter()
            atlas.pack_charts(**fixtures["options"]["pack_charts"])
            pack_seconds = time.perf_counter() - started
            valid = True
            for index, chart in enumerate(fixtures["charts"]):
                mapping, faces, uv = atlas.get_mesh(index)
                valid &= bool(torch.isfinite(uv).all() and (uv >= 0).all() and (uv <= 1).all())
                valid &= np.array_equal(oriented_faces(chart["faces"]), oriented_faces(mapping[faces.long()]))
            if not valid:
                raise RuntimeError("Atlas replay changed topology or emitted invalid UVs")
            trials.append({"repeat": repeat, "compute_seconds": compute_seconds, "pack_seconds": pack_seconds,
                           "topology_preserved": bool(valid)})
            del atlas
        report.write()
    expected = None
    for repeat in range(2):
        for workers in ((1, 2, 4) if repeat == 0 else (4, 2, 1)):
            with report.stage(f"inpaint_{repeat}_{workers}"):
                started = time.perf_counter()
                actual = inpaint_batch(*arguments, workers=workers)
                seconds = time.perf_counter() - started
                if expected is None:
                    expected = actual
                exact = all(np.array_equal(left, right) for left, right in zip(expected, actual))
                if not exact:
                    raise RuntimeError("Parallel inpaint changed output pixels")
                paints.append({"repeat": repeat, "workers": workers, "seconds": seconds, "byte_exact": bool(exact),
                               "output_sha256": [hashlib.sha256(value.tobytes()).hexdigest() for value in actual]})
                del actual
            report.write()
    atomic_json(output / "benchmark.json", report.data)
