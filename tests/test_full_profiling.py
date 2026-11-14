import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from pixalcuda.full_profiling import ExportProbe, LoadProbe, Timings, frozen_autotuning
from pixalcuda.profiling import PrefixProbe, PrefixTrace, tensor_tree
from scripts.profile_modal_native import main


def test_full_plan_is_bounded_and_does_not_contact_modal(tmp_path):
    destination = tmp_path / "plan"
    plan = main(["--source-job", "0" * 64, "--output-dir", str(destination), "--scope", "full"])
    assert not destination.exists() and not plan["execute"]
    assert plan["gpu"] == "L4" and plan["gpu_timeout_seconds"] == 1800
    assert plan["frozen_export_repeats"] == 2 and plan["automatic_retries"] == 0
    assert plan["profiles"] == ["reference_first", "reference_repeat", "optimized"]


def test_kernel_freeze_rejects_new_tuning_and_restores_on_failure():
    class Persistent:
        def _benchmark(self):
            return 1

    class Triton:
        def _bench(self):
            return 2

    originals = Persistent._benchmark, Triton._bench
    with frozen_autotuning(Persistent, Triton):
        for method in (Persistent()._benchmark, Triton()._bench):
            with pytest.raises(RuntimeError, match="frozen"):
                method()
    assert (Persistent._benchmark, Triton._bench) == originals


def test_mesh_snapshot_includes_material_voxels_and_geometry():
    mesh = SimpleNamespace(vertices=torch.zeros(4, 3), faces=torch.zeros(2, 3, dtype=torch.int32),
                           attrs=torch.ones(5, 6), coords=torch.zeros(5, 3, dtype=torch.int32))
    assert set(tensor_tree(mesh)) == {"value.vertices", "value.faces", "value.attrs", "value.coords"}
    with pytest.raises(RuntimeError):
        with Timings().measure("failed"):
            raise RuntimeError()


def test_full_probe_routes_hr_texture_steps_and_restores(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state", torch.get_rng_state)

    class Sampler:
        def sample_once(self, model, sample, **options):
            return SimpleNamespace(pred_x_prev=sample + 1)

        def sample(self, model, noise, **options):
            sample = noise
            for index in range(2):
                sample = self.sample_once(model, sample, **options).pred_x_prev
            return SimpleNamespace(samples=sample)

    decoder = nn.Module()
    decoder.upsample = lambda value: value
    names = ("sparse_structure_flow_model", "shape_slat_flow_model_512", "shape_slat_flow_model_1024", "tex_slat_flow_model_1024")
    pipeline = SimpleNamespace(models={name: object() for name in names}, sparse_structure_sampler=Sampler(),
        shape_slat_sampler=Sampler(), tex_slat_sampler=Sampler(), image_cond_model_shape_1024=object(),
        image_cond_model_tex_1024=object(), get_proj_cond_ss=lambda image: {"cond": torch.zeros(1)},
        get_proj_cond_shape=lambda image_cond_model, image, coords: {"cond": torch.zeros(1)},
        sample_sparse_structure=lambda value: value, decode_shape_slat=lambda value: value,
        decode_tex_slat=lambda value: value, decode_latent=lambda value: value)
    pipeline.models["shape_slat_decoder"] = decoder
    original = pipeline.get_proj_cond_shape
    trace = PrefixTrace(tmp_path / "full")
    with PrefixProbe(pipeline, trace, full=True):
        for extractor in (pipeline.image_cond_model_shape_1024, pipeline.image_cond_model_tex_1024):
            pipeline.get_proj_cond_shape(extractor, [], torch.zeros(2, 4, dtype=torch.int32))
        for name, sampler in (("shape_slat_flow_model_1024", pipeline.shape_slat_sampler),
                              ("tex_slat_flow_model_1024", pipeline.tex_slat_sampler)):
            sampler.sample(pipeline.models[name], torch.zeros(2), concat_cond=torch.ones(2))
        pipeline.decode_latent(torch.ones(3))
    assert pipeline.get_proj_cond_shape is original
    events = {event["name"] for event in trace.events}
    assert {"conditioning.shape_hr", "conditioning.texture", "shape_hr.step.01", "texture.step.01",
            "texture.concat_condition", "decode_latent"} <= events


def test_load_probe_measures_delegated_constructor_mapping_and_application(tmp_path, monkeypatch):
    model_path = tmp_path / "flow"
    Path(str(model_path) + ".json").write_text(json.dumps({"name": "TinyModel"}))

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(2, 2))
            nn.init.uniform_(self.weight)

    class Dino(nn.Module):
        @classmethod
        def from_pretrained(cls, path):
            return cls()

    class Extractor:
        def __init__(self, config):
            self.model = Dino.from_pretrained("private")
            self.image_size = config["image_size"]
            self.grid_resolution = config["grid_resolution"]

        def _load_naf(self):
            return None

    safetensors = ModuleType("safetensors")
    safetensors.torch = SimpleNamespace(load_file=lambda path: {"weight": torch.ones(2, 2)})
    models = ModuleType("pixal3d.models")
    models.TinyModel = TinyModel

    def load(path, **options):
        model = models.TinyModel()
        model.load_state_dict(safetensors.torch.load_file(str(path) + ".safetensors"))
        return model
    models.from_pretrained = load
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors.torch)
    monkeypatch.setitem(sys.modules, "pixal3d", SimpleNamespace(models=models))
    feature = SimpleNamespace(DINOv3ViTModel=Dino, DinoV3ProjFeatureExtractor=Extractor)
    monkeypatch.setitem(sys.modules, "pixal3d.trainers.flow_matching.mixins", SimpleNamespace(image_conditioned_proj=feature))
    entry = SimpleNamespace(build_image_cond_model=lambda config: Extractor(config),
                            IMAGE_COND_CONFIGS={"ss": {"image_size": 512, "grid_resolution": 16},
                                               "shape_1024": {"image_size": 1024, "grid_resolution": 64, "naf_target_size": 512},
                                               "tex_1024": {"image_size": 1024, "grid_resolution": 64, "naf_target_size": 1024}})
    initializer = nn.init.uniform_
    with LoadProbe(entry) as probe:
        result = models.from_pretrained(str(model_path))
        entry.build_image_cond_model(entry.IMAGE_COND_CONFIGS["ss"])._load_naf()
        for name in ("shape_1024", "tex_1024"):
            entry.build_image_cond_model(entry.IMAGE_COND_CONFIGS[name])._load_naf()
    assert torch.equal(result.weight, torch.ones(2, 2))
    assert models.from_pretrained is load and nn.init.uniform_ is initializer
    records = probe.timings.records
    assert {"model.flow.construct", "model.flow.checkpoint_mapping", "model.flow.apply_checkpoint",
            "model.flow.total", "image.ss.dino_load", "image.ss.naf_load", "initialization.uniform_"} <= records.keys()
    assert records["image.shape_1024.naf_load"]["calls"] == records["image.tex_1024.naf_load"]["calls"] == 1


def test_export_probe_restores_wrappers_and_separates_coarse_phases(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    mesh = type("Mesh", (), {name: lambda *arguments, **options: None for name in
        ("fill_holes", "simplify", "uv_unwrap", "compute_charts", "compute_vertex_normals")})
    atlas = type("Atlas", (), {name: lambda *arguments, **options: None for name in
        ("add_mesh", "compute_charts", "pack_charts", "get_mesh")})
    monkeypatch.setitem(sys.modules, "cumesh", SimpleNamespace(CuMesh=mesh, Atlas=atlas))
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(inpaint=lambda *arguments: None))
    rasterizer = SimpleNamespace(rasterize=lambda: None, interpolate=lambda: None)
    monkeypatch.setitem(sys.modules, "nvdiffrast", SimpleNamespace(torch=rasterizer))
    monkeypatch.setitem(sys.modules, "nvdiffrast.torch", rasterizer)
    original_tqdm, original_sampling = object(), lambda: None
    namespace = {"tqdm": original_tqdm, "grid_sample_3d": original_sampling}
    exec("def exporter():\n    return None", namespace)
    original = mesh.uv_unwrap
    with pytest.raises(RuntimeError):
        with ExportProbe(namespace["exporter"]) as probe:
            progress = namespace["tqdm"](total=6, desc="Initial")
            progress.update(1)
            progress.set_description("UV")
            mesh().uv_unwrap()
            progress.update(1)
            raise RuntimeError()
    assert mesh.uv_unwrap is original
    assert namespace["tqdm"] is original_tqdm and namespace["grid_sample_3d"] is original_sampling
    assert [phase["phase"] for phase in probe.coarse] == ["Initial", "UV"]
