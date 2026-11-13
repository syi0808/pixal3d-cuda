from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import trimesh

from pixalcuda import pipeline, upstream_sources, upstream_worker, upstream_weights
from pixalcuda.cameras import orbit_camera_metadata, uploaded_camera_metadata


def toy_glb(destination):
    mesh = trimesh.creation.box()
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new("RGB", (4, 4), "red"),
                                                  metallicRoughnessTexture=Image.new("RGB", (4, 4), "white"))
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(mesh.vertices), 2)), material=material)
    mesh.export(destination)


class Report:
    def __init__(self, directory):
        self.directory = directory
        self.data = {"stages": []}

    @contextmanager
    def stage(self, name):
        yield
        self.data["stages"].append({"name": name})

    def write(self):
        (self.directory / "report.json").write_text(json.dumps(self.data))


@pytest.mark.parametrize("multiview", [False, True])
def test_native_worker_delegates_models_conditioning_and_export(tmp_path, monkeypatch, multiview):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    config_file = "pipeline_mv.json" if multiview else "pipeline.json"
    config = {"args": {"models": {"flow": "ckpts/native"}, "rembg_model": {"args": {"model_name": "remote"}}}}
    (snapshot / config_file).write_text(json.dumps(config))
    image = tmp_path / "input.png"
    Image.new("RGBA", (20, 32), (80, 30, 20, 128)).save(image)
    identity = {"inputs": [{"path": str(image)}], "multiview": multiview, "remove_background": True,
                "low_vram": True, "geometry_resolution": 1024, "texture_size": 4096, "target_faces": 1000000,
                "seed": 42, "max_tokens": 49152, "camera": "fixed", "fov_degrees": 49.13, "save_reference": True}
    if multiview:
        identity["inputs"] *= 4
        identity["cameras"] = uploaded_camera_metadata(orbit_camera_metadata(), 4)
    calls = []
    mesh = SimpleNamespace(vertices=torch.zeros(4, 3), faces=torch.zeros(2, 3, dtype=torch.int64),
                           coords=torch.zeros(4, 3, dtype=torch.int32), attrs=torch.ones(4, 6))
    latent = SimpleNamespace(feats=torch.zeros(4, 32), coords=torch.zeros(4, 4, dtype=torch.int32))
    def run(*arguments, **options):
        calls.append(("run_mv" if multiview else "run", options))
        return [mesh], (latent, latent, 1024)
    native = SimpleNamespace(preprocess_image=lambda image: calls.append(("native_preprocess", image.size)) or image.convert("RGB"),
                             run=run, run_mv=run, pbr_attr_layout={"base_color": slice(0, 3)})
    configs = {name: {"model_name": "remote", "naf_target_size": target, "image_size": size, "grid_resolution": grid}
               for name, target, size, grid in (("ss", 512, 512, 16), ("shape_512", 512, 512, 32),
                                              ("shape_1024", 512, 1024, 64), ("tex_1024", 1024, 1024, 64))}
    def init(**options):
        calls.append(("init", options))
        assert Path(options["model_path"]).is_dir()
        return native
    def load_views(directory, rembg):
        metadata = json.loads((Path(directory) / "transforms.json").read_text())
        assert len(metadata["frames"]) == 4
        assert all((Path(directory) / frame["file_path"]).is_file() for frame in metadata["frames"])
        return {"native_views": metadata}
    entry = SimpleNamespace(IMAGE_COND_CONFIGS=configs, init_pipeline=init,
                            distance_from_fov=lambda *arguments: {"distance_from_x": 2.0},
                            make_rembg=lambda native: object(), load_views=load_views, check_main_view=lambda views: None)
    transform = []
    glb = SimpleNamespace(apply_transform=lambda matrix: transform.append(matrix), export=lambda path, **options: toy_glb(path))
    def export(**options):
        calls.append(("export", options))
        return glb
    assert not (Path(pipeline.__file__).parent / "flow.py").exists()
    assert not (Path(pipeline.__file__).parent / "export.py").exists()
    bundle = {"pixal": str(snapshot), "config_file": config_file, "rembg": "/pinned/rembg", "dino": "/pinned/dino"}
    report = Report(tmp_path)
    result = upstream_worker.execute({"identity": identity, "bundle": bundle}, tmp_path, entry, export, report)
    assert result.is_file() and report.data["status"] == "complete"
    assert report.data["conditioning"]["tex_1024"]["naf_target_size"] == 1024
    assert all(config["model_name"] == "/pinned/dino" for config in configs.values())
    generation = next(options for name, options in calls if name in ("run", "run_mv"))
    assert generation["pipeline_type"] == "1024_cascade" and generation["return_latent"] is True
    assert "shape_slat_sampler_params" not in generation
    export_options = next(options for name, options in calls if name == "export")
    assert export_options["texture_size"] == 4096 and export_options["decimation_target"] == 1000000
    assert export_options["remesh"] is True and export_options["remesh_project"] == 0
    assert report.data["postprocess_device"] == "cuda"
    torch.testing.assert_close(torch.tensor(transform[0]), torch.tensor([[-1., 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=torch.float64))
    reference = torch.load(tmp_path / "reference.pt", weights_only=True)
    assert reference["actual_geometry_resolution"] == 1024 and reference["shape"]["features"].device.type == "cpu"


@pytest.mark.parametrize("capability", [(7, 5), (6, 1)])
def test_native_baseline_refuses_old_hardware_without_precision_or_backend_fallback(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    with pytest.raises(RuntimeError, match="No legacy or GPU fallback"):
        upstream_worker.require_hardware()


def test_native_preflight_refuses_cpu_before_download(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="needs CUDA"):
        upstream_worker.require_hardware()


@pytest.mark.parametrize("options", [{"geometry_resolution": 512}, {"geometry_resolution": True},
                                     {"texture_size": 123}, {"target_faces": 0}])
def test_native_options_never_silently_map_legacy_settings(options):
    with pytest.raises(ValueError):
        pipeline.validate_options(**options)


def test_source_verification_rejects_wrong_revision_or_edits(tmp_path, monkeypatch):
    expected = upstream_sources.lock()["sources"]["pixal3d"]["revision"]
    for revision, changes in (("0" * 40, ""), (expected, " M inference.py")):
        monkeypatch.setattr(upstream_sources, "git", lambda directory, *arguments: revision if arguments[0] == "rev-parse" else changes)
        with pytest.raises(ValueError, match="refusing"):
            upstream_sources.verify_source("pixal3d", tmp_path)
    monkeypatch.setattr(upstream_sources, "git", lambda directory, *arguments: expected if arguments[0] == "rev-parse" else "")
    assert upstream_sources.verify_source("pixal3d", tmp_path) == tmp_path / "pixal3d"


def test_build_uses_explicit_native_architecture_and_never_requests_cloud_gpu(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(upstream_sources, "verify_source", lambda name, root: Path(root) / name)
    monkeypatch.setattr(upstream_sources.subprocess, "run", lambda command, **options: commands.append((command, options)))
    upstream_sources.build(tmp_path, "8.9")
    assert len(commands) == 5
    assert all(options["env"]["NATTEN_CUDA_ARCH"] == "8.9" for command, options in commands)
    assert all("modal" not in command for command, options in commands)
    with pytest.raises(ValueError, match="T4"):
        upstream_sources.build(tmp_path, "7.5")


@pytest.mark.parametrize("multiview", [False, True])
def test_offline_native_prefetch_includes_configs_and_official_backbone_distribution(tmp_path, monkeypatch, multiview):
    calls = []
    def snapshot(repo, **options):
        calls.append((repo, options))
        for pattern in options["allow_patterns"]:
            if "*" not in pattern:
                path = tmp_path / pattern
                path.parent.mkdir(exist_ok=True, parents=True)
                path.touch()
        return str(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(upstream_weights, "snapshot_download", snapshot)
    monkeypatch.setattr(upstream_weights, "naf_weights", lambda: tmp_path / "naf.pth")
    monkeypatch.setattr(upstream_weights, "hf_hub_download", lambda *arguments, **options: str(tmp_path / "moge.pt"))
    result = upstream_weights.UpstreamStore(multiview=multiview).resolve()
    assert all(options["local_files_only"] is True for repo, options in calls)
    assert calls[1][0] == "camenduru/dinov3-vitl16-pretrain-lvd1689m"
    assert len(calls[0][1]["allow_patterns"]) == 15
    assert result["config_file"] == ("pipeline_mv.json" if multiview else "pipeline.json")
    assert ("moge" in result) != multiview


def test_parent_process_completion_resume_and_failure_are_not_legacy_stage_resume(tmp_path, monkeypatch):
    source = tmp_path / "input.png"
    Image.new("RGB", (16, 16), "red").save(source)
    monkeypatch.setattr(pipeline, "verify_source", lambda name, root: Path(root) / name)
    monkeypatch.setattr(pipeline, "preflight", lambda: None)
    store = SimpleNamespace(multiview=False, resolve=lambda **options: {"pixal": "controlled"})
    commands = []
    def run(command, **options):
        commands.append((command, options))
        directory = Path(command[-1]).parent
        toy_glb(directory / "result.glb")
        report_path = directory / "report.json"
        report = json.loads(report_path.read_text())
        report.update(status="complete", validation=pipeline.validate_pbr_glb(directory / "result.glb"))
        report_path.write_text(json.dumps(report))
    monkeypatch.setattr(pipeline.subprocess, "run", run)
    native = pipeline.Pipeline(store=store)
    result = native.generate(source, tmp_path / "run")
    assert pipeline.validate_pbr_glb(result)["embedded_pbr"]
    assert commands[0][1]["env"]["SPARSE_CONV_BACKEND"] == "flex_gemm"
    assert commands[0][1]["env"]["HF_HUB_OFFLINE"] == "1"
    assert native.generate(source, tmp_path / "run", resume=True) == result and len(commands) == 1
    with pytest.raises(ValueError, match="changed"):
        native.generate(source, tmp_path / "run", seed=1, resume=True)
    def fail(*arguments, **options):
        raise pipeline.subprocess.TimeoutExpired("controlled timeout", 1)
    monkeypatch.setattr(pipeline.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="inspect"):
        native.generate(source, tmp_path / "failure")
    assert json.loads((tmp_path / "failure" / "report.json").read_text())["status"] == "failed"
    with pytest.raises(ValueError, match="stage resume"):
        native.generate(source, tmp_path / "failure", resume=True)


def test_default_cli_is_native_and_removed_entry_points_are_rejected(tmp_path, monkeypatch):
    from pixalcuda.cli import main
    calls = []
    class Native:
        def __init__(self, **options):
            calls.append(options)

        def generate(self, image, output, **options):
            calls.append(options)
            return tmp_path / "controlled.glb"
    monkeypatch.setattr(pipeline, "Pipeline", Native)
    main(["generate", "--image", str(tmp_path / "input.png")])
    assert calls[0]["low_vram"] is True and calls[0]["attention_backend"] == "sdpa"
    assert calls[0]["optimizations"].share_dino and calls[0]["optimizations"].sparse_projection
    assert calls[1]["geometry_resolution"] == 1024 and calls[1]["texture_size"] == 4096
    calls.clear()
    main(["generate", "--image", str(tmp_path / "input.png"), "--reference"])
    assert not calls[0]["optimizations"].share_dino and calls[0]["optimizations"].conditioning_cache_mb == 0
    with pytest.raises(SystemExit):
        main(["generate", "--image", "not-opened.png", "--reference", "--conditioning-cache-mb", "0"])
    for arguments in (["legacy", "doctor"], ["export", "--output-dir", str(tmp_path)]):
        with pytest.raises(SystemExit):
            main(arguments)


@pytest.mark.parametrize("arguments", [
    ["--geometry-resolution", "512"], ["--conditioning-cache-mb", "-1"], ["--defer-export"],
    ["--num-views", "0"], ["--num-views", "2"], ["--transforms", "not-opened.json"],
])
def test_cli_rejects_removed_flags_or_invalid_modes_before_execution(arguments):
    from pixalcuda.cli import main
    with pytest.raises(SystemExit) as error:
        main(["generate", "--image", "not-opened.png", *arguments])
    assert error.value.code == 2


def test_benchmark_is_dry_run_without_model_downloads_or_cuda(tmp_path, monkeypatch):
    from scripts import benchmark_upstream
    def forbidden(*arguments, **options):
        raise AssertionError("Dry-run must not construct pipelines or initialize CUDA")
    monkeypatch.setattr(benchmark_upstream, "Pipeline", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    plan = benchmark_upstream.main(["--image", str(tmp_path / "not-opened.png"), "--output-dir", str(tmp_path / "not-created")])
    assert plan["execute"] is False and not (tmp_path / "not-created").exists()


def test_explicit_no_matting_preserves_framing_and_composites_alpha(tmp_path):
    image = tmp_path / "rgba.png"
    Image.new("RGBA", (12, 20), (200, 100, 50, 128)).save(image)
    def forbidden(*arguments, **options):
        raise AssertionError("Explicit no-matting input must not use the original automatic crop/matting")
    native = SimpleNamespace(preprocess_image=forbidden)
    entry = SimpleNamespace(distance_from_fov=lambda *arguments: {"distance_from_x": 2.0})
    identity = {"inputs": [{"path": str(image)}], "remove_background": False, "camera": "fixed", "fov_degrees": 49.13}
    result, camera = upstream_worker.single_input(entry, native, identity, {}, tmp_path)
    assert result.size == (12, 20) and result.getpixel((0, 0)) == (100, 50, 25)
    assert camera["distance"] == 2.0
