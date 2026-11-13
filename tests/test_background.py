import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from pixalcuda.background import OptionalBackgroundRemover, deferred_background_removal
from pixalcuda.upstream_weights import UpstreamStore, local_pipeline_config


def test_disabled_remover_never_loads_weights_even_when_pipeline_moves_devices():
    def forbidden(**options):
        raise AssertionError("Disabled matting must not construct or download a model")
    remover = OptionalBackgroundRemover(forbidden)
    remover.cuda().cpu().to("cuda")
    assert remover.model is None
    with pytest.raises(RuntimeError, match="disabled"):
        remover(Image.new("RGB", (8, 8)))
    assert remover.model is None


def test_non_boolean_opt_in_is_rejected_before_downloads(monkeypatch):
    def forbidden(*arguments, **options):
        raise AssertionError("Invalid opt-in must not start downloads")
    monkeypatch.setattr("pixalcuda.upstream_weights.snapshot_download", forbidden)
    with pytest.raises(ValueError, match="rembg must be boolean"):
        UpstreamStore().resolve(rembg=1)


def test_cli_download_excludes_rmbg_unless_explicitly_requested(monkeypatch):
    from pixalcuda.cli import main
    calls = []
    store = SimpleNamespace(resolve=lambda **options: calls.append(options) or {})
    monkeypatch.setattr("pixalcuda.upstream_weights.UpstreamStore", lambda **options: store)
    main(["download"])
    main(["download", "--include-rembg"])
    assert calls == [{"camera": True, "rembg": False}, {"camera": True, "rembg": True}]


def test_explicit_remover_loads_once_on_first_use_and_obeys_offload():
    events = []
    class Model:
        def __init__(self, model_name):
            events.append(("load", model_name))

        def to(self, device):
            events.append(("to", device))

        def __call__(self, image):
            events.append(("infer", image.size))
            return image.convert("RGBA")
    remover = OptionalBackgroundRemover(Model, "/pinned/rembg")
    remover.cuda()
    assert not events
    image = Image.new("RGB", (8, 8))
    assert remover(image).mode == "RGBA"
    remover.cpu().cuda()
    remover(image)
    assert [event for event in events if event[0] == "load"] == [("load", "/pinned/rembg")]
    assert ("to", "cpu") in events and events.count(("infer", (8, 8))) == 2


@pytest.mark.parametrize("fail", [False, True])
def test_rembg_factory_is_restored_after_pipeline_initialization(monkeypatch, fail):
    def forbidden(**options):
        raise AssertionError("Factory must remain deferred")
    module = SimpleNamespace(BiRefNet=forbidden)
    monkeypatch.setitem(sys.modules, "pixal3d.pipelines.rembg", module)
    def initialize():
        with deferred_background_removal() as removers:
            remover = module.BiRefNet(model_name=None)
            assert removers == [remover] and remover.model is None
            if fail:
                raise ValueError("controlled initialization failure")
    if fail:
        with pytest.raises(ValueError, match="controlled"):
            initialize()
    else:
        initialize()
    assert module.BiRefNet is forbidden


def test_default_config_ignores_even_preexisting_rembg_weights(tmp_path):
    source = tmp_path / "snapshot"
    (source / "ckpts").mkdir(parents=True)
    config = {"args": {"rembg_model": {"name": "BiRefNet", "args": {"model_name": "briaai/RMBG-2.0"}}}}
    (source / "pipeline.json").write_text(json.dumps(config))
    bundle = {"pixal": str(source), "config_file": "pipeline.json", "rembg": "/already/cached/rembg"}
    destination = local_pipeline_config(bundle, tmp_path / "config")
    assert json.loads((destination / "pipeline.json").read_text())["args"]["rembg_model"]["args"]["model_name"] is None
    assert json.loads((source / "pipeline.json").read_text()) == config
    with pytest.raises(ValueError, match="prefetched"):
        local_pipeline_config({**bundle, "rembg": None}, tmp_path / "missing", rembg=True)


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_upstream_constructor_and_alpha_preprocessing_do_not_load_rembg(tmp_path, monkeypatch, enabled):
    source = Path(__file__).resolve().parents[1] / "outputs/upstream-Pixal3D-review/pixal3d/pipelines/pixal3d_image_to_3d.py"
    if not source.exists():
        pytest.fail("Pinned upstream source fixture is required, as in the optimization contracts")
    tree = ast.parse(source.read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Pixal3DImageTo3DPipeline")
    original.bases = [ast.Name(id="Base", ctx=ast.Load())]
    original.body = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in ("from_pretrained", "preprocess_image")]
    def forbidden(**options):
        raise AssertionError("Original pipeline init/RGBA preprocessing must not load RMBG-2.0")
    module = SimpleNamespace(BiRefNet=forbidden)
    monkeypatch.setitem(sys.modules, "pixal3d.pipelines.rembg", module)
    class Base:
        @classmethod
        def from_pretrained(cls, path, config_file):
            pipeline = cls()
            pipeline._pretrained_args = json.loads((Path(path) / config_file).read_text())["args"]
            return pipeline
    namespace = {"Base": Base, "Image": Image, "np": np, "rembg": module,
                 "samplers": SimpleNamespace(ToySampler=lambda: object())}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[original], type_ignores=[])), str(source), "exec"), namespace)
    snapshot = tmp_path / "snapshot"
    (snapshot / "ckpts").mkdir(parents=True)
    sampler = {"name": "ToySampler", "args": {}, "params": {}}
    config = {"args": {"rembg_model": {"name": "BiRefNet", "args": {"model_name": "remote"}},
                       "sparse_structure_sampler": sampler, "shape_slat_sampler": sampler, "tex_slat_sampler": sampler,
                       "shape_slat_normalization": {}, "tex_slat_normalization": {}}}
    (snapshot / "pipeline.json").write_text(json.dumps(config))
    bundle = {"pixal": str(snapshot), "config_file": "pipeline.json"}
    if enabled:
        bundle["rembg"] = "/pinned/rembg"
    local = local_pipeline_config(bundle, tmp_path / "config", rembg=enabled)
    with deferred_background_removal() as removers:
        pipeline = namespace["Pixal3DImageTo3DPipeline"].from_pretrained(str(local))
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    image.paste(Image.new("RGBA", (16, 16), (100, 50, 20, 255)), (8, 8))
    output = pipeline.preprocess_image(image)
    assert output.mode == "RGB" and output.width < image.width
    assert removers == [pipeline.rembg_model] and removers[0].model is None
    assert module.BiRefNet is forbidden
