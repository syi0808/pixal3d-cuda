import ast
import json
from pathlib import Path
import sys
from types import ModuleType

import cv2
import numpy as np
import pytest
from safetensors.torch import save_file, load_file
import torch
from torch import nn

from pixalcuda.initialization import audit_coverage, covered_initialization
from pixalcuda.inpainting import inpaint_batch, parallel_exporter
from pixalcuda.cpu_tuning import oriented_faces
from scripts.optimize_modal_native import main


def toy_exporter(base_color, metallic, roughness, alpha, mask_inv):
    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    return base_color, metallic, roughness, alpha


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
@pytest.mark.parametrize("mask_kind", ["holes", "empty", "border", "full"])
def test_parallel_inpaint_keeps_original_channels_radii_and_bytes(workers, mask_kind):
    random = np.random.default_rng(42)
    color = random.integers(0, 256, (40, 48, 3), dtype=np.uint8)
    channels = [random.integers(0, 256, (40, 48), dtype=np.uint8) for index in range(3)]
    mask = np.zeros((40, 48), dtype=np.uint8)
    if mask_kind == "holes":
        mask[4:12, 8:30] = 1
    elif mask_kind == "border":
        mask[:6] = 1
    elif mask_kind == "full":
        mask[:] = 1
    arguments = (color, *channels, mask)
    expected = toy_exporter(*arguments)
    actual = parallel_exporter(toy_exporter, workers)(*arguments)
    assert all(np.array_equal(left, right) for left, right in zip(expected, actual))
    assert all(np.array_equal(left, right) for left, right in zip(expected, inpaint_batch(*arguments, workers=workers)))


def test_export_adapter_rejects_unknown_source_and_bad_workers():
    def unknown(value):
        return value
    with pytest.raises(ValueError, match="block changed"):
        parallel_exporter(unknown, 2)
    for workers in (0, 9, True):
        with pytest.raises(ValueError):
            parallel_exporter(toy_exporter, workers)


def test_single_channel_hwc_inpaint_and_cpu_plan(tmp_path):
    color = np.zeros((30, 32, 3), np.uint8)
    channel = np.full((30, 32, 1), 100, np.uint8)
    mask = np.zeros((30, 32), np.uint8)
    mask[4:20, 5:20] = 1
    arguments = (color, channel, channel, channel, mask)
    assert all(np.array_equal(left, right) for left, right in zip(toy_exporter(*arguments), inpaint_batch(*arguments, workers=4)))
    destination = tmp_path / "dry"
    plan = main(["--source-job", "0" * 64, "--output-dir", str(destination)])
    assert not destination.exists() and not plan["execute"]
    assert plan["cores"] == [2, 4, 8] and plan["sequential"] and plan["retries"] == 0
    assert np.array_equal(oriented_faces([[1, 2, 0], [5, 3, 4]]), oriented_faces([[4, 5, 3], [0, 1, 2]]))


def test_server_reference_disables_loading_and_export_optimizations(monkeypatch):
    from dataclasses import replace
    from deploy import modal_app
    from deploy.settings import DeploySettings

    monkeypatch.setattr(modal_app, "SETTINGS", replace(DeploySettings(), skip_weight_init=True, inpaint_workers=4))
    assert modal_app.inference_options()["skip_weight_init"]
    assert modal_app.inference_options()["inpaint_workers"] == 4
    assert modal_app.inference_options(reference=True)["skip_weight_init"] is False
    assert modal_app.inference_options(reference=True)["inpaint_workers"] == 1


def test_coverage_requires_every_parameter_and_buffer():
    model = nn.Linear(3, 4)
    model.register_buffer("rope_phases", torch.arange(3))
    shapes = {name: list(value.shape) for name, value in model.named_parameters()}
    with pytest.raises(ValueError, match="coverage"):
        audit_coverage(model, shapes, "SparseStructureFlowModel")
    shapes["rope_phases"] = [3]
    record = audit_coverage(model, shapes, "SparseStructureFlowModel")
    assert record["derived_buffers"] == []
    with pytest.raises(ValueError, match="coverage"):
        audit_coverage(model, {"weight": [4, 3]}, "SparseStructureFlowModel")
    with pytest.raises(ValueError, match="coverage"):
        audit_coverage(model, {**shapes, "weight": [3, 4]}, "SparseStructureFlowModel")


@pytest.mark.parametrize("complete", [False, True])
def test_guarded_no_init_delegates_load_and_restores_after_failure(tmp_path, monkeypatch, complete):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4)
            self.register_buffer("rope_phases", torch.arange(3))
            nn.init.normal_(self.linear.weight)

    path = tmp_path / "model"
    Path(str(path) + ".json").write_text(json.dumps({"name": "SparseStructureFlowModel", "args": {}}))
    state = {"linear.weight": torch.ones(4, 3), "linear.bias": torch.zeros(4), "rope_phases": torch.arange(3)}
    if not complete:
        state.pop("linear.bias")
    save_file(state, str(path) + ".safetensors")
    models = ModuleType("pixal3d.models")
    models.SparseStructureFlowModel = Model

    def load(path, **options):
        model = models.SparseStructureFlowModel()
        model.load_state_dict(load_file(path + ".safetensors"), strict=False)
        return model
    models.from_pretrained = load
    parent = ModuleType("pixal3d")
    parent.models = models
    monkeypatch.setitem(sys.modules, "pixal3d", parent)
    original_initializer, original_init = nn.init.normal_, Model.__init__
    if complete:
        with covered_initialization() as records:
            model = models.from_pretrained(str(path))
        assert torch.equal(model.linear.weight, state["linear.weight"])
        assert torch.equal(model.rope_phases, torch.arange(3))
        assert records[0]["applied"] and records[0]["skipped_calls"] > 0
    else:
        with pytest.raises(ValueError, match="coverage"):
            with covered_initialization():
                models.from_pretrained(str(path))
    assert nn.init.normal_ is original_initializer and Model.__init__ is original_init
    assert models.from_pretrained is load
