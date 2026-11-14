import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pixalcuda.profiling import (PrefixComplete, PrefixProbe, PrefixTrace, alias_loaded_backbones,
                                fixed_splitk, tensor_comparison, tensor_digest, tensor_tree)
from scripts.profile_modal_native import main


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16, torch.int32, torch.bool])
def test_digest_handles_strides_and_storage_dtypes(dtype):
    value = torch.arange(12).reshape(3, 4).to(dtype).transpose(0, 1)
    assert tensor_digest(value) == tensor_digest(value.contiguous())
    assert tensor_comparison(value, value.clone())["passed"]


def test_comparison_rejects_structure_nonfinite_and_integer_drift():
    assert not tensor_comparison(torch.zeros(2), torch.zeros(3))["passed"]
    assert not tensor_comparison(torch.zeros(2), torch.zeros(2).half())["passed"]
    assert not tensor_comparison(torch.tensor([float("nan")]), torch.tensor([float("nan")]))["passed"]
    assert not tensor_comparison(torch.tensor([1]), torch.tensor([2]))["passed"]
    assert tensor_comparison(torch.empty(0), torch.empty(0))["max_abs"] == 0
    result = tensor_comparison(torch.ones(3), torch.ones(3) + 1e-7)
    assert result["passed"] and not result["exact"]


def test_sparse_tree_is_explicit_and_does_not_drop_unknown_values():
    sparse = SimpleNamespace(feats=torch.ones(2, 4), coords=torch.zeros(2, 4, dtype=torch.int32))
    assert set(tensor_tree({"condition": [sparse, None]})) == {"value.condition.0.features", "value.condition.0.coordinates"}
    with pytest.raises(TypeError, match="Unsupported"):
        tensor_tree(object())


def test_private_trace_snapshots_rng_and_first_real_divergence(tmp_path):
    baseline = PrefixTrace(tmp_path / "baseline")
    state = torch.get_rng_state()
    baseline.emit("input.rng", state)
    baseline.emit("condition", torch.ones(2))
    baseline.emit("sample", torch.zeros(2))
    assert torch.equal(state, torch.get_rng_state())
    assert baseline.finish()["passed"] is None
    candidate = PrefixTrace(tmp_path / "candidate", baseline.directory)
    candidate.emit("input.rng", state ^ 1)
    candidate.emit("condition", torch.ones(2))
    candidate.emit("sample", torch.ones(2))
    result = candidate.finish(3)
    assert result["first_failure"] == "input.rng"
    assert result["first_activation_failure"] == "sample"
    assert not result["passed"]
    assert not list(candidate.directory.glob("*.pt"))
    assert len(list(baseline.directory.glob("*.pt"))) == 3


def test_trace_does_not_accept_missing_or_changed_events(tmp_path):
    baseline = PrefixTrace(tmp_path / "baseline")
    baseline.emit("first", torch.ones(2))
    candidate = PrefixTrace(tmp_path / "candidate", baseline.directory)
    assert not candidate.finish(1)["passed"]
    changed = PrefixTrace(tmp_path / "changed", baseline.directory)
    changed.emit("different", torch.ones(2))
    assert changed.finish(1)["first_failure"] == "different"


def test_trace_budget_is_a_failure_not_silent_truncation(tmp_path):
    trace = PrefixTrace(tmp_path / "trace", max_snapshot_bytes=3)
    with pytest.raises(RuntimeError, match="budget"):
        trace.emit("large", torch.ones(2))
    assert not trace.events


def test_backbone_aliasing_restores_registered_modules_after_failure():
    extractors = [nn.Module() for _ in range(4)]
    for extractor in extractors:
        extractor.model = nn.Linear(2, 2)
    pipeline = SimpleNamespace(**dict(zip(("image_cond_model_ss", "image_cond_model_shape_512", "image_cond_model_shape_1024", "image_cond_model_tex_1024"), extractors)))
    originals = [extractor.model for extractor in extractors]
    with pytest.raises(RuntimeError):
        with alias_loaded_backbones(pipeline, True):
            assert all(extractor.model is originals[0] for extractor in extractors)
            raise RuntimeError()
    assert all(extractor.model is original and extractor._modules["model"] is original for extractor, original in zip(extractors, originals))


def test_splitk_control_is_scoped_and_targets_only_the_observed_key():
    class Tuner:
        configs = [{"SPLITK": 1}, {"SPLITK": 2}]
        key_fn = staticmethod(lambda key: key)

        def __init__(self):
            def kernel(key, **options):
                return options["SPLITK"]
            kernel.__name__ = "sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk"
            self.kernel = kernel

        def __call__(self, key):
            return "autotuned"

    original = Tuner.__call__
    tuner = Tuner()
    with fixed_splitk(Tuner, alternate=True) as choices:
        assert tuner("(2^13, 512, 512, 27)") == 2
        assert tuner("other") == 1
    assert Tuner.__call__ is original
    assert choices["(2^13, 512, 512, 27)"] == {"SPLITK": 2}
    with pytest.raises(RuntimeError, match="unavailable"):
        with fixed_splitk(Tuner, alternate=True):
            tuner.configs = [{"SPLITK": 1}]
            tuner("(2^13, 512, 512, 27)")
    assert Tuner.__call__ is original


def test_prefix_stop_occurs_before_hr_condition_and_restores_methods(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    decoder = nn.Module()
    decoder.upsample = lambda value: value
    sampler = SimpleNamespace(sample=lambda model, noise, **options: SimpleNamespace(samples=noise), sample_once=lambda *arguments: None)
    pipeline = SimpleNamespace(models={"shape_slat_decoder": decoder}, sparse_structure_sampler=sampler,
                               shape_slat_sampler=SimpleNamespace(**vars(sampler)), image_cond_model_shape_1024=object(),
                               get_proj_cond_ss=lambda image: {}, get_proj_cond_shape=lambda image_cond_model, image, coords: {},
                               sample_sparse_structure=lambda *arguments: torch.zeros(1, 4, dtype=torch.int32))
    original = pipeline.get_proj_cond_shape
    trace = PrefixTrace(tmp_path / "trace")
    with PrefixProbe(pipeline, trace):
        with pytest.raises(PrefixComplete):
            pipeline.get_proj_cond_shape(pipeline.image_cond_model_shape_1024, [], torch.zeros(2, 4, dtype=torch.int32))
    assert pipeline.get_proj_cond_shape is original
    assert trace.events[0]["name"] == "coordinates.shape_hr"


def test_modal_profiling_dry_run_has_no_gpu_or_artifacts(tmp_path, capsys):
    destination = tmp_path / "dry-run"
    result = main(["--source-job", "0" * 64, "--output-dir", str(destination)])
    assert not destination.exists()
    assert result["execute"] is False
    assert result["gpu"] == "L4" and result["gpu_timeout_seconds"] == 1200
    assert result["automatic_retries"] == 0
    assert json.loads(capsys.readouterr().out)["private_tensors"]
