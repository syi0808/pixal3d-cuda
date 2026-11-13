import json

import pytest
import torch

from scripts.benchmark_cuda_ops import main


def test_operator_benchmark_cpu_smoke_is_explicit_and_all_cases_are_checked(tmp_path):
    output = tmp_path / "operators.json"
    report = main(["--device", "cpu", "--output", str(output), "--tokens", "16", "--channels", "24", "--heads", "3",
                   "--projection-channels", "16", "--sparse-voxels", "8", "--sparse-channels", "4", "--sparse-rows", "4",
                   "--coordinate-voxels", "8", "--query-chunks", "4", "8", "--warmup", "0", "--repeats", "1"])
    assert json.loads(output.read_text())["status"] == "complete"
    assert report["device"] == "cpu" and report["scope"].startswith("Synthetic")
    assert len(report["cases"]) == 11
    assert all(case["status"] == "complete" for case in report["cases"])
    assert all("cuda_interval_ms" not in case for case in report["cases"])
    assert all("max_abs_error" in case for case in report["cases"] if case["name"] != "neighbor-map")


def test_operator_benchmark_refuses_fake_cuda_before_creating_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="real NVIDIA"):
        main(["--output", str(tmp_path / "missing.json")])
    assert not (tmp_path / "missing.json").exists()


def test_operator_benchmark_cpu_trace_records_actual_sdpa_dispatch(tmp_path):
    report = main(["--device", "cpu", "--output", str(tmp_path / "report.json"), "--trace-dir", str(tmp_path / "trace"),
                   "--cases", "attention", "--tokens", "8", "--channels", "12", "--heads", "3", "--query-chunks", "4",
                   "--warmup", "0", "--repeats", "1"])
    for case in report["cases"]:
        assert case["trace"]["sdpa_operators"]
        assert case["trace"]["operators"]
    assert len(list((tmp_path / "trace").glob("*.json"))) == 2


def test_chunk_tuning_checks_cross_ffn_block_and_frozen_sparse_geometry(tmp_path):
    checkpoint = tmp_path / "geometry.pt"
    coordinates = torch.nn.functional.pad(torch.cartesian_prod(torch.arange(2), torch.arange(2), torch.arange(2)), (1, 0))
    torch.save({"value": {"coordinates": coordinates}}, checkpoint)
    report = main(["--device", "cpu", "--output", str(tmp_path / "chunks.json"),
                   "--cases", "cross_attention", "ffn", "flow_block", "sparse_chunks",
                   "--tokens", "8", "--channels", "24", "--heads", "3", "--projection-channels", "16",
                   "--context-channels", "8", "--ffn-hidden", "32", "--ffn-rows", "2", "4",
                   "--query-chunks", "4", "8", "--cross-query-chunks", "4", "8", "0",
                   "--sparse-coordinate-checkpoint", str(checkpoint), "--sparse-channels", "4",
                   "--sparse-chunk-rows", "4", "8", "--warmup", "0", "--repeats", "1"])
    assert report["status"] == "complete" and len(report["cases"]) == 12
    assert all(case["status"] == "complete" and "max_abs_error" in case for case in report["cases"])
    assert report["sparse_geometry_source"]["sha256"]
    assert report["cases"][2]["parameters"]["cross_attention_queries"] == 0
    profiles = [case["parameters"] for case in report["cases"] if case["name"].startswith("block-crossfull")]
    assert {(profile["attention_queries"], profile["ffn_rows"]) for profile in profiles} == {(4, 2), (4, 4), (8, 2), (8, 4)}


def test_pretrained_forward_sweep_interleaves_limits_on_one_model(tmp_path):
    pytest.importorskip("modal")
    from deploy.modal_tuning import sweep_pretrained_forwards
    from pixalcuda.flow import Flow
    model = Flow(4, 4, 8, channels=12, heads=2, blocks=2, hidden=24, context_channels=8)
    coordinates = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1]])
    noise = torch.randn(2, 4)
    context = {"global": torch.randn(1, 5, 8), "proj": torch.randn(1, 2, 8)}
    with torch.inference_mode():
        report = sweep_pretrained_forwards(model, noise, context, coordinates, tmp_path / "forwards.json")
    assert report["status"] == "complete" and len(report["cases"]) == 4
    assert all(len(case["samples"]) == 3 and case["warm_relative_l2_error"] < 1e-5 for case in report["cases"])
    assert model.blocks[0].self_attn.limits.attention_queries == 1024
    assert model.blocks[0].cross_attn.cross_attn_block.limits.cross_attention_queries == 0
    assert model.blocks[0].mlp.rows == 8192
