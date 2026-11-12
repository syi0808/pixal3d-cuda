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
