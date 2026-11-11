import json

import pytest
from safetensors.torch import save_file
import torch

from pixalcuda.runtime import Runtime
from pixalcuda.weights import load_model


def test_strict_streaming_loader_materializes_meta_and_casts(tmp_path):
    expected = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "weights.safetensors"
    save_file(expected.state_dict(), checkpoint)
    actual = load_model(lambda: torch.nn.Linear(3, 2), checkpoint, "cpu", torch.float32)
    torch.testing.assert_close(actual.weight, expected.weight)
    assert not any(parameter.is_meta or parameter.requires_grad for parameter in actual.parameters())
    with pytest.raises(ValueError, match="shape"):
        load_model(lambda: torch.nn.Linear(4, 2), checkpoint, "cpu", torch.float32)
    with pytest.raises(ValueError, match="mismatch"):
        load_model(lambda: torch.nn.Sequential(torch.nn.Linear(3, 2)), checkpoint, "cpu", torch.float32)


def test_resume_rejects_changed_input_and_settings(tmp_path):
    runtime = Runtime(tmp_path, "cpu", {"seed": 42})
    runtime.save("shape", {"value": torch.ones(3)})
    resumed = Runtime(tmp_path, "cpu", {"seed": 42}, resume=True)
    torch.testing.assert_close(resumed.load("shape")["value"], torch.ones(3))
    with pytest.raises(ValueError, match="changed"):
        Runtime(tmp_path, "cpu", {"seed": 43}, resume=True)


def test_failure_writes_report_and_keeps_checkpoint(tmp_path):
    runtime = Runtime(tmp_path, "cpu", {"seed": 42})
    runtime.save("structure", torch.zeros(1))
    with pytest.raises(MemoryError):
        with runtime.stage("failing"):
            raise MemoryError("test")
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["stages"][-1]["status"] == "failed"
    assert (tmp_path / "structure.pt").exists()
