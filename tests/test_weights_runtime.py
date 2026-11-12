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


def test_loader_rejects_overflowing_fp16_conversion(tmp_path):
    model = torch.nn.Linear(3, 2)
    with torch.no_grad():
        model.weight.fill_(70000)
    checkpoint = tmp_path / "large.safetensors"
    save_file(model.state_dict(), checkpoint)
    with pytest.raises(FloatingPointError, match="wider dtype"):
        load_model(lambda: torch.nn.Linear(3, 2), checkpoint, "cpu", torch.float16)
    loaded = load_model(lambda: torch.nn.Linear(3, 2), checkpoint, "cpu", torch.float32)
    assert torch.isfinite(loaded.weight).all()


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


def test_cpu_export_restore_rejects_tampered_manifest_or_source(tmp_path):
    runtime = Runtime(tmp_path, "cpu", {"seed": 42})
    restored = Runtime.restore(tmp_path)
    assert restored.device.type == "cpu" and restored.fingerprint == runtime.fingerprint
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["identity"]["seed"] = 43
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity mismatch"):
        Runtime.restore(tmp_path)
    changed = tmp_path / "changed"
    Runtime(changed, "cpu", {"source_files": {"not_current.py": "hash"}})
    with pytest.raises(ValueError, match="source changed"):
        Runtime.restore(changed)


def test_offline_naf_cache_miss_cannot_download(tmp_path, monkeypatch):
    from huggingface_hub import constants
    from pixalcuda.weights import ModelStore

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    with pytest.raises(FileNotFoundError, match="prefetch weights on CPU"):
        ModelStore().naf()
