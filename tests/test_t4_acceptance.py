import importlib.util
from pathlib import Path

import pytest
import torch


def test_acceptance_runner_refuses_cpu_before_download(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_colab_t4.py"
    specification = importlib.util.spec_from_file_location("t4_acceptance", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="real NVIDIA T4"):
        module.main(["--output-dir", str(tmp_path / "run")])
    assert not (tmp_path / "run").exists()
