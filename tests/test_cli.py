from pathlib import Path
from types import SimpleNamespace

import pytest

from pixalcuda.cli import main


@pytest.mark.parametrize("budget", [512, 1024])
def test_cli_forwards_geometry_and_cache_without_loading_models(tmp_path, monkeypatch, budget):
    calls = {}
    def pipeline(device, limits, store):
        calls.update(device=device, limits=limits)
        def generate(*arguments, **options):
            calls["options"] = options
            return Path(tmp_path / "result.glb")
        return SimpleNamespace(generate=generate)
    monkeypatch.setattr("pixalcuda.pipeline.Pipeline", pipeline)
    main(["generate", "--image", str(tmp_path / "input.png"), "--device", "cpu",
          "--geometry-resolution", "1024", "--conditioning-cache-mb", str(budget),
          "--max-tokens", "32768", "--max-voxels", "6000000"])
    assert calls["options"]["geometry_resolution"] == 1024
    assert calls["limits"].conditioning_cache_mb == budget
    assert calls["limits"].max_tokens == 32768 and calls["limits"].max_voxels == 6000000
