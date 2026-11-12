from dataclasses import asdict, replace

import pytest

from deploy.settings import DeploySettings, load_settings


def test_checked_in_declarations_have_low_credit_defaults():
    settings = load_settings(environ={})
    assert settings == DeploySettings()
    assert settings.gpu == "T4" and settings.min_containers == 0 and settings.max_gpu_containers == 1
    assert settings.max_views == 16
    assert settings.gpu_memory_mib == (8192, 16384)
    assert load_settings(environ={"PIXAL_GPU": "L4"}).gpu == "L4"


@pytest.mark.parametrize("changed", [
    {"gpu": "H100"}, {"min_containers": 1}, {"min_containers": False}, {"max_gpu_containers": 0},
    {"max_views": True}, {"max_views": 0}, {"cpu_cores": float("nan")}, {"cpu_cores": False},
    {"gpu_memory_mib": (16384, 8192)}, {"gpu_memory_mib": (0, 8192)}, {"scaledown_window": 1},
    {"app_name": ""}, {"max_upload_bytes": 65 * 1024 ** 2},
])
def test_invalid_or_unexpectedly_warm_declarations_are_rejected(changed):
    with pytest.raises(ValueError):
        replace(DeploySettings(), **changed)


def test_unknown_config_fields_fail_instead_of_being_ignored(tmp_path):
    config = tmp_path / "modal.toml"
    config.write_text('gpu = "T4"\nmin_container = 10\n')
    with pytest.raises(ValueError, match="Unknown"):
        load_settings(config, environ={})
    assert "HF_TOKEN" not in asdict(DeploySettings())
