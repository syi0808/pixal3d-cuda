"""Local, validated infrastructure declarations; loading never calls Modal."""

from dataclasses import dataclass, fields
import math
import os
from pathlib import Path
import re
import tomllib


@dataclass(frozen=True)
class DeploySettings:
    environment: str = "main"
    app_name: str = "pixal3d-cuda"
    weights_volume: str = "pixal3d-weights"
    jobs_volume: str = "pixal3d-jobs"
    jobs_dict: str = "pixal3d-jobs"
    hf_secret: str = "pixal3d-hf"
    gpu: str = "L4"
    min_containers: int = 0
    max_gpu_containers: int = 1
    scaledown_window: int = 2
    cpu_cores: float = 2.0
    cpu_memory_mib: tuple[int, int] = (4096, 16384)
    gpu_memory_mib: tuple[int, int] = (16384, 65536)
    prefetch_memory_mib: tuple[int, int] = (2048, 8192)
    gpu_timeout_seconds: int = 1800
    cpu_timeout_seconds: int = 1800
    prefetch_timeout_seconds: int = 3600
    max_views: int = 16
    max_upload_bytes: int = 16 * 1024 ** 2
    max_total_image_bytes: int = 64 * 1024 ** 2
    max_transforms_bytes: int = 64 * 1024
    max_image_pixels: int = 16_000_000
    api_memory_mib: tuple[int, int] = (1024, 4096)
    api_max_jobs_per_day: int = 10
    low_vram: bool = True
    attention_backend: str = "sdpa"
    max_tokens: int = 49152

    def __post_init__(self):
        if self.environment != "main":
            raise ValueError("This deployment uses only the main Modal environment")
        for name in ("app_name", "weights_volume", "jobs_volume", "jobs_dict", "hf_secret"):
            value = getattr(self, name)
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", value):
                raise ValueError(f"{name} must be a nonempty resource name")
        if self.gpu != "L4":
            raise ValueError("The official BF16 baseline image targets L4/SM89; T4 requires a separately validated port. No GPU fallback.")
        if type(self.min_containers) is not int or self.min_containers != 0:
            raise ValueError("min_containers must stay zero for the low-credit deployment")
        if (isinstance(self.cpu_cores, bool) or not isinstance(self.cpu_cores, (int, float))
                or not math.isfinite(self.cpu_cores) or self.cpu_cores <= 0):
            raise ValueError("cpu_cores must be finite and positive")
        for name in ("cpu_memory_mib", "gpu_memory_mib", "prefetch_memory_mib", "api_memory_mib"):
            value = getattr(self, name)
            if (not isinstance(value, (tuple, list)) or len(value) != 2
                    or any(type(item) is not int or item <= 0 for item in value) or value[0] > value[1]):
                raise ValueError(f"{name} must be [positive request MiB, limit MiB >= request]")
            object.__setattr__(self, name, tuple(value))
        if type(self.low_vram) is not bool or self.attention_backend not in ("sdpa", "flash_attn"):
            raise ValueError("Invalid upstream low_vram/attention_backend setting")
        if self.attention_backend == "flash_attn":
            raise ValueError("The baseline image builds SDPA only; install and validate FlashAttention before enabling it")
        excluded = {"min_containers", "cpu_cores"}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name not in excluded and field.type is int and (type(value) is not int or value <= 0):
                raise ValueError(f"{field.name} must be a positive integer")
        if not 2 <= self.scaledown_window <= 3600:
            raise ValueError("scaledown_window must be between 2 and 3600 seconds")
        if self.gpu_timeout_seconds <= 30:
            raise ValueError("gpu_timeout_seconds must exceed the worker cleanup margin of 30 seconds")
        if self.max_upload_bytes > self.max_total_image_bytes:
            raise ValueError("Per-image upload limit must not exceed the aggregate limit")


def load_settings(config_path=None, environ=None):
    config_path = Path(config_path) if config_path is not None else Path(__file__).with_name("modal.toml")
    values = tomllib.loads(config_path.read_text())
    environment = os.environ if environ is None else environ
    if environment.get("MODAL_ENVIRONMENT", "main") != "main":
        raise ValueError("Use MODAL_ENVIRONMENT=main; no separate dev/prod deployment is configured")
    if "PIXAL_GPU" in environment:
        values["gpu"] = environment["PIXAL_GPU"]
    try:
        return DeploySettings(**values)
    except TypeError as error:
        raise ValueError(f"Unknown or invalid deployment setting: {error}") from error
