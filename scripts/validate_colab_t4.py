"""Run the real pretrained 512/PBR acceptance case on an allocated Colab T4."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import time
import urllib.request

import psutil
import torch

from pixalcuda.export import validate_glb
from pixalcuda.pipeline import Pipeline
from pixalcuda.runtime import atomic_json
from pixalcuda.weights import sha256_file


SAMPLE_URL = "https://raw.githubusercontent.com/syi0808/pixal3d-mlx/788604820daa467b56f1affa40d6f443a1cc2cc1/assets/shoe_input.png"
SAMPLE_SHA256 = "abf395cc52d81c26dadae9f024072d6c7301679be4e8fc08d572723d7ae32a21"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/colab-t4-shoe")
    parser.add_argument("--image", help="Optional local image; otherwise download the pinned public shoe sample")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("This acceptance run requires a real NVIDIA T4 CUDA runtime")
    name = torch.cuda.get_device_name()
    if "T4" not in name or torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError(f"Expected an actual T4 (SM 7.5), received {name}")
    directory = Path(arguments.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    image = Path(arguments.image) if arguments.image else directory / "source.png"
    if not arguments.image and not image.exists():
        with urllib.request.urlopen(SAMPLE_URL, timeout=120) as response:
            image.write_bytes(response.read())
    if not arguments.image and sha256_file(image) != SAMPLE_SHA256:
        raise ValueError("Pinned public sample SHA-256 mismatch")
    evidence = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "gpu": name,
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "system_ram_bytes": psutil.virtual_memory().total,
        "input_sha256": sha256_file(image), "sample_url": None if arguments.image else SAMPLE_URL,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "seed": 42, "geometry_resolution": 512, "texture_size": 1024,
        "camera": "auto", "status": "running",
    }
    atomic_json(directory / "acceptance.json", evidence)
    print(json.dumps(evidence, indent=2), flush=True)
    started = time.perf_counter()
    try:
        output = Pipeline("cuda").generate(image, directory / "run", resume=arguments.resume)
        evidence.update(status="complete", validation=validate_glb(output), output_sha256=sha256_file(output))
        print(f"T4_ACCEPTANCE_COMPLETE: {output}", flush=True)
    except BaseException as error:
        evidence.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        evidence["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(directory / "acceptance.json", evidence)
        print(json.dumps(evidence, indent=2), flush=True)


if __name__ == "__main__":
    main()
