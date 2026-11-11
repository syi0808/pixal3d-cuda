"""Command-line generation and environment diagnostics."""

import argparse
from dataclasses import fields
import json

import psutil
import torch

from .ops import Limits
from .weights import ModelStore


def doctor():
    report = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
              "cuda_runtime": torch.version.cuda, "system_ram_gib": psutil.virtual_memory().total / 1024 ** 3,
              "available_ram_gib": psutil.virtual_memory().available / 1024 ** 3}
    if torch.cuda.is_available():
        from .ops import attention
        report.update(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                      vram_gib=torch.cuda.get_device_properties(0).total_memory / 1024 ** 3)
        with torch.inference_mode():
            for dtype in (torch.float16, torch.float32):
                tensor = torch.randn(1, 12, 513, 128, device="cuda", dtype=dtype)
                output = attention(tensor, tensor, tensor, 128)
                report[f"sdpa_{dtype}"] = bool(output.isfinite().all())
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    download = commands.add_parser("download")
    download.add_argument("--skip-camera", action="store_true")
    generation = commands.add_parser("generate")
    generation.add_argument("--image", required=True)
    generation.add_argument("--output-dir", default="outputs/run")
    generation.add_argument("--seed", type=int, default=42)
    generation.add_argument("--texture-size", type=int, choices=(512, 1024, 2048), default=1024)
    generation.add_argument("--target-faces", type=int, default=100000)
    generation.add_argument("--camera", choices=("auto", "fixed"), default="auto")
    generation.add_argument("--fov-degrees", type=float, default=49.13)
    generation.add_argument("--no-rembg", action="store_true")
    generation.add_argument("--resume", action="store_true")
    generation.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    generation.add_argument("--model-dir", help="Local pinned snapshot containing pipeline.json and ckpts/")
    for field in fields(Limits):
        generation.add_argument("--" + field.name.replace("_", "-"), type=int, default=field.default)
    arguments = parser.parse_args(argv)
    if arguments.command == "doctor":
        doctor()
    elif arguments.command == "download":
        ModelStore().download(camera=not arguments.skip_camera)
    else:
        if arguments.target_faces < 4:
            parser.error("--target-faces must be at least 4")
        from .pipeline import Pipeline
        limits = Limits(**{field.name: getattr(arguments, field.name) for field in fields(Limits)})
        output = Pipeline(arguments.device, limits, ModelStore(arguments.model_dir)).generate(
            arguments.image, arguments.output_dir, arguments.seed, arguments.texture_size, arguments.target_faces,
            arguments.camera, arguments.fov_degrees, not arguments.no_rembg, arguments.resume,
        )
        print(f"Saved PBR GLB: {output}")


if __name__ == "__main__":
    main()
