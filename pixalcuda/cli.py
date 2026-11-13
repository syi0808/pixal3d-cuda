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
    download.add_argument("--multiview", action="store_true")
    postprocess = commands.add_parser("export", help="Finish GPU checkpoints on a CPU-only host")
    postprocess.add_argument("--output-dir", required=True)
    generation = commands.add_parser("generate")
    inputs = generation.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image")
    inputs.add_argument("--views-dir", help="Posed images and transforms.json; no recropping or MoGe")
    inputs.add_argument("--images", nargs="+", help="Ordered image paths, each mapped to a camera in --transforms")
    generation.add_argument("--transforms", help="Camera JSON for --images; indexed frames or input-order frames")
    generation.add_argument("--num-views", type=int, help="Use a positive prefix of the mapped images")
    generation.add_argument("--output-dir", default="outputs/run")
    generation.add_argument("--seed", type=int, default=42)
    generation.add_argument("--geometry-resolution", type=int, choices=(512, 1024), default=512)
    generation.add_argument("--texture-size", type=int, choices=(512, 1024, 2048), default=1024)
    generation.add_argument("--target-faces", type=int, default=100000)
    generation.add_argument("--camera", choices=("auto", "fixed"))
    generation.add_argument("--fov-degrees", type=float)
    generation.add_argument("--no-rembg", action="store_true")
    generation.add_argument("--resume", action="store_true")
    generation.add_argument("--defer-export", action="store_true", help="Save neural outputs; finish with the CPU export command")
    generation.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    generation.add_argument("--model-dir", help="Local pinned snapshot containing pipeline.json and ckpts/")
    for field in fields(Limits):
        generation.add_argument("--" + field.name.replace("_", "-"), type=int, default=field.default)
    arguments = parser.parse_args(argv)
    if arguments.command == "doctor":
        doctor()
    elif arguments.command == "download":
        ModelStore(multiview=arguments.multiview).download(camera=not arguments.skip_camera)
    elif arguments.command == "export":
        from .pipeline import finish_export
        print(f"Saved PBR GLB: {finish_export(arguments.output_dir)}")
    else:
        if arguments.target_faces < 4:
            parser.error("--target-faces must be at least 4")
        if bool(arguments.images) != bool(arguments.transforms):
            parser.error("--images and --transforms must be supplied together")
        multiview = bool(arguments.views_dir or arguments.images)
        if arguments.num_views is not None and (not multiview or arguments.num_views < 1):
            parser.error("--num-views requires posed inputs and must be positive")
        if multiview and (arguments.camera is not None or arguments.fov_degrees is not None):
            parser.error("Mapped cameras supply the FOV; camera/FOV overrides are not allowed")
        camera_metadata = None
        if arguments.transforms:
            try:
                with open(arguments.transforms) as source:
                    camera_metadata = json.load(source)
            except (OSError, ValueError) as error:
                parser.error(f"Invalid camera JSON: {error}")
        from .pipeline import Pipeline
        limits = Limits(**{field.name: getattr(arguments, field.name) for field in fields(Limits)})
        output = Pipeline(arguments.device, limits, ModelStore(arguments.model_dir, multiview=multiview)).generate(
            arguments.image, arguments.output_dir, arguments.seed, arguments.texture_size, arguments.target_faces,
            arguments.camera or "auto", arguments.fov_degrees if arguments.fov_degrees is not None else 49.13,
            not arguments.no_rembg, arguments.resume, views_dir=arguments.views_dir, num_views=arguments.num_views,
            defer_export=arguments.defer_export,
            images=arguments.images, camera_metadata=camera_metadata,
            geometry_resolution=arguments.geometry_resolution,
        )
        print(f"Saved {'GPU checkpoints' if arguments.defer_export else 'PBR GLB'}: {output}")


if __name__ == "__main__":
    main()
