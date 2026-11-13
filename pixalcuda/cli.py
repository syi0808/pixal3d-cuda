"""Official Pixal3D CUDA entry points."""

import argparse
import importlib.util
import json
import sys

import torch

from .upstream_sources import lock


def doctor():
    available = torch.cuda.is_available()
    report = {"runtime": "official", "upstream_revision": lock()["sources"]["pixal3d"]["revision"],
              "torch": torch.__version__, "cuda_available": available,
              "native_extensions": {name: importlib.util.find_spec(name) is not None for name in
                                    ("flex_gemm", "cumesh", "o_voxel", "natten", "nvdiffrast")}}
    if available:
        report.update(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                      native_bf16_baseline_supported=torch.cuda.get_device_capability()[0] >= 8)
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "setup-upstream":
        from .upstream_sources import main as setup_main
        return setup_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup-upstream", help="Fetch/build the pinned official native sources")
    commands.add_parser("doctor")
    download = commands.add_parser("download")
    download.add_argument("--multiview", action="store_true")
    download.add_argument("--skip-camera", action="store_true")
    generation = commands.add_parser("generate")
    inputs = generation.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image")
    inputs.add_argument("--views-dir")
    inputs.add_argument("--images", nargs="+")
    generation.add_argument("--transforms")
    generation.add_argument("--num-views", type=int)
    generation.add_argument("--output-dir", default="outputs/upstream-run")
    generation.add_argument("--upstream-root")
    generation.add_argument("--model-dir")
    generation.add_argument("--seed", type=int, default=42)
    generation.add_argument("--geometry-resolution", type=int, choices=(1024, 1536), default=1024)
    generation.add_argument("--texture-size", type=int, choices=(512, 1024, 2048, 4096), default=4096)
    generation.add_argument("--target-faces", type=int, default=1000000)
    generation.add_argument("--camera", choices=("auto", "fixed"))
    generation.add_argument("--fov-degrees", type=float)
    generation.add_argument("--no-rembg", action="store_true")
    generation.add_argument("--resume", action="store_true")
    generation.add_argument("--standard", action="store_true", help="Keep original models on GPU instead of upstream low-VRAM mode")
    generation.add_argument("--attention-backend", choices=("sdpa", "flash_attn"), default="sdpa")
    generation.add_argument("--max-tokens", type=int, default=49152)
    generation.add_argument("--timeout-seconds", type=int, default=1800)
    generation.add_argument("--save-reference", action="store_true", help="Save private native latents and pre-remesh mesh/voxel tensors")
    arguments = parser.parse_args(argv)
    if arguments.command == "doctor":
        return doctor()
    from .upstream_weights import UpstreamStore
    if arguments.command == "download":
        print(json.dumps(UpstreamStore(multiview=arguments.multiview).resolve(camera=not arguments.skip_camera), indent=2))
        return
    if bool(arguments.images) != bool(arguments.transforms):
        parser.error("--images and --transforms must be supplied together")
    multiview = bool(arguments.images or arguments.views_dir)
    if arguments.num_views is not None and (not multiview or arguments.num_views < 1):
        parser.error("--num-views requires mapped inputs and must be positive")
    if multiview and (arguments.camera is not None or arguments.fov_degrees is not None):
        parser.error("Mapped cameras supply their own FOV")
    metadata = None
    if arguments.transforms:
        try:
            with open(arguments.transforms) as source:
                metadata = json.load(source)
        except (OSError, ValueError) as error:
            parser.error(f"Invalid camera JSON: {error}")
    from .pipeline import Pipeline
    result = Pipeline(upstream_root=arguments.upstream_root,
                      store=UpstreamStore(arguments.model_dir, multiview=multiview), low_vram=not arguments.standard,
                      attention_backend=arguments.attention_backend, max_tokens=arguments.max_tokens,
                      timeout_seconds=arguments.timeout_seconds, save_reference=arguments.save_reference).generate(
        arguments.image, arguments.output_dir, seed=arguments.seed, texture_size=arguments.texture_size,
        target_faces=arguments.target_faces, camera=arguments.camera or "auto",
        fov_degrees=arguments.fov_degrees if arguments.fov_degrees is not None else 49.13,
        remove_background=not arguments.no_rembg, resume=arguments.resume, geometry_resolution=arguments.geometry_resolution,
        views_dir=arguments.views_dir, images=arguments.images, camera_metadata=metadata, num_views=arguments.num_views)
    print(f"Saved official PBR GLB: {result}")


if __name__ == "__main__":
    main()
