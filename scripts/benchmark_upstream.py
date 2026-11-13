"""Plan first; explicitly execute alternating official low-VRAM/standard baselines."""

import argparse
import json
from pathlib import Path
import statistics
import time

from pixalcuda.pipeline import Pipeline
from pixalcuda.runtime import atomic_json
from pixalcuda.upstream_sources import lock
from pixalcuda.optimizations import OptimizationOptions
from dataclasses import asdict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="outputs/upstream-baselines")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--geometry-resolution", type=int, choices=(1024, 1536), default=1024)
    parser.add_argument("--texture-size", type=int, choices=(1024, 4096), default=4096)
    parser.add_argument("--profiles", nargs="+", choices=("reference", "reuse", "projected", "optimized", "standard"), default=["reference", "optimized"])
    parser.add_argument("--conditioning-cache-mb", type=int, default=256)
    parser.add_argument("--save-reference", action="store_true")
    parser.add_argument("--remove-background", action="store_true", help="Opt in to RMBG-2.0 after reviewing its separate license")
    parser.add_argument("--execute", action="store_true", help="Actually run local CUDA generation; no Modal calls")
    arguments = parser.parse_args(argv)
    if not 1 <= arguments.repeats <= 10:
        parser.error("repeats must be between 1 and 10")
    configurations = {"reference": OptimizationOptions.disabled(), "reuse": OptimizationOptions(True, 128, False, 0),
                      "projected": OptimizationOptions(True, 128, True, 0),
                      "optimized": OptimizationOptions(conditioning_cache_mb=arguments.conditioning_cache_mb)}
    configurations["standard"] = configurations["optimized"]
    if len(set(arguments.profiles)) != len(arguments.profiles):
        parser.error("profiles must not repeat")
    plan = {"schema": "pixalcuda.upstream-benchmark.v1", "execute": arguments.execute,
            "upstream_lock": lock(), "image": str(Path(arguments.image).expanduser().resolve()),
            "profiles": arguments.profiles, "repeats": arguments.repeats,
            "optimizations": {name: asdict(configurations[name]) for name in arguments.profiles},
            "geometry_resolution": arguments.geometry_resolution, "texture_size": arguments.texture_size,
            "seed": 42, "remove_background": arguments.remove_background,
            "process_policy": "Fresh worker per asset; kernel/autotune cache may persist. Not a warm-forward benchmark."}
    print(json.dumps(plan, indent=2))
    if not arguments.execute:
        return plan
    directory = Path(arguments.output_dir)
    if directory.exists():
        raise ValueError("Use a fresh output directory for a controlled benchmark")
    directory.mkdir(parents=True)
    atomic_json(directory / "plan.json", plan)
    results = []
    for round_index in range(arguments.repeats):
        profiles = plan["profiles"] if round_index % 2 == 0 else list(reversed(plan["profiles"]))
        for profile in profiles:
            destination = directory / f"round-{round_index}-{profile}"
            started = time.perf_counter()
            try:
                Pipeline(low_vram=profile != "standard", optimizations=configurations[profile], save_reference=arguments.save_reference).generate(arguments.image, destination, seed=42,
                    geometry_resolution=arguments.geometry_resolution, texture_size=arguments.texture_size,
                    remove_background=arguments.remove_background)
                report = json.loads((destination / "report.json").read_text())
                record = {"profile": profile, "round": round_index, "status": "complete",
                          "wall_seconds": time.perf_counter() - started, "report": str(destination / "report.json"),
                          "peak_allocated_bytes": max(stage["peak_allocated_bytes"] for stage in report["stages"]),
                          "glb_sha256": report["validation"]["sha256"], "optimizations": report["optimizations"]}
            except Exception as error:
                record = {"profile": profile, "round": round_index, "status": "failed",
                          "error": f"{type(error).__name__}: {error}"}
            results.append(record)
            atomic_json(directory / "results.json", results)
            if record["status"] == "failed":
                raise RuntimeError("Benchmark stopped on failure; no fallback or failed-case exclusion")
    summary = {profile: {"median_asset_wall_seconds": statistics.median(record["wall_seconds"] for record in results if record["profile"] == profile)}
               for profile in plan["profiles"]}
    atomic_json(directory / "summary.json", summary)
    return summary


if __name__ == "__main__":
    main()
