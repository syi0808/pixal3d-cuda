"""Explicit single-L4 private prefix/full profiling; never submits API jobs."""

import argparse
import json
from pathlib import Path
import sys
import uuid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-job", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("prefix", "full"), default="prefix")
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args(argv)
    if len(arguments.source_job) != 64 or any(character not in "0123456789abcdef" for character in arguments.source_job):
        parser.error("Use a completed native job's 64-character ID")
    plan = {"execute": arguments.execute, "environment": "main", "gpu": "L4", "max_gpu_containers": 1,
            "gpu_timeout_seconds": 1200, "automatic_retries": 0, "source_job": arguments.source_job,
            "scope": "SS and LR shape through quantized HR coordinates; no HR/texture flow, full decode or GLB export",
            "profiles": ["reference_first", "reference_repeat", "dino", "projection", "conditioning", "optimized", "reference_splitk2"],
            "splitk": "1, except reference_splitk2 uses 2 only for (2^13, 512, 512, 27)",
            "model_policy": "One reference pipeline loaded once; shared backbone aliases restored between probes",
            "private_tensors": True}
    plan["scope_name"] = arguments.scope
    if arguments.scope == "full":
        plan.update(gpu_timeout_seconds=1800, profiles=["reference_first", "reference_repeat", "optimized"],
                    scope="Full native neural parity, frozen-mesh 1024/4K export twice, reference/optimized model-load components",
                    splitk="1; inner autotuning calibrated once then new keys rejected", frozen_export_repeats=2,
                    model_policy="Reference pipeline reused for parity; a second optimized initialization is measured separately")
    print(json.dumps(plan, indent=2), flush=True)
    if not arguments.execute:
        return plan
    if arguments.output_dir.exists():
        raise ValueError("Use a fresh directory; never resubmit an ambiguous profiling call")

    import modal
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from deploy.modal_app import jobs, offline, weights
    from pixalcuda.runtime import atomic_json

    stats = modal.Function.from_name("pixal3d-cuda", "infer_job", environment_name="main").get_current_stats()
    if stats.backlog or stats.num_total_runners or stats.num_running_inputs:
        raise RuntimeError("Production GPU work is present; profiling will not request another GPU")
    app = modal.App("pixal3d-native-profile")

    @app.function(image=offline, gpu="L4", cpu=(2, 2), memory=(16384, 65536), timeout=plan["gpu_timeout_seconds"], retries=0,
                  min_containers=0, max_containers=1, buffer_containers=0, scaledown_window=2,
                  include_source=False, serialized=True,
                  volumes={"/weights": weights.with_mount_options(read_only=True), "/jobs": jobs})
    def profile(source_job, attempt, scope):
        import hashlib
        import json
        import os
        from pathlib import Path
        import sys

        if scope == "full":
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        import numpy as np
        from PIL import Image

        from pixalcuda.background import deferred_background_removal
        from pixalcuda.profiling import run_prefix_profiles
        from pixalcuda.runtime import atomic_json
        from pixalcuda.upstream_sources import lock, verify_source
        from pixalcuda.upstream_weights import UpstreamStore, local_pipeline_config
        from pixalcuda.upstream_worker import NativeReport, load_entry, pin_naf_loader, require_hardware

        jobs.reload()
        weights.reload()
        source = Path("/jobs") / source_job
        directory = Path("/jobs/profiling") / attempt
        directory.mkdir(parents=True, exist_ok=False)
        previous = json.loads((source / "run/report.json").read_text())
        image_path = source / "prepared/input.png"
        image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if (previous["status"] != "complete" or previous["identity"]["upstream_lock"] != lock()
                or previous["identity"]["multiview"] or previous["identity"]["remove_background"]
                or previous["identity"]["inputs"][0]["sha256"] != image_hash):
            raise ValueError("Profiling requires a completed, unmatted, matching native single-image job")
        identity = {"upstream_lock": lock(), "image_sha256": image_hash, "camera": previous["camera"],
                    "seed": 42, "geometry_resolution": 1024, "remove_background": False,
                    "model_policy": "Same loaded pipeline and warmed in-process Triton configurations; fixed top-level SPLITK",
                    "runtime_sources": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                        for path in sorted(Path("/opt/pixal3d/pixalcuda").glob("*.py"))}}
        report = NativeReport(directory, identity)
        try:
            report.data.update(require_hardware())
            for name in lock()["sources"]:
                verify_source(name, "/opt/upstream")
            sys.path.insert(0, "/opt/upstream/pixal3d")
            os.environ["FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE"] = "0"
            entry = load_entry("/opt/upstream", False)
            bundle = UpstreamStore().resolve(camera=False, rembg=False)
            pin_naf_loader("/opt/upstream", bundle["naf_weights"])
            for config in entry.IMAGE_COND_CONFIGS.values():
                config["model_name"] = bundle["dino"]
            with Image.open(image_path) as image:
                rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
                prepared = Image.fromarray((rgba[..., :3] * (rgba[..., 3:] / 255)).astype(np.uint8))
            if scope == "full":
                from pixalcuda.full_profiling import execute_full_profile

                execute_full_profile(entry, bundle, prepared, previous["camera"], directory, report)
                report.data["scope"] = "Full native tensor parity and frozen export/load profiling; not an original CLI or production throughput benchmark"
            else:
                with report.stage("load_pipeline"):
                    local = local_pipeline_config(bundle, directory / "pipeline-config", rembg=False)
                    with deferred_background_removal():
                        pipeline = entry.init_pipeline(model_path=str(local), low_vram=True)
                report.data["profiles"] = run_prefix_profiles(pipeline, prepared, previous["camera"], directory, report)
                report.data["scope"] = "Private SS/LR prefix ablations only; not GLB parity or production performance"
            report.data["status"] = "complete"
        except BaseException as error:
            report.data.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            report.write()
            jobs.commit()
        return {"report": report.data, "private_volume_directory": str(directory)}

    arguments.output_dir.mkdir(parents=True)
    atomic_json(arguments.output_dir / "plan.json", plan)
    attempt = arguments.scope + "-" + uuid.uuid4().hex
    with modal.enable_output(), app.run(environment_name="main"):
        call = profile.spawn(arguments.source_job, attempt, arguments.scope)
        atomic_json(arguments.output_dir / "call.json", {"call_id": call.object_id, "attempt": attempt})
        print(json.dumps({"call_id": call.object_id, "attempt": attempt}), flush=True)
        try:
            result = call.get(timeout=7200)
        except BaseException:
            call.cancel(terminate_containers=True)
            raise
        atomic_json(arguments.output_dir / "result.json", result)
        print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
