"""Explicit one-shot T4 operator tuning in main; no deployed endpoint or model downloads."""

import json
from dataclasses import asdict, replace
from pathlib import Path
import re
import statistics
import subprocess

import modal

from deploy.modal_app import ROOT, SETTINGS, jobs, offline, weights


app = modal.App("pixal3d-cuda-tuning")
image = (offline
         .add_local_file(ROOT / "scripts" / "benchmark_cuda_ops.py", "/opt/pixal3d/scripts/benchmark_cuda_ops.py", copy=True)
         .add_local_file(ROOT / "deploy" / "modal_tuning.py", "/opt/pixal3d/deploy/modal_tuning.py", copy=True))


@app.function(image=image, gpu="T4", cpu=(2, 2), memory=(4096, 8192), timeout=600, retries=0,
              min_containers=0, max_containers=1, buffer_containers=0, scaledown_window=2,
              include_source=False, volumes={"/jobs": jobs, "/weights": weights.with_mount_options(read_only=True)})
def benchmark(run_id, source_job, math_attention_only=False, pretrained_forwards=False):
    import torch

    from pixalcuda.runtime import atomic_json
    from scripts.benchmark_cuda_ops import main

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id) or not re.fullmatch(r"[0-9a-f]{64}", source_job):
        raise ValueError("Expected a safe run name and a 64-digit source job ID")
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name() or torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError("This tuning run requires an actual NVIDIA T4")
    jobs.reload()
    source = Path("/jobs") / source_job / "run"
    source_report = json.loads((source / "report.json").read_text())
    tokens = source_report["active_counts"]["refinement_tokens"]
    if source_report["status"] != "complete" or not 1 <= tokens <= 32768:
        raise ValueError("Use a completed bounded source run")
    destination = Path("/jobs/benchmarks") / run_id
    destination.mkdir(parents=True, exist_ok=False)
    cases = ["attention"] if math_attention_only else ["cross_attention", "attention", "ffn", "flow_block", "sparse_chunks"]
    backends = ["auto", "math"] if math_attention_only else ["auto"]
    sparse_source = [] if math_attention_only else ["--sparse-coordinate-checkpoint", str(source / "texture_decoded.pt")]
    try:
        if pretrained_forwards:
            from pixalcuda.flow import Flow
            from pixalcuda.ops import Limits
            from pixalcuda.weights import ModelStore, load_model

            if math_attention_only or source_report["identity"]["conditioning_mode"] != "single":
                raise ValueError("Pretrained forward study requires a single-image run and no math-only flag")
            store = ModelStore()
            if store.revision != source_report["identity"]["pixal_revision"]:
                raise ValueError("Frozen source and cached model revision differ")
            limits = Limits(conditioning_cache_mb=1024, max_tokens=32768, max_voxels=6000000)
            with torch.inference_mode():
                model = load_model(lambda: Flow(limits=limits), store.checkpoint("shape_hr"), "cuda", torch.float32)
                context = torch.load(source / "hr_conditioning.pt", map_location="cpu", weights_only=True)["value"]
                coordinates = torch.load(source / "coordinates.pt", map_location="cpu", weights_only=True)["value"].to("cuda")
                generator = torch.Generator(device="cuda").manual_seed(source_report["identity"]["seed"] + 2)
                noise = torch.randn((len(coordinates), 32), generator=generator, device="cuda", dtype=torch.float32)
                report = sweep_pretrained_forwards(model, noise, context, coordinates, destination / "operators.json")
                report["checkpoint"] = {"name": store.checkpoint("shape_hr").name, "revision": store.revision}
                report["frozen_source"] = {"job_id": source_job, "input_sha256": source_report["identity"]["input_sha256"]}
                atomic_json(destination / "operators.json", report)
            return {"path": f"benchmarks/{run_id}/operators.json", "status": report["status"]}
        report = main(["--device", "cuda", "--output", str(destination / "operators.json"),
                       "--trace-dir", str(destination / "traces"),
                       "--cases", *cases, "--sdpa-backends", *backends,
                       "--tokens", str(tokens), "--channels", "1536", "--heads", "12",
                       "--context-tokens", "5", "--context-channels", "1024", "--projection-channels", "2048",
                       "--query-chunks", "256", "1024", "2048", "--cross-query-chunks", "256", "1024", "0",
                       "--ffn-hidden", "8192", "--ffn-rows", "1024", "4096", "8192",
                       *sparse_source,
                       "--sparse-sample-limit", "262144", "--sparse-channels", "64",
                       "--sparse-chunk-rows", "4096", "16384", "--warmup", "2", "--repeats", "5"])
        report["frozen_source"] = {"job_id": source_job, "runtime_fingerprint": json.loads((source / "manifest.json").read_text())["fingerprint"],
                                   "input_sha256": source_report["identity"]["input_sha256"],
                                   "geometry_resolution": source_report["identity"]["geometry_resolution"]}
        atomic_json(destination / "operators.json", report)
        return {"path": f"benchmarks/{run_id}/operators.json", "status": report["status"]}
    finally:
        jobs.commit()


def sweep_pretrained_forwards(model, noise, context, coordinates, output, repeats=3):
    import torch

    from pixalcuda.ops import Limits, rope_phases
    from pixalcuda.runtime import atomic_json
    from scripts.benchmark_cuda_ops import measure

    device = noise.device
    context = {name: value.to(device=device, dtype=torch.float32) for name, value in context.items()}
    phases = rope_phases(coordinates[:, 1:], model.head_dim)
    remaining = 1024 * 1024 ** 2
    projection_bytes = len(noise) * model.input_layer.out_features * 4
    prepared = []
    for block in model.blocks:
        projection = None
        if projection_bytes <= remaining:
            projection = block.cross_attn.proj_linear(context["proj"]).reshape(-1, model.input_layer.out_features)
            remaining -= projection_bytes
        prepared.append({"kv": block.cross_attn.cross_attn_block.prepare_context(context["global"]),
                         "projection": projection, "negative": block.cross_attn.unconditional()})
    base = Limits(conditioning_cache_mb=1024, max_tokens=32768, max_voxels=6000000)
    profiles = [("baseline", base), ("cross-full", replace(base, cross_attention_queries=0)),
                ("self-1024", replace(base, cross_attention_queries=0, attention_queries=1024)),
                ("ffn-8192", replace(base, cross_attention_queries=0, attention_queries=1024, ffn_rows=8192))]
    def configure(limits):
        model.limits = limits
        for block in model.blocks:
            block.self_attn.limits = limits
            block.cross_attn.cross_attn_block.limits = limits
            block.mlp.rows = limits.ffn_rows
    timestep = noise.new_tensor([1000.])
    def operation():
        return model(noise, timestep, context, phases, prepared)
    report = {"status": "running", "scope": "One positive pretrained HR-flow forward at t=1000 on frozen initial noise/context; not a 12-step generation",
              "device": str(device), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "tokens": len(noise), "repeats": repeats, "warmups_per_profile": 1,
              "conditioning_cache_mb": 1024, "cached_blocks": sum(item["projection"] is not None for item in prepared),
              "parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()),
              "round_order": "Forward, reverse, forward; one loaded model, identical inputs and prepared cache", "cases": []}
    if device.type == "cuda":
        report.update(gpu=torch.cuda.get_device_name(device), capability=list(torch.cuda.get_device_capability(device)),
                      gpu_total_bytes=torch.cuda.get_device_properties(device).total_memory,
                      sdpa_flags={"efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                                  "math": torch.backends.cuda.math_sdp_enabled(), "flash": torch.backends.cuda.flash_sdp_enabled()})
    reference = None
    for name, limits in profiles:
        configure(limits)
        actual = operation().detach().cpu()
        if not bool(torch.isfinite(actual).all()):
            raise FloatingPointError("Non-finite pretrained forward: " + name)
        if reference is None:
            reference = actual
        report["cases"].append({"name": name, "parameters": asdict(limits), "samples": [],
                                "warm_max_abs_error": float((actual - reference).abs().max()),
                                "warm_relative_l2_error": float(torch.linalg.vector_norm(actual - reference) /
                                                                 torch.linalg.vector_norm(reference).clamp_min(1e-12))})
        atomic_json(output, report)
        print("PRETRAINED_WARM=" + name, flush=True)
    for iteration in range(repeats):
        order = list(range(len(profiles))) if iteration % 2 == 0 else list(reversed(range(len(profiles))))
        for index in order:
            name, limits = profiles[index]
            configure(limits)
            result = measure(operation, device, 0, 1)
            report["cases"][index]["samples"].append(result)
            atomic_json(output, report)
            print(json.dumps({"round": iteration, "profile": name, "wall_ms": result["median_wall_ms"]}), flush=True)
    for case in report["cases"]:
        case["median_wall_ms"] = statistics.median(sample["median_wall_ms"] for sample in case["samples"])
        case["status"] = "complete"
    report["status"] = "complete"
    atomic_json(output, report)
    return report


@app.local_entrypoint()
def run(run_id: str, source_job: str, output_dir: str = "outputs/modal-tuning", math_attention_only: bool = False,
        pretrained_forwards: bool = False):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    try:
        print(benchmark.remote(run_id, source_job, math_attention_only, pretrained_forwards))
    finally:
        prefix = f"benchmarks/{run_id}"
        for entry in jobs.iterdir(prefix, recursive=True):
            if entry.path.endswith(".json"):
                relative = Path(entry.path).relative_to(prefix)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as output:
                    for chunk in jobs.read_file(entry.path):
                        output.write(chunk)
        path = destination / "operators.json"
        if path.exists():
            report = json.loads(path.read_text())
            report.update(source_revision=revision, environment=SETTINGS.environment,
                          source_revision_origin="Local checkout copied into the tuning image")
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"Saved: {path.resolve()}")
