"""Explicit one-shot T4 operator tuning in main; no deployed endpoint or model downloads."""

import json
from pathlib import Path
import re
import subprocess

import modal

from deploy.modal_app import ROOT, SETTINGS, jobs, offline


app = modal.App("pixal3d-cuda-tuning")
image = (offline
         .add_local_file(ROOT / "scripts" / "benchmark_cuda_ops.py", "/opt/pixal3d/scripts/benchmark_cuda_ops.py", copy=True)
         .add_local_file(ROOT / "deploy" / "modal_tuning.py", "/opt/pixal3d/deploy/modal_tuning.py", copy=True))


@app.function(image=image, gpu="T4", cpu=(2, 2), memory=(4096, 8192), timeout=600, retries=0,
              min_containers=0, max_containers=1, buffer_containers=0, scaledown_window=2,
              include_source=False, volumes={"/jobs": jobs})
def benchmark(run_id, source_job, math_attention_only=False):
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


@app.local_entrypoint()
def run(run_id: str, source_job: str, output_dir: str = "outputs/modal-tuning", math_attention_only: bool = False):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    try:
        print(benchmark.remote(run_id, source_job, math_attention_only))
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
