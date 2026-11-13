"""Synthetic warm operator sweeps; no pretrained downloads or asset-quality claims."""

import argparse
from contextlib import nullcontext
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from pixalcuda.decoder import SparseDecoder, SparseNext
from pixalcuda.flow import Block
from pixalcuda.ops import Limits, attention, chunked_mlp, rope_phases
from pixalcuda.runtime import atomic_json
from pixalcuda.sparse import SparseConv, neighbor_map
from pixalcuda.weights import sha256_file


def grid_coordinates(count, device):
    side = math.ceil(count ** (1 / 3))
    while side ** 3 < count:
        side += 1
    axis = torch.arange(side, device=device)
    spatial = torch.cartesian_prod(axis, axis, axis)[:count]
    return torch.nn.functional.pad(spatial, (1, 0)).int()


def reference_coordinates(model, features, coordinates):
    features = model.from_latent(features)
    topology = neighbor_map(coordinates)
    for level in model.blocks:
        for block in level:
            if isinstance(block, SparseNext):
                features = block(features, topology)
            else:
                features, coordinates, topology, mask = block(features, coordinates, topology)
                mask.cpu()
    return coordinates


def backend_context(name):
    backends = {"math": SDPBackend.MATH, "efficient": SDPBackend.EFFICIENT_ATTENTION, "flash": SDPBackend.FLASH_ATTENTION}
    return nullcontext() if name == "auto" else sdpa_kernel(backends=[backends[name]])


def make_cases(arguments, device):
    cases = []
    if "cross_attention" in arguments.cases:
        cross_shape = (arguments.tokens, arguments.heads, arguments.channels // arguments.heads)
        cross_queries = torch.randn(cross_shape, device=device).transpose(0, 1).unsqueeze(0)
        cross_keys, cross_values = [torch.randn(arguments.context_tokens, arguments.heads, arguments.channels // arguments.heads,
                                               device=device).transpose(0, 1).unsqueeze(0) for _ in range(2)]
        expected = attention(cross_queries, cross_keys, cross_values, min(arguments.query_chunks))
        for chunk in arguments.cross_query_chunks:
            cases.append({"name": f"cross-attention-q{chunk}", "expected": expected,
                          "parameters": {"cross_attention_queries": chunk},
                          "operation": lambda chunk=chunk: attention(cross_queries, cross_keys, cross_values,
                                                                     chunk or arguments.tokens)})
    if "attention" in arguments.cases:
        shape = (arguments.tokens, arguments.heads, arguments.channels // arguments.heads)
        queries, keys, values = [torch.randn(shape, device=device).transpose(0, 1).unsqueeze(0) for _ in range(3)]
        expected = attention(queries, keys, values, min(arguments.query_chunks))
        for backend in arguments.sdpa_backends:
            for chunk in arguments.query_chunks:
                for layout in ("strided", "contiguous"):
                    def operation(chunk=chunk, layout=layout):
                        if layout == "contiguous":
                            return attention(queries.contiguous(), keys.contiguous(), values.contiguous(), chunk)
                        return attention(queries, keys, values, chunk)
                    cases.append({"name": f"attention-{backend}-q{chunk}-{layout}", "operation": operation,
                                  "expected": expected, "backend": backend})
    if "ffn" in arguments.cases:
        ffn_features = torch.randn(arguments.tokens, arguments.channels, device=device)
        ffn_first = torch.nn.Linear(arguments.channels, arguments.ffn_hidden).to(device).eval().requires_grad_(False)
        ffn_second = torch.nn.Linear(arguments.ffn_hidden, arguments.channels).to(device).eval().requires_grad_(False)
        expected = chunked_mlp(ffn_features, ffn_first, ffn_second, min(arguments.ffn_rows))
        for rows in arguments.ffn_rows:
            cases.append({"name": f"ffn-rows{rows}", "expected": expected, "parameters": {"ffn_rows": rows},
                          "operation": lambda rows=rows: chunked_mlp(ffn_features, ffn_first, ffn_second, rows)})
    if "flow_block" in arguments.cases:
        baseline = Limits(attention_queries=min(arguments.query_chunks), cross_attention_queries=min(arguments.query_chunks),
                          ffn_rows=min(arguments.ffn_rows))
        block = Block(arguments.channels, arguments.heads, arguments.context_channels, arguments.projection_channels,
                      arguments.ffn_hidden, baseline).to(device).eval().requires_grad_(False)
        block_features = torch.randn(arguments.tokens, arguments.channels, device=device)
        block_modulation = torch.randn(6 * arguments.channels, device=device) * 0.1
        block_context = {"global": torch.randn(1, arguments.context_tokens, arguments.context_channels, device=device),
                         "proj": torch.randn(1, arguments.tokens, arguments.projection_channels, device=device)}
        block_phases = rope_phases(grid_coordinates(arguments.tokens, device)[:, 1:], arguments.channels // arguments.heads)
        block_prepared = {"kv": block.cross_attn.cross_attn_block.prepare_context(block_context["global"]),
                          "projection": block.cross_attn.proj_linear(block_context["proj"]).reshape_as(block_features)}
        def block_operation():
            return block(block_features, block_modulation, block_context, block_phases, block_prepared, False)
        expected = block_operation()
        profiles = [("block-baseline", baseline)]
        for chunk in arguments.query_chunks:
            for rows in arguments.ffn_rows:
                profiles.append((f"block-crossfull-q{chunk}-ffn{rows}",
                                 Limits(attention_queries=chunk, cross_attention_queries=0, ffn_rows=rows)))
        for name, limits in profiles:
            def prepare(limits=limits):
                block.self_attn.limits = limits
                block.cross_attn.cross_attn_block.limits = limits
                block.mlp.rows = limits.ffn_rows
            cases.append({"name": name, "operation": block_operation, "prepare": prepare,
                          "expected": expected, "parameters": asdict(limits)})
    if "sparse_chunks" in arguments.cases:
        if arguments.sparse_coordinate_checkpoint:
            checkpoint = torch.load(arguments.sparse_coordinate_checkpoint, map_location="cpu", weights_only=True)
            chunk_coordinates = checkpoint["value"]["coordinates"][:arguments.sparse_sample_limit].to(device)
            del checkpoint
        else:
            chunk_coordinates = grid_coordinates(arguments.sparse_voxels, device)
        chunk_features = torch.randn(len(chunk_coordinates), arguments.sparse_channels, device=device)
        chunk_topology = neighbor_map(chunk_coordinates)
        chunk_original = SparseConv(arguments.sparse_channels, arguments.sparse_channels,
                                    Limits(sparse_rows=min(arguments.sparse_chunk_rows))).to(device).eval().requires_grad_(False)
        expected = chunk_original(chunk_features, chunk_topology)
        for rows in arguments.sparse_chunk_rows:
            module = copy.deepcopy(chunk_original)
            module.rows = rows
            cases.append({"name": f"sparse-rows{rows}", "expected": expected,
                          "parameters": {"sparse_rows": rows, "voxels": len(chunk_coordinates)},
                          "operation": lambda module=module: module(chunk_features, chunk_topology)})
    if "sparse" in arguments.cases:
        coordinates = grid_coordinates(arguments.sparse_voxels, device)
        features = torch.randn(len(coordinates), arguments.sparse_channels, device=device)
        topology = neighbor_map(coordinates)
        original = SparseConv(arguments.sparse_channels, arguments.sparse_channels,
                              Limits(sparse_rows=arguments.sparse_rows)).to(device).eval().requires_grad_(False)
        packed = copy.deepcopy(original).prepack_weights_()
        expected = original(features, topology)
        cases.extend([
            {"name": "neighbor-map", "operation": lambda coordinates=coordinates: neighbor_map(coordinates)},
            {"name": "sparse-original", "operation": lambda module=original, features=features, topology=topology: module(features, topology), "expected": expected},
            {"name": "sparse-prepacked", "operation": lambda module=packed, features=features, topology=topology: module(features, topology), "expected": expected},
        ])
    if "coordinates" in arguments.cases:
        coordinates = grid_coordinates(arguments.coordinate_voxels, device)
        features = torch.randn(len(coordinates), 32, device=device)
        decoder = SparseDecoder(channels=(128, 64), block_counts=(0, 0), verbose=False).to(device).eval().requires_grad_(False)
        expected = reference_coordinates(decoder, features, coordinates)
        cases.extend([
            {"name": "coordinates-reference", "operation": lambda: reference_coordinates(decoder, features, coordinates), "expected": expected},
            {"name": "coordinates-fast", "operation": lambda: decoder(features, coordinates, coordinates_only=True), "expected": expected},
        ])
    if "projection" in arguments.cases:
        context = torch.randn(1, arguments.tokens, arguments.projection_channels, device=device)
        projection = torch.nn.Linear(arguments.projection_channels, arguments.channels).to(device).eval().requires_grad_(False)
        def repeated_projection():
            output = None
            for _ in range(12):
                output = projection(context)
            return output
        def cached_projection():
            cached = projection(context)
            output = None
            for _ in range(12):
                output = cached
            return output
        expected = projection(context)
        cases.extend([
            {"name": "projection-recompute-12", "operation": repeated_projection, "expected": expected},
            {"name": "projection-precompute-and-reuse-12", "operation": cached_projection, "expected": expected},
        ])
    return cases


def measure(operation, device, warmup, repeats):
    cuda = device.type == "cuda"
    for _ in range(warmup):
        operation()
    if cuda:
        torch.cuda.synchronize(device)
    walls, intervals = [], []
    peaks, reserved, baselines = [], [], []
    for _ in range(repeats):
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
            baselines.append(torch.cuda.memory_allocated(device))
            beginning, ending = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            beginning.record()
        started = time.perf_counter()
        output = operation()
        if cuda:
            ending.record()
            ending.synchronize()
            intervals.append(beginning.elapsed_time(ending))
            peaks.append(torch.cuda.max_memory_allocated(device))
            reserved.append(torch.cuda.max_memory_reserved(device))
        walls.append((time.perf_counter() - started) * 1000)
        del output
    result = {"wall_ms": walls, "median_wall_ms": statistics.median(walls)}
    if cuda:
        result.update(cuda_interval_ms=intervals, median_cuda_interval_ms=statistics.median(intervals),
                      peak_allocated_bytes=max(peaks), peak_reserved_bytes=max(reserved),
                      baseline_allocated_bytes=max(baselines),
                      incremental_peak_allocated_bytes=max(peak - baseline for peak, baseline in zip(peaks, baselines)))
    return result


def trace_case(operation, device, directory, name):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False, acc_events=True) as profiler:
        with torch.profiler.record_function(f"pixalcuda.benchmark/{name}"):
            operation()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    path = directory / f"{name}.json"
    profiler.export_chrome_trace(str(path))
    events = list(profiler.key_averages())
    cuda_events = sum(str(event.device_type).endswith("CUDA") for event in profiler.events())
    return {"path": str(path), "sdpa_operators": sorted(event.key for event in events if "scaled_dot_product" in event.key),
            "cuda_device_events": cuda_events, "cuda_activity_captured": cuda_events > 0 if device.type == "cuda" else None,
            "operators": [{"name": event.key, "calls": event.count, "self_cpu_us": event.self_cpu_time_total,
                           "self_device_us": getattr(event, "self_device_time_total", 0)} for event in events]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", default="outputs/operator-benchmark.json")
    parser.add_argument("--trace-dir")
    parser.add_argument("--cases", nargs="+", choices=("attention", "sparse", "coordinates", "projection", "cross_attention", "ffn", "flow_block", "sparse_chunks"),
                        default=["attention", "sparse", "coordinates", "projection"])
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--channels", type=int, default=1536)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--projection-channels", type=int, default=2048)
    parser.add_argument("--sparse-voxels", type=int, default=4096)
    parser.add_argument("--sparse-channels", type=int, default=64)
    parser.add_argument("--sparse-rows", type=int, default=4096)
    parser.add_argument("--coordinate-voxels", type=int, default=128)
    parser.add_argument("--query-chunks", nargs="+", type=int, default=[256, 512, 1024, 2048])
    parser.add_argument("--cross-query-chunks", nargs="+", type=int, default=[256, 1024, 0])
    parser.add_argument("--context-tokens", type=int, default=5)
    parser.add_argument("--context-channels", type=int, default=1024)
    parser.add_argument("--ffn-hidden", type=int, default=8192)
    parser.add_argument("--ffn-rows", nargs="+", type=int, default=[1024, 4096, 8192])
    parser.add_argument("--sparse-chunk-rows", nargs="+", type=int, default=[4096, 16384])
    parser.add_argument("--sparse-coordinate-checkpoint")
    parser.add_argument("--sparse-sample-limit", type=int, default=262144)
    parser.add_argument("--sdpa-backends", nargs="+", choices=("auto", "math", "efficient", "flash"), default=["auto"])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=2)
    arguments = parser.parse_args(argv)
    positive = (arguments.tokens, arguments.channels, arguments.heads, arguments.projection_channels,
                arguments.sparse_voxels, arguments.sparse_channels, arguments.sparse_rows,
                arguments.coordinate_voxels, arguments.repeats, arguments.cpu_threads, *arguments.query_chunks,
                arguments.context_tokens, arguments.context_channels, arguments.ffn_hidden, arguments.sparse_sample_limit,
                *arguments.ffn_rows, *arguments.sparse_chunk_rows)
    if any(value <= 0 for value in positive) or any(value < 0 for value in arguments.cross_query_chunks) or arguments.warmup < 0 or arguments.channels % arguments.heads:
        parser.error("Sizes/repeats must be positive, warmup nonnegative, and channels divisible by heads")
    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A real NVIDIA CUDA GPU is required; --device cpu is a smoke test, not a T4 benchmark")
    device = torch.device(arguments.device)
    torch.set_num_threads(arguments.cpu_threads)
    torch.manual_seed(arguments.seed)
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    report = {"schema": "pixalcuda.operators.v1", "scope": "Synthetic operators, not pretrained inference or asset quality",
              "device": str(device), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "platform": platform.platform(), "architecture": platform.machine(),
              "source_revision": revision, "source_files": {path.name: sha256_file(path) for path in sorted((root / "pixalcuda").glob("*.py"))},
              "script_sha256": sha256_file(__file__), "configuration": vars(arguments), "cases": [],
              "defaults": asdict(Limits()), "float32_matmul_precision": torch.get_float32_matmul_precision()}
    if arguments.sparse_coordinate_checkpoint:
        report["sparse_geometry_source"] = {"sha256": sha256_file(arguments.sparse_coordinate_checkpoint),
                                            "features": "Random FP32 features on a prefix of frozen decoded geometry; not a full decoder benchmark"}
    if device.type == "cuda":
        report.update(gpu=torch.cuda.get_device_name(device), capability=list(torch.cuda.get_device_capability(device)),
                      gpu_total_bytes=torch.cuda.get_device_properties(device).total_memory,
                      free_device_bytes_before=torch.cuda.mem_get_info(device)[0],
                      allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32)
        try:
            report["driver_version"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True,
                stderr=subprocess.DEVNULL).splitlines()[0].strip()
        except (OSError, subprocess.CalledProcessError, IndexError):
            report["driver_version"] = None
    trace_directory = Path(arguments.trace_dir) if arguments.trace_dir else None
    if trace_directory:
        trace_directory.mkdir(parents=True, exist_ok=True)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report["status"] = "running"
    atomic_json(output, report)
    try:
        with torch.inference_mode():
            cases = make_cases(arguments, device)
            for case in cases:
                record = {"name": case["name"], "status": "running"}
                if "parameters" in case:
                    record["parameters"] = case["parameters"]
                report["cases"].append(record)
                try:
                    if "prepare" in case:
                        case["prepare"]()
                    with backend_context(case.get("backend", "auto")):
                        if "expected" in case:
                            actual = case["operation"]()
                            torch.testing.assert_close(actual, case["expected"], atol=1e-5, rtol=1e-4)
                            record["max_abs_error"] = float((actual - case["expected"]).abs().max())
                            del actual
                        record.update(measure(case["operation"], device, arguments.warmup, arguments.repeats))
                        if trace_directory:
                            record["trace"] = trace_case(case["operation"], device, trace_directory, case["name"])
                    record["status"] = "complete"
                    if device.type == "cuda":
                        record["free_device_bytes_after"] = torch.cuda.mem_get_info(device)[0]
                except RuntimeError as error:
                    if case.get("backend", "auto") != "auto" and any(message in str(error) for message in ("No available kernel", "No viable backend")):
                        record.update(status="unsupported", error=str(error))
                    else:
                        record.update(status="failed", error=f"{type(error).__name__}: {error}")
                        raise
                except BaseException as error:
                    record.update(status="failed", error=f"{type(error).__name__}: {error}")
                    raise
                print(json.dumps({key: record[key] for key in ("name", "status", "median_wall_ms") if key in record}), flush=True)
                atomic_json(output, report)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(output, report)
    return report


if __name__ == "__main__":
    main()
