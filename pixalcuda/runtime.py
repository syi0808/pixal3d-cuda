"""Stage-scoped ownership, restart-safe artifacts, and resource telemetry."""

from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import threading
import time

import psutil
import torch

from .weights import load_model


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def release(device, synchronize=True):
    gc.collect()
    if torch.device(device).type == "cuda":
        if synchronize:
            torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


class Runtime:
    @classmethod
    def restore(cls, directory, device="cpu"):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        report = json.loads((directory / "report.json").read_text())
        identity = manifest["identity"]
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if fingerprint != manifest["fingerprint"] or report["identity"] != identity:
            raise ValueError("Export manifest/report identity mismatch")
        if "source_files" in identity:
            actual = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted(Path(__file__).parent.glob("*.py"))}
            if actual != identity["source_files"]:
                raise ValueError("Runtime source changed; use the same source revision for CPU export")
        runtime = cls.__new__(cls)
        runtime.directory, runtime.device = directory, torch.device(device)
        runtime.fingerprint, runtime.resume, runtime.report = fingerprint, True, report
        return runtime

    def __init__(self, directory, device, identity, resume=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.resume = resume
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        manifest = self.directory / "manifest.json"
        if manifest.exists():
            previous = json.loads(manifest.read_text())
            if previous["fingerprint"] != self.fingerprint:
                raise ValueError("Checkpoint settings/input/weights changed. Select a new output directory.")
            if not resume:
                raise ValueError("Output directory already has a run. Use --resume or a new output directory.")
        atomic_json(manifest, {"fingerprint": self.fingerprint, "identity": identity})
        self.report = {"schema": "pixalcuda.runtime.v1", "status": "running", "identity": identity,
                       "device": str(self.device), "torch": torch.__version__, "stages": [], "models": [],
                       "memory_limit_is_hard": False}
        if self.device.type == "cuda":
            self.report.update(gpu=torch.cuda.get_device_name(self.device), capability=list(torch.cuda.get_device_capability(self.device)),
                               gpu_total_bytes=torch.cuda.get_device_properties(self.device).total_memory)
        self.write_report()

    def write_report(self):
        atomic_json(self.directory / "report.json", self.report)

    def load(self, name):
        path = self.directory / f"{name}.pt"
        if not self.resume or not path.exists():
            return None
        record = torch.load(path, map_location="cpu", weights_only=True)
        if record["fingerprint"] != self.fingerprint:
            raise ValueError(f"Checkpoint fingerprint mismatch: {name}")
        print(f"  Resuming {name}", flush=True)
        return record["value"]

    def save(self, name, value):
        def cpu(item):
            if isinstance(item, torch.Tensor):
                return item.detach().cpu()
            if isinstance(item, dict):
                return {key: cpu(child) for key, child in item.items()}
            if isinstance(item, (list, tuple)):
                return [cpu(child) for child in item]
            return item
        path = self.directory / f"{name}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save({"fingerprint": self.fingerprint, "value": cpu(value)}, temporary)
        temporary.replace(path)

    @contextmanager
    def stage(self, name):
        previous_stage = getattr(self, "current_stage", None)
        self.current_stage = name
        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        process = psutil.Process()
        record = {"name": name, "peak_sampled_rss_bytes": process.memory_info().rss}
        if cuda:
            record["free_device_bytes_before"] = torch.cuda.mem_get_info(self.device)[0]
        finished = threading.Event()
        def monitor():
            while not finished.wait(0.1):
                record["peak_sampled_rss_bytes"] = max(record["peak_sampled_rss_bytes"], process.memory_info().rss)
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        started = time.perf_counter()
        print(f"\n=== {name} ===", flush=True)
        try:
            yield
            if cuda:
                torch.cuda.synchronize(self.device)
            record["status"] = "complete"
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            self.report["status"] = "failed"
            raise
        finally:
            finished.set()
            thread.join()
            record["elapsed_seconds"] = time.perf_counter() - started
            record["peak_sampled_rss_bytes"] = max(record["peak_sampled_rss_bytes"], process.memory_info().rss)
            if cuda:
                record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
                              free_device_bytes_after=torch.cuda.mem_get_info(self.device)[0])
            self.report["stages"].append(record)
            self.current_stage = previous_stage
            self.write_report()

    def run_model(self, factory, path, callback, dtype=torch.float16, ignored=()):
        path = Path(path)
        cuda = self.device.type == "cuda"
        model, completed = None, False
        record = {"checkpoint": path.name, "dtype": str(dtype), "stage": getattr(self, "current_stage", None)}
        phase, started = "load", time.perf_counter()
        try:
            with torch.profiler.record_function(f"pixalcuda.load/{path.name}"):
                model = load_model(factory, path, self.device, dtype, ignored)
                if cuda:
                    torch.cuda.synchronize(self.device)
            record["load_wall_seconds"] = time.perf_counter() - started
            record["parameter_bytes"] = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
            phase, started = "callback", time.perf_counter()
            if cuda:
                beginning, ending = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                beginning.record()
            with torch.profiler.record_function(f"pixalcuda.callback/{path.name}"):
                output = callback(model)
                if cuda:
                    ending.record()
                    torch.cuda.synchronize(self.device)
            record["callback_wall_seconds"] = time.perf_counter() - started
            if cuda:
                record["callback_cuda_interval_ms"] = beginning.elapsed_time(ending)
            if hasattr(model, "last_cache_report"):
                record["conditioning_cache"] = model.last_cache_report
            record["status"], completed = "complete", True
            return output
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            record[f"{phase}_wall_seconds"] = time.perf_counter() - started
            raise
        finally:
            del model
            started = time.perf_counter()
            with torch.profiler.record_function("pixalcuda.model_cleanup"):
                release(self.device, synchronize=not completed)
            record["cleanup_wall_seconds"] = time.perf_counter() - started
            self.report.setdefault("models", []).append(record)
            self.write_report()
