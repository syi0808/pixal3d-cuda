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


def release(device):
    gc.collect()
    if torch.device(device).type == "cuda":
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
                       "device": str(self.device), "torch": torch.__version__, "stages": [],
                       "memory_limit_is_hard": False}
        if self.device.type == "cuda":
            self.report.update(gpu=torch.cuda.get_device_name(self.device), capability=list(torch.cuda.get_device_capability(self.device)))
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
        release(self.device)
        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        process = psutil.Process()
        record = {"name": name, "peak_sampled_rss_bytes": process.memory_info().rss}
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
                              peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
            self.report["stages"].append(record)
            self.write_report()

    def run_model(self, factory, path, callback, dtype=torch.float16, ignored=()):
        model = load_model(factory, path, self.device, dtype, ignored)
        try:
            return callback(model)
        finally:
            del model
            release(self.device)
