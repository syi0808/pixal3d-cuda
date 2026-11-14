"""Private, bounded first-divergence probes around the pinned native pipeline."""

from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import gc
import hashlib
import inspect
from pathlib import Path
import time

import torch

from .optimizations import OptimizationOptions, OptimizationSession, Patches
from .runtime import atomic_json


PROFILES = {
    "reference_first": OptimizationOptions.disabled(),
    "reference_repeat": OptimizationOptions.disabled(),
    "dino": OptimizationOptions(True, 128, False, 0),
    "projection": OptimizationOptions(False, 0, True, 0),
    "conditioning": OptimizationOptions(False, 0, False, 256),
    "optimized": OptimizationOptions(),
    "reference_splitk2": OptimizationOptions.disabled(),
}


def tensor_digest(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def tensor_comparison(expected, actual, atol=1e-6, rtol=1e-5):
    result = {"expected_shape": list(expected.shape), "actual_shape": list(actual.shape),
              "expected_dtype": str(expected.dtype), "actual_dtype": str(actual.dtype)}
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return {**result, "exact": False, "passed": False, "reason": "shape/dtype mismatch"}
    expected, actual = expected.detach().cpu(), actual.detach().cpu()
    exact = bool(torch.equal(expected, actual))
    finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
    result.update(exact=exact, finite=finite)
    if not expected.is_floating_point():
        return {**result, "passed": exact, "comparison": "exact"}
    if not finite:
        return {**result, "passed": False, "reason": "non-finite tensor"}
    difference = actual.double() - expected.double()
    result.update(passed=bool(torch.allclose(expected, actual, atol=atol, rtol=rtol)),
                  max_abs=float(difference.abs().max()) if difference.numel() else 0.0,
                  relative_l2=float(difference.norm() / expected.double().norm().clamp_min(1e-30)))
    return result


def tensor_tree(value, prefix="value"):
    if isinstance(value, torch.Tensor):
        return {prefix: value}
    if hasattr(value, "feats") and hasattr(value, "coords"):
        return {prefix + ".features": value.feats, prefix + ".coordinates": value.coords}
    if isinstance(value, dict):
        result = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise ValueError("Trace dictionaries require string keys")
            result.update(tensor_tree(value[key], prefix + "." + key))
        return result
    if isinstance(value, (tuple, list)):
        return {key: tensor for index, item in enumerate(value)
                for key, tensor in tensor_tree(item, prefix + f".{index}").items()}
    if value is None:
        return {}
    raise TypeError(f"Unsupported trace value: {type(value).__name__}")


class PrefixTrace:
    def __init__(self, directory, reference_directory=None, max_snapshot_bytes=512 * 1024**2):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.reference_directory = Path(reference_directory) if reference_directory is not None else None
        self.max_snapshot_bytes = max_snapshot_bytes
        self.snapshot_bytes = 0
        self.events = []
        self.first_nonexact = None
        self.first_failure = None

    def emit(self, name, value):
        tensors = {key: tensor.detach().cpu().clone() for key, tensor in tensor_tree(value).items()}
        size = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        if size + self.snapshot_bytes > self.max_snapshot_bytes:
            raise RuntimeError("Private trace snapshot budget exceeded; no partial parity acceptance")
        self.snapshot_bytes += size
        event = {"index": len(self.events), "name": name,
                 "tensors": {key: {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                                   "sha256": tensor_digest(tensor), "finite": bool(torch.isfinite(tensor).all())}
                             for key, tensor in tensors.items()}}
        filename = f"{len(self.events):04d}.pt"
        if self.reference_directory is None:
            torch.save({"name": name, "tensors": tensors}, self.directory / filename)
        else:
            path = self.reference_directory / filename
            previous = torch.load(path, map_location="cpu", weights_only=True) if path.exists() else None
            if previous is None or previous["name"] != name or previous["tensors"].keys() != tensors.keys():
                event["comparison"] = {"passed": False, "exact": False, "reason": "event/tree structure mismatch"}
            else:
                comparisons = {key: tensor_comparison(previous["tensors"][key], tensor) for key, tensor in tensors.items()}
                event["comparison"] = {"passed": all(item["passed"] for item in comparisons.values()),
                                       "exact": all(item["exact"] for item in comparisons.values()), "tensors": comparisons}
            if not event["comparison"]["exact"] and self.first_nonexact is None:
                self.first_nonexact = name
            if not event["comparison"]["passed"] and self.first_failure is None:
                self.first_failure = name
        self.events.append(event)
        atomic_json(self.directory / "trace.json", self.summary())

    def summary(self):
        return {"events": self.events, "first_nonexact": self.first_nonexact, "first_failure": self.first_failure,
                "snapshot_payload_bytes": self.snapshot_bytes, "atol": 1e-6, "rtol": 1e-5}

    def finish(self, reference_event_count=None):
        if reference_event_count is not None and len(self.events) != reference_event_count:
            self.first_failure = self.first_failure or "event_count"
        result = self.summary()
        result["passed"] = self.first_failure is None if self.reference_directory is not None else None
        for field, check in (("first_activation_nonexact", "exact"), ("first_activation_failure", "passed")):
            result[field] = next((event["name"] for event in self.events if ".rng" not in event["name"]
                                  and check in event.get("comparison", {}) and not event["comparison"][check]), None)
        atomic_json(self.directory / "trace.json", result)
        return result


@contextmanager
def fixed_splitk(tuner_type, alternate=False):
    original = tuner_type.__call__
    choices = {}

    def forward(tuner, *arguments, **options):
        if tuner.kernel.__name__ != "sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk":
            return original(tuner, *arguments, **options)
        key = str(tuner.key_fn(*arguments, **options))
        configurations = tuner.configs or tuner.config_fn(*arguments, **options)
        split = 2 if alternate and key == "(2^13, 512, 512, 27)" else 1
        selected = {"SPLITK": split}
        if selected not in configurations:
            raise RuntimeError(f"Diagnostic SPLITK={split} unavailable for {key}")
        choices[key] = selected
        return tuner.kernel(*arguments, **options, **selected)

    tuner_type.__call__ = forward
    try:
        yield choices
    finally:
        tuner_type.__call__ = original


@contextmanager
def alias_loaded_backbones(pipeline, enabled):
    names = ("ss", "shape_512", "shape_1024", "tex_1024")
    extractors = [getattr(pipeline, "image_cond_model_" + name) for name in names]
    originals = [extractor.model for extractor in extractors]
    try:
        if enabled:
            for extractor in extractors:
                extractor.model = originals[0]
        yield
    finally:
        for extractor, original in zip(extractors, originals):
            extractor.model = original


class PrefixComplete(Exception):
    pass


class PrefixProbe:
    def __init__(self, pipeline, trace, kernel_profile=False):
        self.pipeline, self.trace = pipeline, trace
        self.kernel_profile = kernel_profile
        self.patches = Patches()
        self.measurements = []
        self.kernel_summary = None
        self.active_flow = None
        self.step = 0

    def measured(self, name, compute):
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            return compute()
        finally:
            torch.cuda.synchronize()
            self.measurements.append({"name": name, "inclusive_seconds": time.perf_counter() - started})

    def rng(self, name):
        self.trace.emit(name, {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()})

    def condition(self, original, sparse=False):
        signature = inspect.signature(original)

        def forward(*arguments, **options):
            bound = signature.bind(*arguments, **options)
            if sparse and bound.arguments["image_cond_model"] is self.pipeline.image_cond_model_shape_1024:
                self.trace.emit("coordinates.shape_hr", bound.arguments["coords"])
                raise PrefixComplete()
            name = "conditioning.shape_lr" if sparse else "conditioning.ss"
            self.rng(name + ".rng_before")
            result = self.measured(name, lambda: original(*arguments, **options))
            self.trace.emit(name, result)
            self.rng(name + ".rng_after")
            return result
        return forward

    def sampler(self, original):
        signature = inspect.signature(original)

        def forward(*arguments, **options):
            bound = signature.bind(*arguments, **options)
            name = "ss" if bound.arguments["model"] is self.pipeline.models["sparse_structure_flow_model"] else "shape_lr"
            self.active_flow, self.step = name, 0
            self.trace.emit(name + ".noise", bound.arguments["noise"])
            self.rng(name + ".rng_before_sampling")
            result = self.measured(name + ".sampler", lambda: original(*arguments, **options))
            self.trace.emit(name + ".sample", result.samples)
            return result
        return forward

    def sample_once(self, original):
        def forward(*arguments, **options):
            name = f"{self.active_flow}.step.{self.step:02d}"
            capture = self.kernel_profile and self.active_flow == "shape_lr" and self.step == 1
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                         torch.profiler.ProfilerActivity.CUDA],
                                               record_shapes=True) if capture else nullcontext()
            with profiler as recording:
                result = self.measured(name, lambda: original(*arguments, **options))
            if capture:
                recording.export_chrome_trace(str(self.trace.directory / "kernel-trace.json"))
                averages = recording.key_averages()
                self.kernel_summary = [{"operator": item.key, "calls": item.count,
                                        "self_device_microseconds": item.self_device_time_total,
                                        "self_cpu_microseconds": item.self_cpu_time_total}
                                       for item in sorted(averages, key=lambda item: item.self_device_time_total, reverse=True)[:25]]
            self.trace.emit(name, result.pred_x_prev)
            self.step += 1
            return result
        return forward

    def output(self, original, name):
        def forward(*arguments, **options):
            result = self.measured(name, lambda: original(*arguments, **options))
            self.trace.emit(name, result)
            return result
        return forward

    def convolution(self, original, name):
        def forward(value):
            self.trace.emit(name + ".input", value)
            result = self.measured(name, lambda: original(value))
            self.trace.emit(name + ".output", result)
            return result
        return forward

    def __enter__(self):
        pipeline = self.pipeline
        self.patches.set(pipeline, "get_proj_cond_ss", self.condition(pipeline.get_proj_cond_ss))
        self.patches.set(pipeline, "get_proj_cond_shape", self.condition(pipeline.get_proj_cond_shape, sparse=True))
        self.patches.set(pipeline, "sample_sparse_structure", self.output(pipeline.sample_sparse_structure, "coordinates.ss"))
        decoder = pipeline.models["shape_slat_decoder"]
        self.patches.set(decoder, "upsample", self.output(decoder.upsample, "coordinates.upsampled"))
        for name, module in decoder.named_modules():
            if type(module).__name__ == "SparseConv3d" and module.in_channels == module.out_channels == 512:
                self.patches.set(module, "forward", self.convolution(module.forward, "decoder." + name))
        for sampler in (pipeline.sparse_structure_sampler, pipeline.shape_slat_sampler):
            self.patches.set(sampler, "sample", self.sampler(sampler.sample))
            self.patches.set(sampler, "sample_once", self.sample_once(sampler.sample_once))
        return self

    def __exit__(self, *error):
        self.patches.close()


@torch.inference_mode()
def run_prefix_profiles(pipeline, image, camera, directory, report):
    from flex_gemm.utils.autotuner import PersistentCacheAutoTuner, get_autotune_cache
    from pixal3d.modules.sparse import SparseTensor
    from pixal3d.trainers.flow_matching.mixins import image_conditioned_proj

    directory = Path(directory)
    results = []
    reference_count = None
    for name, options in PROFILES.items():
        trace = PrefixTrace(directory / name, directory / "reference_first" if results else None)
        with report.stage(name):
            with alias_loaded_backbones(pipeline, options.share_dino), fixed_splitk(PersistentCacheAutoTuner, name == "reference_splitk2") as choices:
                with OptimizationSession(options, image_conditioned_proj.DINOv3ViTModel,
                                         image_conditioned_proj.DinoV3ProjFeatureExtractor, SparseTensor) as session:
                    session.install(pipeline, False)
                    with PrefixProbe(pipeline, trace, name in ("reference_repeat", "optimized")) as probe:
                        try:
                            pipeline.run(image, camera_params=camera, seed=42, preprocess_image=False,
                                         return_latent=True, pipeline_type="1024_cascade", max_num_tokens=49152)
                        except PrefixComplete:
                            pass
                        else:
                            raise RuntimeError("Prefix probe unexpectedly continued past shape upsampling")
                cache_report = session.report()
            summary = trace.finish(reference_count)
            if reference_count is None:
                reference_count = len(trace.events)
            results.append({"profile": name, "options": asdict(options),
                            "splitk_choices": choices, "first_nonexact": summary["first_nonexact"],
                            "first_failure": summary["first_failure"], "passed": summary["passed"],
                            "first_activation_nonexact": summary["first_activation_nonexact"],
                            "first_activation_failure": summary["first_activation_failure"],
                            "control_role": "kernel-choice negative control" if name == "reference_splitk2" else "reference" if name.startswith("reference") else "hook ablation",
                            "events": len(trace.events), "snapshot_payload_bytes": trace.snapshot_bytes,
                            "timings_inclusive_with_diagnostic_overhead": probe.measurements,
                            "kernel_profile_shape_lr_step_1": probe.kernel_summary,
                            "optimization_counters": cache_report})
            atomic_json(directory / "prefix-results.json", results)
            atomic_json(directory / (name + "-autotune.json"), get_autotune_cache())
        gc.collect()
        torch.cuda.empty_cache()
    return results
