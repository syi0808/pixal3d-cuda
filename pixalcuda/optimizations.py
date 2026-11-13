"""Job-scoped reuse and sparse projection around the pinned official pipeline."""

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import wraps
import inspect

import torch


@dataclass(frozen=True)
class OptimizationOptions:
    share_dino: bool = True
    dino_cache_mb: int = 128
    sparse_projection: bool = True
    conditioning_cache_mb: int = 256

    def __post_init__(self):
        if type(self.share_dino) is not bool or type(self.sparse_projection) is not bool:
            raise ValueError("Optimization switches must be boolean")
        for name in ("dino_cache_mb", "conditioning_cache_mb"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 2048:
                raise ValueError(f"{name} must be an integer between 0 and 2048 MiB")

    @classmethod
    def disabled(cls):
        return cls(False, 0, False, 0)


class Patches:
    def __init__(self):
        self.previous = []

    def set(self, target, name, value):
        self.previous.append((target, name, name in vars(target), vars(target).get(name)))
        setattr(target, name, value)

    def close(self):
        for target, name, existed, previous in reversed(self.previous):
            if existed:
                setattr(target, name, previous)
            else:
                delattr(target, name)
        self.previous.clear()


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def execution_key(tensor):
    enabled = torch.is_autocast_enabled(tensor.device.type)
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device, enabled,
            torch.get_autocast_dtype(tensor.device.type) if enabled else None)


class FeatureCache:
    def __init__(self, budget):
        self.budget = budget
        self.entries = OrderedDict()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "oversized": 0, "used_bytes": 0, "peak_bytes": 0}

    def get(self, model, image, compute):
        parameter = next(model.parameters())
        key = (id(model), parameter.dtype, parameter.device, execution_key(image))
        previous = self.entries.get(key)
        if previous is not None and torch.equal(previous[0], image):
            self.stats["hits"] += 1
            self.entries.move_to_end(key)
            return previous[1].clone()
        self.stats["misses"] += 1
        if previous is not None:
            self.remove(key)
        output = compute()
        size = tensor_bytes(image) + tensor_bytes(output)
        if size > self.budget:
            self.stats["oversized"] += 1
            return output
        while self.stats["used_bytes"] + size > self.budget:
            self.remove(next(iter(self.entries)))
        self.entries[key] = (image.detach().clone(), output.detach().clone())
        self.stats["used_bytes"] += size
        self.stats["peak_bytes"] = max(self.stats["peak_bytes"], self.stats["used_bytes"])
        return output

    def remove(self, key):
        image, output = self.entries.pop(key)
        self.stats["used_bytes"] -= tensor_bytes(image) + tensor_bytes(output)
        self.stats["evictions"] += 1

    def clear(self):
        self.entries.clear()
        self.stats["used_bytes"] = 0


def context_tensors(value):
    if isinstance(value, torch.Tensor):
        return ("tensor",), [value]
    if hasattr(value, "feats"):
        tensors = [value.feats]
        if hasattr(value, "coords"):
            tensors.append(value.coords)
        return (type(value).__name__, len(tensors)), tensors
    if isinstance(value, dict):
        items = [(key, context_tensors(value[key])) for key in sorted(value)]
        return ("dict", tuple((key, record[0]) for key, record in items)), [tensor for key, record in items for tensor in record[1]]
    if isinstance(value, (tuple, list)):
        items = [context_tensors(item) for item in value]
        return (type(value).__name__, tuple(record[0] for record in items)), [tensor for record in items for tensor in record[1]]
    raise TypeError("Only tensor conditioning is supported by the static cache")


class ConditioningCache:
    def __init__(self, budget):
        self.budget = budget
        self.contexts = []
        self.outputs = OrderedDict()
        self.active = None
        self.stats = {"hits": 0, "misses": 0, "snapshot_checks": 0, "snapshot_skips": 0,
                      "snapshot_bytes": 0, "output_bytes": 0, "peak_bytes": 0}

    @property
    def used_bytes(self):
        return self.stats["snapshot_bytes"] + self.stats["output_bytes"]

    def context(self, value):
        structure, tensors = context_tensors(value)
        for index, (previous_structure, snapshots) in enumerate(self.contexts):
            if previous_structure != structure:
                continue
            self.stats["snapshot_checks"] += 1
            if all(execution_key(snapshot) == execution_key(tensor) and torch.equal(snapshot, tensor)
                   for snapshot, tensor in zip(snapshots, tensors)):
                return index
        size = sum(tensor_bytes(tensor) for tensor in tensors)
        if self.stats["snapshot_bytes"] + size > self.budget:
            self.stats["snapshot_skips"] += 1
            return None
        while self.used_bytes + size > self.budget:
            output = self.outputs.popitem(last=True)[1]
            self.stats["output_bytes"] -= tensor_bytes(output)
        self.contexts.append((structure, [tensor.detach().clone() for tensor in tensors]))
        self.stats["snapshot_bytes"] += size
        self.peak()
        return len(self.contexts) - 1

    def peak(self):
        self.stats["peak_bytes"] = max(self.stats["peak_bytes"], self.used_bytes)

    def linear(self, module, original):
        @wraps(original)
        def forward(features):
            if self.active is None or torch.is_grad_enabled() or module.training:
                return original(features)
            key = (id(module), self.active, execution_key(features), module.weight.dtype, module.weight.device)
            if key in self.outputs:
                self.stats["hits"] += 1
                return self.outputs[key].clone()
            self.stats["misses"] += 1
            output = original(features)
            size = tensor_bytes(output)
            if self.used_bytes + size <= self.budget:
                self.outputs[key] = output.detach().clone()
                self.stats["output_bytes"] += size
                self.peak()
            return output
        return forward

    def root(self, model, original):
        signature = inspect.signature(original)
        if "cond" not in signature.parameters:
            for parent in type(model).__mro__:
                method = vars(parent).get("forward")
                if method is not None and "cond" in inspect.signature(method).parameters:
                    signature = inspect.signature(method.__get__(model, type(model)))
                    break
            else:
                raise ValueError("Cannot identify the pinned flow's conditioning signature")
        @wraps(original)
        def forward(*arguments, **options):
            bound = signature.bind(*arguments, **options)
            bound.apply_defaults()
            if not torch.is_grad_enabled() and not model.training:
                self.active = self.context(bound.arguments["cond"])
            try:
                return original(*arguments, **options)
            finally:
                self.active = None
        return forward

    def clear(self):
        self.contexts.clear()
        self.outputs.clear()
        self.active = None
        self.stats["snapshot_bytes"] = self.stats["output_bytes"] = 0


@contextmanager
def sampling_cache(model, budget):
    cache, patches = ConditioningCache(budget), Patches()
    try:
        if getattr(model, "image_attn_mode", None) != "proj":
            raise ValueError("Static conditioning reuse requires the pinned projection-attention flow")
        patches.set(model, "forward", cache.root(model, model.forward))
        for block in model.blocks:
            projection = block.cross_attn.proj_linear
            global_kv = block.cross_attn.cross_attn_block.to_kv
            for module in (projection, global_kv):
                patches.set(module, "forward", cache.linear(module, module.forward))
        yield cache
    finally:
        patches.close()
        cache.clear()


def active_projection(pipeline, sparse_type, original, multiview, statistics):
    signature = inspect.signature(original)
    @wraps(original)
    @torch.no_grad()
    def forward(*arguments, **options):
        bound = signature.bind(*arguments, **options)
        bound.apply_defaults()
        values = bound.arguments
        extractor, coordinates = values["image_cond_model"], values["coords"]
        previous_grid, previous_resolution = extractor.proj_grid, extractor.grid_resolution
        override = values["grid_resolution_override"]
        resolution = previous_resolution if override is None else override
        if type(resolution) is not int or resolution < 2:
            raise ValueError("Projection grid resolution must be an integer >= 2")
        if coordinates.ndim != 2 or coordinates.shape[1] != 4 or not len(coordinates):
            raise ValueError("Active projection requires nonempty [N,4] coordinates")
        if (coordinates[:, 0] != 0).any() or (coordinates[:, 1:] < 0).any() or (coordinates[:, 1:] >= resolution).any():
            raise ValueError("Active projection supports the current single-object worker and in-grid coordinates")
        if pipeline.low_vram:
            extractor.to(pipeline.device)
        grid = previous_grid
        if resolution != previous_resolution:
            grid = previous_grid.__class__(grid_resolution=resolution, image_resolution=previous_grid.image_resolution).to(pipeline.device)
        points = grid.grid_points
        try:
            extractor.proj_grid, extractor.grid_resolution = grid, resolution
            spatial = coordinates[:, 1:].long()
            indices = spatial[:, 0] * resolution**2 + spatial[:, 1] * resolution + spatial[:, 2]
            grid.grid_points = points[indices]
            if multiview:
                global_features, projected = pipeline._run_extractor(extractor, pipeline._views_of(values["image"]),
                    values["camera_angle_x"], values["distance"], values["mesh_scale"])
            else:
                device = pipeline.device
                global_features, projected = extractor(values["image"],
                    camera_angle_x=torch.tensor([values["camera_angle_x"]], device=device),
                    distance=torch.tensor([values["distance"]], device=device),
                    mesh_scale=torch.tensor([values["mesh_scale"]], device=device))
            if projected.shape[:2] != (1, len(coordinates)):
                raise ValueError("Unexpected batch/fusion shape from the pinned extractor")
            statistics.append({"grid_resolution": resolution, "dense_points": resolution**3,
                               "projected_points": len(coordinates), "naf_target_size": list(extractor.naf_target_size)})
            features = projected[0]
            return {"cond": {"global": global_features, "proj": sparse_type(feats=features, coords=coordinates)},
                    "neg_cond": {"global": torch.zeros_like(global_features),
                                 "proj": sparse_type(feats=torch.zeros_like(features), coords=coordinates)}}
        finally:
            grid.grid_points = points
            extractor.proj_grid, extractor.grid_resolution = previous_grid, previous_resolution
            if pipeline.low_vram:
                extractor.cpu()
    return forward


class OptimizationSession:
    def __init__(self, options, backbone_type, extractor_type, sparse_type):
        self.options = options
        self.backbone_type, self.extractor_type, self.sparse_type = backbone_type, extractor_type, sparse_type
        self.patches = Patches()
        self.backbones = {}
        self.model_names = {}
        self.features = FeatureCache(options.dino_cache_mb * 1024**2)
        self.statistics = {"backbone_loads": 0, "shared_backbone_hits": 0, "active_projections": [], "flow_caches": []}

    def __enter__(self):
        if self.options.share_dino:
            original = self.backbone_type.from_pretrained
            def load(path, *arguments, **options):
                if arguments or options:
                    raise ValueError("Shared DINO loader requires the pinned extractor's path-only load contract")
                if str(path) not in self.backbones:
                    self.backbones[str(path)] = original(path)
                    self.statistics["backbone_loads"] += 1
                else:
                    self.statistics["shared_backbone_hits"] += 1
                return self.backbones[str(path)]
            self.patches.set(self.backbone_type, "from_pretrained", staticmethod(load))
        if self.options.dino_cache_mb:
            original_features = self.extractor_type.extract_features
            def extract(extractor, image):
                if torch.is_grad_enabled() or extractor.model.training:
                    return original_features(extractor, image)
                return self.features.get(extractor.model, image, lambda: original_features(extractor, image))
            self.patches.set(self.extractor_type, "extract_features", extract)
        return self

    def install(self, pipeline, multiview):
        self.model_names = {id(model): name for name, model in pipeline.models.items()}
        if self.options.sparse_projection:
            self.patches.set(pipeline, "get_proj_cond_shape", active_projection(pipeline, self.sparse_type,
                pipeline.get_proj_cond_shape, multiview, self.statistics["active_projections"]))
        if self.options.conditioning_cache_mb:
            seen = set()
            for name in ("sparse_structure_sampler", "shape_slat_sampler", "tex_slat_sampler"):
                sampler = getattr(pipeline, name)
                if id(sampler) not in seen:
                    self.patches.set(sampler, "sample", self.sampler(sampler.sample))
                    seen.add(id(sampler))

    def sampler(self, original):
        signature = inspect.signature(original)
        @wraps(original)
        def sample(*arguments, **options):
            model = signature.bind(*arguments, **options).arguments["model"]
            cache = None
            try:
                with sampling_cache(model, self.options.conditioning_cache_mb * 1024**2) as cache:
                    return original(*arguments, **options)
            finally:
                if cache is not None:
                    self.statistics["flow_caches"].append({"model": self.model_names.get(id(model), type(model).__name__),
                        "budget_bytes": cache.budget, **cache.stats})
        return sample

    def report(self):
        return {"options": asdict(self.options), "dino_feature_cache": {"budget_bytes": self.features.budget, **self.features.stats},
                **self.statistics}

    def __exit__(self, *error):
        self.patches.close()
        self.features.clear()
        self.backbones.clear()
