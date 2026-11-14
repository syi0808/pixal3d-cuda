"""Skip overwritten random fills only for fully covered, pinned native models."""

from contextlib import contextmanager
import json
from pathlib import Path

import torch

from .optimizations import Patches


MODEL_TYPES = {"SparseStructureFlowModel", "SparseStructureDecoder", "SLatFlowModel",
               "ElasticSLatFlowModel", "SparseUnetVaeDecoder", "FlexiDualGridVaeDecoder"}
RANDOM_INITIALIZERS = ("kaiming_uniform_", "kaiming_normal_", "xavier_uniform_", "xavier_normal_",
                       "normal_", "uniform_", "trunc_normal_")


def audit_coverage(model, shapes, model_type):
    parameters = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers(remove_duplicate=False))
    missing = [name for name in parameters if name not in shapes]
    missing_buffers = [name for name in buffers if name not in shapes]
    mismatched = [name for name, tensor in {**parameters, **buffers}.items()
                  if name in shapes and list(tensor.shape) != list(shapes[name])]
    if missing or missing_buffers or mismatched:
        raise ValueError(f"Checkpoint coverage rejected: parameters={missing}, buffers={missing_buffers}, shapes={mismatched}")
    return {"model_type": model_type, "parameter_tensors": len(parameters),
            "buffer_tensors": len(buffers), "covered_buffer_tensors": len(buffers),
            "parameter_elements": sum(tensor.numel() for tensor in parameters.values()),
            "covered_parameter_tensors": len(parameters), "checkpoint_tensors": len(shapes),
            "derived_buffers": [],
            "unexpected_checkpoint_keys": sorted(set(shapes) - parameters.keys() - buffers.keys())}


@contextmanager
def covered_initialization(enabled=True):
    records = []
    if not enabled:
        yield records
        return
    from safetensors import safe_open
    from pixal3d import models

    original_loader = models.from_pretrained
    patches = Patches()

    def load(path, **options):
        path = Path(path)
        config = json.loads(Path(str(path) + ".json").read_text())
        name = config["name"]
        if name not in MODEL_TYPES:
            raise ValueError(f"Unaudited model constructor: {name}")
        with safe_open(str(path) + ".safetensors", framework="pt", device="cpu") as checkpoint:
            shapes = {key: checkpoint.get_slice(key).get_shape() for key in checkpoint.keys()}
        kind = getattr(models, name)
        original_init = kind.__init__
        original_state = kind.load_state_dict
        local = Patches()
        record = {"checkpoint": path.name, "skipped_calls": 0, "skipped_elements": 0}

        def initialize(model, *arguments, **keywords):
            random_patches = Patches()

            def skip(tensor, *arguments, **keywords):
                record["skipped_calls"] += 1
                record["skipped_elements"] += tensor.numel()
                return tensor
            try:
                for initializer in RANDOM_INITIALIZERS:
                    random_patches.set(torch.nn.init, initializer, skip)
                original_init(model, *arguments, **keywords)
            finally:
                random_patches.close()
            record.update(audit_coverage(model, shapes, name))

        def apply(model, state, *arguments, **keywords):
            result = original_state(model, state, *arguments, **keywords)
            if result.missing_keys:
                raise ValueError("Checkpoint application left uncovered model state")
            record.update(applied=True, missing_keys=list(result.missing_keys), unexpected_keys=list(result.unexpected_keys))
            return result
        try:
            local.set(kind, "__init__", initialize)
            local.set(kind, "load_state_dict", apply)
            result = original_loader(str(path), **options)
            if not record.get("applied"):
                raise ValueError("Official loader did not apply the audited checkpoint")
            records.append(record)
            return result
        finally:
            local.close()
    try:
        patches.set(models, "from_pretrained", load)
        yield records
    finally:
        patches.close()
