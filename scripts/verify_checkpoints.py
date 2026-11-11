"""Check every official tensor name/shape without allocating full model weights."""

import argparse
import json

from safetensors import safe_open
import torch

from pixalcuda.conditioning import Dino, NAF
from pixalcuda.decoder import SparseDecoder, StructureDecoder
from pixalcuda.flow import Flow
from pixalcuda.weights import ModelStore


def verify(factory, path, ignored=()):
    with torch.device("meta"):
        model = factory()
    expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    with safe_open(path, framework="pt") as checkpoint:
        actual = {key: tuple(checkpoint.get_slice(key).get_shape()) for key in checkpoint.keys()}
    missing = set(expected) - set(actual)
    extra = set(actual) - set(expected) - set(ignored)
    mismatch = [key for key in set(expected) & set(actual) if expected[key] != actual[key]]
    if missing or extra or mismatch:
        raise ValueError(f"{path.name}: missing={missing}, unexpected={extra}, shape_mismatch={mismatch}")
    return {"checkpoint": path.name, "matched_tensors": len(expected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir")
    arguments = parser.parse_args()
    store = ModelStore(arguments.model_dir)
    specifications = [
        ("structure", lambda: Flow(8, 8, 1024, dense=True), ("rope_phases",)),
        ("structure_decoder", StructureDecoder, ()),
        ("shape_lr", Flow, ()),
        ("shape_hr", Flow, ()),
        ("shape_decoder", SparseDecoder, ()),
        ("texture", lambda: Flow(64), ()),
        ("texture_decoder", lambda: SparseDecoder(texture=True), ()),
    ]
    report = [verify(factory, store.checkpoint(name), ignored) for name, factory, ignored in specifications]
    report.append(verify(Dino, store.dino(), ("embeddings.mask_token", "norm.weight", "norm.bias")))
    naf = torch.load(store.naf(), map_location="cpu", weights_only=True)
    with torch.device("meta"):
        model = NAF()
    expected = model.state_dict()
    if set(expected) != set(naf) or any(expected[key].shape != naf[key].shape for key in expected):
        raise ValueError("NAF checkpoint does not match the runtime")
    report.append({"checkpoint": "naf_release.pth", "matched_tensors": len(expected)})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
