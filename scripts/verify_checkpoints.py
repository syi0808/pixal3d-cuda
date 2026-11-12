"""Check every official tensor name/shape without allocating full model weights."""

import argparse
import json
import struct
import urllib.request

from safetensors import safe_open
import torch

from pixalcuda.conditioning import Dino, NAF
from pixalcuda.decoder import SparseDecoder, StructureDecoder
from pixalcuda.flow import Flow
from pixalcuda.weights import CHECKPOINTS, MULTIVIEW_CHECKPOINTS, PIXAL_REPO, ModelStore


def verify(factory, path, ignored=()):
    with safe_open(path, framework="pt") as checkpoint:
        actual = {key: tuple(checkpoint.get_slice(key).get_shape()) for key in checkpoint.keys()}
    return verify_shapes(factory, actual, path.name, ignored)


def verify_shapes(factory, actual, label, ignored=()):
    with torch.device("meta"):
        model = factory()
    expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    missing = set(expected) - set(actual)
    extra = set(actual) - set(expected) - set(ignored)
    mismatch = [key for key in set(expected) & set(actual) if expected[key] != actual[key]]
    if missing or extra or mismatch:
        raise ValueError(f"{label}: missing={missing}, unexpected={extra}, shape_mismatch={mismatch}")
    return {"checkpoint": label, "matched_tensors": len(expected)}


def remote_shapes(revision, filename):
    url = f"https://huggingface.co/{PIXAL_REPO}/resolve/{revision}/ckpts/{filename}.safetensors"
    request = urllib.request.Request(url, headers={"Range": "bytes=0-1048575"})
    with urllib.request.urlopen(request, timeout=120) as response:
        size = struct.unpack("<Q", response.read(8))[0]
        if size > 1048568:
            raise ValueError("Safetensors header exceeds the 1 MiB metadata validation limit")
        header = json.loads(response.read(size))
    return {key: tuple(value["shape"]) for key, value in header.items() if key != "__metadata__"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir")
    parser.add_argument("--multiview", action="store_true")
    parser.add_argument("--headers-only", action="store_true", help="Check public Pixal safetensors metadata only; no full weights, DINO or NAF")
    arguments = parser.parse_args()
    store = ModelStore(arguments.model_dir, multiview=arguments.multiview)
    specifications = [
        ("structure", lambda: Flow(8, 8, 1024, dense=True), ("rope_phases",)),
        ("structure_decoder", StructureDecoder, ()),
        ("shape_lr", Flow, ()),
        ("shape_hr", Flow, ()),
        ("shape_decoder", SparseDecoder, ()),
        ("texture", lambda: Flow(64), ()),
        ("texture_decoder", lambda: SparseDecoder(texture=True), ()),
    ]
    if arguments.headers_only:
        if arguments.model_dir:
            parser.error("--headers-only uses pinned remote metadata, not --model-dir")
        filenames = MULTIVIEW_CHECKPOINTS if arguments.multiview else CHECKPOINTS
        report = [verify_shapes(factory, remote_shapes(store.revision, filenames[name]), filenames[name], ignored)
                  for name, factory, ignored in specifications]
        print(json.dumps({"scope": "Pixal tensor names/shapes only; not weight values or inference", "revision": store.revision,
                          "checkpoints": report}, indent=2))
        return
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
