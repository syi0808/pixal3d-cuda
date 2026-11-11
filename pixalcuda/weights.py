"""Pinned downloads and strict, tensor-at-a-time materialization from meta."""

import hashlib
import json
from pathlib import Path
import urllib.request

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open


PIXAL_REPO = "TencentARC/Pixal3D"
PIXAL_REVISION = "0b31f9160aa400719af409098bff7936a932f726"
DINO_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINO_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
MOGE_REPO = "Ruicheng/moge-2-vitl"
MOGE_REVISION = "39c4d5e957afe587e04eec59dc2bcc3be5ecd968"
NAF_URL = "https://github.com/valeoai/NAF/releases/download/model/naf_release.pth"
NAF_SHA256 = "c096c1ab2217a5c3ac136365f721685e2201379cb69d509cfb0261183847c98f"
CHECKPOINTS = {
    "structure": "ss_flow_img_dit_1_3B_64_bf16",
    "structure_decoder": "ss_dec_conv3d_16l8_fp16",
    "shape_lr": "slat_flow_img2shape_dit_1_3B_512_bf16",
    "shape_hr": "slat_flow_img2shape_dit_1_3B_1024_bf16",
    "shape_decoder": "shape_dec_next_dc_f16c32_fp16",
    "texture": "slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
    "texture_decoder": "tex_dec_next_dc_f16c32_fp16",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ModelStore:
    def __init__(self, directory=None):
        self.directory = Path(directory).expanduser() if directory else None

    def pixal_file(self, filename):
        if self.directory is not None:
            path = self.directory / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
        return Path(hf_hub_download(PIXAL_REPO, filename, revision=PIXAL_REVISION))

    def checkpoint(self, name):
        return self.pixal_file(f"ckpts/{CHECKPOINTS[name]}.safetensors")

    def config(self):
        return json.loads(self.pixal_file("pipeline.json").read_text())["args"]

    def dino(self):
        return Path(hf_hub_download(DINO_REPO, "model.safetensors", revision=DINO_REVISION))

    def naf(self):
        from huggingface_hub.constants import HF_HUB_CACHE
        destination = Path(HF_HUB_CACHE) / "pixal3d-cuda" / "naf_release.pth"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temporary = destination.with_suffix(".download")
            try:
                with urllib.request.urlopen(NAF_URL, timeout=120) as response, temporary.open("wb") as target:
                    while block := response.read(1024 * 1024):
                        target.write(block)
                if sha256_file(temporary) != NAF_SHA256:
                    raise ValueError("Downloaded NAF checkpoint failed SHA-256 verification")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        if sha256_file(destination) != NAF_SHA256:
            raise ValueError("Cached NAF checkpoint failed SHA-256 verification")
        return destination

    def download(self, camera=True):
        self.config()
        self.dino()
        self.naf()
        for name in CHECKPOINTS:
            self.checkpoint(name)
        if camera:
            hf_hub_download(MOGE_REPO, "model.pt", revision=MOGE_REVISION)


def load_model(factory, path, device, dtype, ignored=()):
    with torch.device("meta"):
        model = factory()
    expected = model.state_dict()
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    if str(path).endswith(".safetensors"):
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            actual = set(checkpoint.keys())
            _check_keys(expected, actual, ignored)
            for key in expected:
                _assign(model, key, checkpoint.get_tensor(key), expected[key], parameters, buffers, device, dtype)
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        _check_keys(expected, set(checkpoint), ignored)
        for key in expected:
            _assign(model, key, checkpoint.pop(key), expected[key], parameters, buffers, device, dtype)
    return model.eval().requires_grad_(False)


def _check_keys(expected, actual, ignored):
    missing = set(expected) - actual
    extra = actual - set(expected) - set(ignored)
    if missing or extra:
        raise ValueError(f"Checkpoint mismatch: missing={sorted(missing)}, unexpected={sorted(extra)}")


def _assign(model, key, tensor, expected, parameters, buffers, device, dtype):
    if tensor.shape != expected.shape:
        raise ValueError(f"Checkpoint shape mismatch for {key}: {tuple(tensor.shape)} != {tuple(expected.shape)}")
    parent_name, _, attribute = key.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    tensor = tensor.to(device=device, dtype=dtype if tensor.is_floating_point() else tensor.dtype)
    if key in parameters:
        setattr(parent, attribute, torch.nn.Parameter(tensor, requires_grad=False))
    elif key in buffers:
        setattr(parent, attribute, tensor)
    else:
        raise ValueError(f"Unknown model tensor {key}")
