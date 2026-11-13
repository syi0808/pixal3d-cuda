"""Official checkpoint names, file digests and verified NAF downloads."""

import hashlib
from pathlib import Path
import urllib.request

from .upstream_sources import lock

PIXAL_REPO = lock()["models"]["single"]["repo"]
PIXAL_REVISION = lock()["models"]["single"]["revision"]
MULTIVIEW_REVISION = lock()["models"]["multiview"]["revision"]


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


MULTIVIEW_CHECKPOINTS = {name: filename + ("_mv" if name in ("structure", "shape_lr", "shape_hr", "texture") else "")
                         for name, filename in CHECKPOINTS.items()}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def naf_weights():
    from huggingface_hub.constants import HF_HUB_CACHE, HF_HUB_OFFLINE
    destination = Path(HF_HUB_CACHE) / "pixal3d-cuda" / "naf_release.pth"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        if HF_HUB_OFFLINE:
            raise FileNotFoundError("NAF is not cached; prefetch weights on CPU before offline GPU inference")
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
