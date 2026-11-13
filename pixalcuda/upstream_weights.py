"""Resolve official model distributions to pinned local snapshots."""

import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from .runtime import atomic_json
from .upstream_sources import lock
from .weights import CHECKPOINTS, MULTIVIEW_CHECKPOINTS, naf_weights


class UpstreamStore:
    def __init__(self, directory=None, multiview=False):
        self.directory = Path(directory).expanduser().resolve() if directory else None
        self.multiview = multiview
        self.revision = lock()["models"]["multiview" if multiview else "single"]["revision"]

    def resolve(self, camera=True):
        models = lock()["models"]
        mode = "multiview" if self.multiview else "single"
        offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        config_file = "pipeline_mv.json" if self.multiview else "pipeline.json"
        names = MULTIVIEW_CHECKPOINTS if self.multiview else CHECKPOINTS
        patterns = [config_file, *[f"ckpts/{name}.{suffix}" for name in names.values() for suffix in ("json", "safetensors")]]
        if self.directory is None:
            model = models[mode]
            snapshot = Path(snapshot_download(model["repo"], revision=model["revision"], allow_patterns=patterns, local_files_only=offline))
        else:
            snapshot = self.directory
        for filename in patterns:
            if not (snapshot / filename).is_file():
                raise FileNotFoundError(f"Official model snapshot is incomplete: {snapshot / filename}")
        resolved = {"pixal": str(snapshot), "config_file": config_file, "revision": self.revision}
        for name in ("dino", "rembg"):
            model = models[name]
            resolved[name] = snapshot_download(model["repo"], revision=model["revision"],
                                               allow_patterns=["*.json", "*.py", "model.safetensors"], local_files_only=offline)
        resolved["naf_weights"] = str(naf_weights())
        if camera and not self.multiview:
            model = models["moge"]
            resolved["moge"] = hf_hub_download(model["repo"], "model.pt", revision=model["revision"], local_files_only=offline)
        return resolved


def local_pipeline_config(bundle, destination):
    source = Path(bundle["pixal"])
    config = json.loads((source / bundle["config_file"]).read_text())
    config["args"]["rembg_model"]["args"]["model_name"] = bundle["rembg"]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "ckpts").symlink_to(source / "ckpts", target_is_directory=True)
    atomic_json(destination / bundle["config_file"], config)
    return destination
