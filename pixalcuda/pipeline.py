"""Official Pixal3D orchestration; no MLX-derived neural operators or exporter."""

import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

from .runtime import atomic_json
from .upstream_sources import lock, source_root, verify_source
from .upstream_weights import UpstreamStore
from .weights import sha256_file


def validate_options(geometry_resolution=1024, texture_size=4096, target_faces=1000000):
    if type(geometry_resolution) is not int or geometry_resolution not in (1024, 1536):
        raise ValueError("Official Pixal3D supports 1024/1536 geometry, not a 512 cascade")
    if type(texture_size) is not int or texture_size not in (512, 1024, 2048, 4096):
        raise ValueError("Texture size must be 512, 1024, 2048 or 4096")
    if type(target_faces) is not int or not 4 <= target_faces <= 1000000:
        raise ValueError("Target faces must be between 4 and 1000000")


def validate_pbr_glb(path):
    path = Path(path)
    with path.open("rb") as source:
        header = source.read(20)
        if len(header) != 20:
            raise ValueError("Incomplete GLB header")
        magic, version, size, length, kind = struct.unpack("<4sIIII", header)
        if magic != b"glTF" or version != 2 or size != path.stat().st_size or kind != 0x4E4F534A or length > 16 * 1024**2:
            raise ValueError("Invalid GLB v2 JSON chunk")
        document = json.loads(source.read(length))
    material = document["materials"][0]["pbrMetallicRoughness"]
    for name in ("baseColorTexture", "metallicRoughnessTexture"):
        texture = document["textures"][material[name]["index"]]
        index = texture.get("source", texture.get("extensions", {}).get("EXT_texture_webp", {}).get("source"))
        if index is None or "bufferView" not in document["images"][index]:
            raise ValueError(f"Missing embedded PBR image: {name}")
    return {"embedded_pbr": True, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


class Pipeline:
    def __init__(self, device="cuda", *, upstream_root=None, store=None, low_vram=True,
                 attention_backend="sdpa", max_tokens=49152, timeout_seconds=1800, save_reference=False):
        if device != "cuda":
            raise ValueError("The official runtime requires CUDA")
        if attention_backend not in ("sdpa", "flash_attn"):
            raise ValueError("Choose an explicit supported upstream attention backend")
        if type(low_vram) is not bool or type(save_reference) is not bool or type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("low_vram must be boolean and max_tokens must be positive")
        if type(timeout_seconds) is not int or timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")
        self.root = source_root(upstream_root)
        self.store = store
        self.options = {"low_vram": low_vram, "attention_backend": attention_backend, "max_tokens": max_tokens, "save_reference": save_reference}
        self.timeout_seconds = timeout_seconds

    def generate(self, image_path, output_dir, seed=42, texture_size=4096, target_faces=1000000,
                 camera="auto", fov_degrees=49.13, remove_background=True, resume=False,
                 *, views_dir=None, num_views=None, images=None, camera_metadata=None, geometry_resolution=1024):
        validate_options(geometry_resolution, texture_size, target_faces)
        if sum((bool(image_path), bool(views_dir), images is not None)) != 1:
            raise ValueError("Supply one image, a views directory, or mapped images")
        multiview = bool(views_dir) or images is not None
        if camera not in ("auto", "fixed") or not math.isfinite(fov_degrees) or not 1 <= fov_degrees <= 150:
            raise ValueError("Invalid camera/FOV")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Seed must be a nonnegative 63-bit integer")
        if type(remove_background) is not bool:
            raise ValueError("remove_background must be boolean")
        if multiview and (camera != "auto" or fov_degrees != 49.13):
            raise ValueError("Mapped cameras supply their own FOV")
        if not multiview and (camera_metadata is not None or num_views is not None):
            raise ValueError("Camera metadata and num_views require mapped inputs")
        if num_views is not None and (type(num_views) is not int or num_views < 1):
            raise ValueError("num_views must be positive")
        if multiview:
            from .multiview import read_mapped_views, read_views
            from .cameras import uploaded_camera_metadata
            manifest = read_mapped_views(images, camera_metadata, num_views) if images is not None else read_views(views_dir, num_views)
            paths = manifest.paths
            camera_metadata = uploaded_camera_metadata({"mesh_scale": manifest.rig.mesh_scale, "frames": [
                {"image_index": index, "camera_angle_x": float(manifest.rig.camera_angle_x[0, index]),
                 "transform_matrix": manifest.rig.transform_matrix[0, index].tolist()} for index in range(len(paths))]}, len(paths))
        else:
            paths = [Path(image_path).expanduser().resolve()]
        local_weights = {}
        if self.store is not None and getattr(self.store, "directory", None) is not None:
            local_weights = {str(path.relative_to(self.store.directory)): sha256_file(path)
                             for path in sorted(self.store.directory.rglob("*")) if path.is_file() and path.suffix in (".json", ".safetensors")}
        identity = {"schema": "pixalcuda.upstream.v1", "upstream_lock": lock(),
                    "runtime_sources": {path.name: sha256_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
                    "inputs": [{"path": str(path), "sha256": sha256_file(path)} for path in paths],
                    "local_weight_hashes": local_weights,
                    "cameras": camera_metadata, "multiview": multiview,
                    "geometry_resolution": geometry_resolution, "texture_size": texture_size, "target_faces": target_faces,
                    "seed": seed, "camera": camera, "fov_degrees": fov_degrees, "remove_background": remove_background,
                    **self.options}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        request_path = directory / "upstream-request.json"
        if request_path.exists():
            previous = json.loads(request_path.read_text())
            if previous["fingerprint"] != fingerprint:
                raise ValueError("Input/source/settings changed; use a new output directory")
            report_path = directory / "report.json"
            if resume and report_path.exists() and json.loads(report_path.read_text()).get("status") == "complete":
                validation = validate_pbr_glb(directory / "result.glb")
                if validation != json.loads(report_path.read_text())["validation"]:
                    raise ValueError("Completed GLB changed since generation")
                return directory / "result.glb"
            raise ValueError("Native stage resume is not implemented; use a new output directory, or --resume for a completed result")
        for name in ("pixal3d", "naf"):
            verify_source(name, self.root)
        preflight()
        store = self.store or UpstreamStore(multiview=multiview)
        if store.multiview != multiview:
            raise ValueError("Dedicated MV weights are required for mapped inputs")
        bundle = store.resolve(camera=camera == "auto")
        request = {"identity": identity, "fingerprint": fingerprint, "bundle": bundle, "upstream_root": str(self.root)}
        atomic_json(request_path, request)
        atomic_json(directory / "report.json", {"schema": identity["schema"], "identity": identity, "status": "starting", "stages": []})
        environment = {**os.environ, "PIXAL_UPSTREAM_ROOT": str(self.root), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "ATTN_BACKEND": self.options["attention_backend"], "SPARSE_ATTN_BACKEND": self.options["attention_backend"],
                       "SPARSE_CONV_BACKEND": "flex_gemm"}
        environment["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1]), str(self.root / "pixal3d"), environment.get("PYTHONPATH", "")])
        try:
            with (directory / "upstream.log").open("w") as log:
                subprocess.run([sys.executable, "-m", "pixalcuda.upstream_worker", str(request_path)], env=environment,
                               stdout=log, stderr=subprocess.STDOUT, timeout=self.timeout_seconds, check=True)
            report = json.loads((directory / "report.json").read_text())
            validation = validate_pbr_glb(directory / "result.glb")
            if report.get("status") != "complete" or report.get("validation") != validation:
                raise RuntimeError("Native worker did not produce a matching complete PBR report")
        except BaseException as error:
            report = json.loads((directory / "report.json").read_text())
            report.update(status="failed", worker_error=f"{type(error).__name__}: {error}", log=str(directory / "upstream.log"))
            atomic_json(directory / "report.json", report)
            raise RuntimeError(f"Official Pixal3D failed; inspect {directory / 'upstream.log'} and report.json") from error
        return directory / "result.glb"


def preflight():
    result = subprocess.run([sys.executable, "-m", "pixalcuda.upstream_worker", "--check-hardware"], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Official CUDA preflight failed before model downloads: {result.stderr.strip()}")
