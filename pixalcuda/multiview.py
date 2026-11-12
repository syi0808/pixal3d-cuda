"""Posed image input preserving calibration, framing, and alpha."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from PIL import Image

from .cameras import CameraRig, parse_camera_metadata, uploaded_camera_metadata
from .weights import sha256_file


def relative_cameras(rig):
    cameras = rig.transform_matrix[0]
    front = np.array([[1, 0, 0, 0], [0, 0, -1, -rig.distance[0, 0]],
                      [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
    return front[None] @ (np.linalg.inv(cameras[:1]) @ cameras)


@dataclass(frozen=True)
class ViewManifest:
    paths: tuple[Path, ...]
    rig: CameraRig
    identity: dict


@dataclass(frozen=True)
class PreparedViews:
    images: tuple[Image.Image, ...]
    fingerprints: tuple[str, ...]
    rig: CameraRig
    transforms: np.ndarray

    @property
    def main_camera(self):
        return {"camera_angle_x": float(self.rig.camera_angle_x[0, 0]),
                "distance": float(self.rig.distance[0, 0]), "mesh_scale": self.rig.mesh_scale}

    def camera(self, index):
        return {"camera_angle_x": float(self.rig.camera_angle_x[0, index]),
                "distance": float(self.rig.distance[0, index]), "mesh_scale": self.rig.mesh_scale,
                "transform_matrix": self.transforms[index]}

    def tensor(self, index, size, device):
        import torch

        resized = self.images[index].resize((size, size), Image.Resampling.LANCZOS)
        array = np.asarray(resized, dtype=np.float32) / 255
        array = array[..., :3] * array[..., 3:4]
        return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).to(device)


def read_views(directory, num_views=None):
    root = Path(directory).expanduser().resolve()
    metadata = json.loads((root / "transforms.json").read_text())
    normalized = uploaded_camera_metadata(metadata)
    rig = parse_camera_metadata(normalized, num_views)
    ordered = [None] * len(rig.frames)
    for position, frame in enumerate(metadata["frames"]):
        index = frame.get("image_index", position)
        if index >= len(ordered):
            continue
        name = frame.get("file_path")
        if not isinstance(name, str) or not name:
            raise ValueError("Directory frames need file_path; use mapped images for separate camera metadata")
        path = (root / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(root):
            raise ValueError("View file_path must stay within views_dir")
        ordered[index] = (path, name)
    return view_manifest(tuple(item[0] for item in ordered), rig, tuple(item[1] for item in ordered))


def read_mapped_views(images, metadata, num_views=None):
    if isinstance(images, (str, bytes, Path)) or images is None:
        raise ValueError("images must be a nonempty sequence of image paths")
    paths = tuple(Path(path).expanduser().resolve() for path in images)
    normalized = uploaded_camera_metadata(metadata, len(paths))
    rig = parse_camera_metadata(normalized, num_views)
    return view_manifest(paths[:len(rig.frames)], rig, tuple(frame["file_path"] for frame in rig.frames))


def view_manifest(paths, rig, names):
    records = []
    for index, (path, name) in enumerate(zip(paths, names)):
        if not path.is_file():
            raise FileNotFoundError(f"View image not found: {path}")
        records.append({"image_index": index, "file_path": name, "sha256": sha256_file(path),
                        "camera_angle_x": float(rig.camera_angle_x[0, index]),
                        "transform_matrix": rig.transform_matrix[0, index].tolist()})
    canonical = np.array([[1, 0, 0, 0], [0, 0, -1, -rig.distance[0, 0]],
                          [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
    if not np.allclose(rig.transform_matrix[0, 0], canonical, atol=1e-4, rtol=0):
        warnings.warn("View zero is not canonical front; the output uses its frame.", UserWarning, stacklevel=2)
    return ViewManifest(tuple(paths), rig, {"frames": records, "mesh_scale": rig.mesh_scale})


def prepare_views(manifest, remove_background=True):
    images, fingerprints, session = [], [], None
    for path in manifest.paths:
        with Image.open(path) as source:
            image = source.convert("RGBA")
        if remove_background and (np.asarray(image.getchannel("A")) == 255).all():
            try:
                from rembg import new_session, remove
            except ImportError as error:
                raise RuntimeError("Install pixal3d-cuda[preprocess], or use --no-rembg with masked views") from error
            session = session or new_session("u2net", providers=["CPUExecutionProvider"])
            size = image.size
            image = remove(image.convert("RGB"), session=session).convert("RGBA")
            if image.size != size:
                raise ValueError("Background removal must preserve calibrated view framing")
        if not np.any(np.asarray(image.getchannel("A")) > 127):
            raise ValueError(f"View has no foreground pixels: {path.name}")
        images.append(image)
        fingerprints.append(hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest())
    return PreparedViews(tuple(images), tuple(fingerprints), manifest.rig, relative_cameras(manifest.rig))


def encode_views(model, views, size, device):
    cache, encoded = {}, []
    for index, fingerprint in enumerate(views.fingerprints):
        if fingerprint not in cache:
            cache[fingerprint] = model(views.tensor(index, size, device)).cpu()
        encoded.append(cache[fingerprint])
    return tuple(encoded)
