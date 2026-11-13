"""Image/camera manifests only; conditioning is delegated to official Pixal3D."""

from dataclasses import dataclass
import json
from pathlib import Path
import warnings

import numpy as np

from .cameras import CameraRig, parse_camera_metadata, uploaded_camera_metadata
from .weights import sha256_file


@dataclass(frozen=True)
class ViewManifest:
    paths: tuple[Path, ...]
    rig: CameraRig
    identity: dict


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
