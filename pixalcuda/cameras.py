"""Validated Blender/NeRF camera metadata for bounded CUDA multi-view input."""

from dataclasses import dataclass
import math

import numpy as np


VIEW_NAMES = ("azim000", "azim090", "azim180", "azim270")
VIEW_PATHS = tuple(f"view{index:02d}.png" for index in range(4))


def orbit_camera_metadata():
    """Return the upstream four-view example's assumed level, 20-degree-FOV rig."""
    distance = 3.1192049980163574
    poses = (
        [[1, 0, 0, 0], [0, 0, -1, -distance], [0, 1, 0, 0], [0, 0, 0, 1]],
        [[0, 0, 1, distance], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        [[-1, 0, 0, 0], [0, 0, 1, distance], [0, 1, 0, 0], [0, 0, 0, 1]],
        [[0, 0, -1, -distance], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    )
    return {"camera_angle_x": math.radians(20), "mesh_scale": 1.0,
            "frames": [{"file_path": path, "name": name, "transform_matrix": pose}
                       for path, name, pose in zip(VIEW_PATHS, VIEW_NAMES, poses)]}


@dataclass(frozen=True)
class CameraRig:
    frames: tuple[dict, ...]
    camera_angle_x: np.ndarray
    distance: np.ndarray
    transform_matrix: np.ndarray
    mesh_scale: float


def parse_camera_metadata(metadata, num_views=None):
    if not isinstance(metadata, dict):
        raise ValueError("transforms.json must be an object")
    frames = metadata.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= 4:
        raise ValueError("transforms.json must contain between one and four frames")
    if num_views is not None:
        if isinstance(num_views, bool) or not isinstance(num_views, int) or not 1 <= num_views <= len(frames):
            raise ValueError("num_views must be between 1 and the number of frames")
        frames = frames[:num_views]
    mesh_scale = float(metadata.get("mesh_scale", 1.0))
    if not math.isfinite(mesh_scale) or mesh_scale <= 0:
        raise ValueError("mesh_scale must be finite and positive")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        inverse_scale = np.float32(1.0) / np.float32(mesh_scale)
    if not np.isfinite(inverse_scale) or inverse_scale <= 0:
        raise ValueError("mesh_scale must have a finite positive float32 reciprocal")
    angles, cameras = [], []
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Each frame must be an object")
        if not isinstance(frame.get("file_path"), str) or not frame["file_path"]:
            raise ValueError("Each frame must have a nonempty file_path")
        angle = float(frame.get("camera_angle_x", metadata.get("camera_angle_x", float("nan"))))
        if not math.isfinite(angle) or not 0 < angle < math.pi:
            raise ValueError("Each view needs camera_angle_x in radians between 0 and pi")
        camera = np.asarray(frame.get("transform_matrix"), dtype=np.float32)
        if camera.shape != (4, 4) or not np.isfinite(camera).all():
            raise ValueError("Each transform_matrix must be a finite 4x4 c2w matrix")
        if (not np.allclose(camera[3], [0, 0, 0, 1], atol=1e-5)
                or not np.allclose(camera[:3, :3].T @ camera[:3, :3], np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(camera[:3, :3]), 1.0, atol=1e-4)):
            raise ValueError("Each transform_matrix must be a rigid, right-handed c2w pose")
        if np.linalg.norm(camera[:3, 3].astype(np.float64)) <= 0:
            raise ValueError("Camera distance must be positive")
        angles.append(angle)
        cameras.append(camera)
    cameras = np.stack(cameras)[None]
    distances = np.linalg.norm(cameras[..., :3, 3].astype(np.float64), axis=-1).astype(np.float32)
    if not np.isfinite(distances).all():
        raise ValueError("Camera distances must be finite")
    angles = np.array([angles], dtype=np.float32)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        focal_pixels = (np.float32(16) / np.tan(angles / np.float32(2))) * np.float32(1024) / np.float32(32)
    if not np.isfinite(focal_pixels).all() or np.any(focal_pixels <= 0):
        raise ValueError("Camera FOV must produce finite positive float32 focal lengths")
    return CameraRig(tuple(frames), angles, distances, cameras, mesh_scale)


def uploaded_camera_metadata(metadata):
    """Map four ordered camera frames to fixed server filenames, ignoring user paths."""
    if not isinstance(metadata, dict) or not isinstance(metadata.get("frames"), list) or len(metadata["frames"]) != 4:
        raise ValueError("Multi-view camera metadata must contain exactly four frames in upload order")
    frames = []
    for index, frame in enumerate(metadata["frames"]):
        if not isinstance(frame, dict):
            raise ValueError("Each frame must be an object")
        frames.append({**frame, "file_path": VIEW_PATHS[index]})
    rig = parse_camera_metadata({**metadata, "frames": frames})
    return {"mesh_scale": rig.mesh_scale, "frames": [
        {"file_path": VIEW_PATHS[index], "name": VIEW_NAMES[index],
         "camera_angle_x": float(rig.camera_angle_x[0, index]),
         "transform_matrix": rig.transform_matrix[0, index].tolist()}
        for index in range(4)]}
