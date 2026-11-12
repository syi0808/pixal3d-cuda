"""Validated Blender/NeRF camera metadata for bounded CUDA multi-view input."""

from dataclasses import dataclass
import math

import numpy as np


VIEW_NAMES = ("azim000", "azim090", "azim180", "azim270")
VIEW_PATHS = tuple(f"view{index:02d}.png" for index in range(4))


def view_filename(index):
    return f"view{index:02d}.png"


def orbit_camera_pose(azimuth_degrees, elevation_degrees, distance):
    """Build a Z-up c2w camera looking at the origin, with zero roll."""
    values = (azimuth_degrees, elevation_degrees, distance)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in values):
        raise ValueError("Orbit angles and distance must be finite JSON numbers")
    if not -90 <= elevation_degrees <= 90 or distance <= 0:
        raise ValueError("Elevation must be between -90 and 90 degrees; distance must be positive")
    azimuth = math.radians(azimuth_degrees % 360)
    elevation = math.radians(elevation_degrees)
    sin_azimuth, cos_azimuth = math.sin(azimuth), math.cos(azimuth)
    sin_elevation, cos_elevation = math.sin(elevation), math.cos(elevation)
    return [
        [cos_azimuth, -sin_elevation * sin_azimuth, cos_elevation * sin_azimuth,
         distance * cos_elevation * sin_azimuth],
        [sin_azimuth, sin_elevation * cos_azimuth, -cos_elevation * cos_azimuth,
         -distance * cos_elevation * cos_azimuth],
        [0, cos_elevation, sin_elevation, distance * sin_elevation],
        [0, 0, 0, 1],
    ]


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
    if not isinstance(frames, list) or not frames:
        raise ValueError("transforms.json must contain a nonempty frames list")
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


def uploaded_camera_metadata(metadata, view_count=None):
    """Map each image to one explicit camera, ignoring client filesystem paths."""
    if not isinstance(metadata, dict) or not isinstance(metadata.get("frames"), list) or not metadata["frames"]:
        raise ValueError("Camera metadata must contain a nonempty frames list")
    source_frames = metadata["frames"]
    view_count = len(source_frames) if view_count is None else view_count
    if type(view_count) is not int or view_count < 1 or len(source_frames) != view_count:
        raise ValueError("Provide exactly one camera frame for each uploaded image")
    if any(not isinstance(frame, dict) for frame in source_frames):
        raise ValueError("Each frame must be an object")
    indexed = ["image_index" in frame for frame in source_frames]
    if any(indexed) and not all(indexed):
        raise ValueError("Use image_index on every frame, or use upload order for every frame")
    frames = [None] * view_count
    for position, frame in enumerate(source_frames):
        index = frame.get("image_index", position)
        if type(index) is not int or not 0 <= index < view_count or frames[index] is not None:
            raise ValueError("image_index must map each uploaded image exactly once")
        normalized = {**frame, "file_path": view_filename(index)}
        orbit_fields = ("azimuth_degrees", "elevation_degrees", "distance")
        if "transform_matrix" in frame:
            if any(name in frame for name in orbit_fields):
                raise ValueError("Choose a camera matrix or orbit parameters, not both")
        else:
            if any(name not in frame for name in (*orbit_fields, "fov_degrees")):
                raise ValueError("Each image needs a camera matrix or azimuth/elevation/distance/FOV")
            normalized["transform_matrix"] = orbit_camera_pose(*(frame[name] for name in orbit_fields))
        if "fov_degrees" in frame:
            fov = frame["fov_degrees"]
            if (isinstance(fov, bool) or not isinstance(fov, (int, float)) or not math.isfinite(fov)
                    or not 0 < fov < 180 or "camera_angle_x" in frame):
                raise ValueError("Use one finite FOV in degrees (0-180) or camera_angle_x in radians")
            normalized["camera_angle_x"] = math.radians(fov)
        frames[index] = normalized
    rig = parse_camera_metadata({**metadata, "frames": frames})
    return {"mesh_scale": rig.mesh_scale, "frames": [
        {"image_index": index, "file_path": view_filename(index), "name": f"view-{index}",
         "camera_angle_x": float(rig.camera_angle_x[0, index]),
         "transform_matrix": rig.transform_matrix[0, index].tolist()}
        for index in range(view_count)]}
