import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from pixalcuda.cameras import orbit_camera_metadata, parse_camera_metadata
from pixalcuda.conditioning import build_conditioning, build_multiview_conditioning, project_grid
from pixalcuda.multiview import encode_views, prepare_views, read_views, relative_cameras
from pixalcuda.weights import CHECKPOINTS, MULTIVIEW_REVISION, ModelStore


@pytest.fixture
def views_directory(tmp_path):
    metadata = orbit_camera_metadata()
    for frame in metadata["frames"]:
        Image.new("RGBA", (23, 31), (255, 0, 0, 128)).save(tmp_path / frame["file_path"])
    (tmp_path / "transforms.json").write_text(json.dumps(metadata))
    return tmp_path


def test_premultiplication_preserves_calibrated_framing(views_directory):
    views = prepare_views(read_views(views_directory), remove_background=False)
    assert [image.size for image in views.images] == [(23, 31)] * 4
    tensor = views.tensor(0, 32, "cpu")
    assert tensor.shape == (1, 3, 32, 32)
    torch.testing.assert_close(tensor[:, 0], torch.full((1, 32, 32), 128 / 255))
    assert torch.count_nonzero(tensor[:, 1:]) == 0


def test_backbone_cache_is_per_image_not_camera(views_directory):
    views = prepare_views(read_views(views_directory), remove_background=False)
    calls = []
    def encode(image):
        calls.append(image.shape)
        return torch.zeros((1, 9, 8))
    encoded = encode_views(encode, views, 32, "cpu")
    assert len(calls) == 1 and len(encoded) == 4
    assert encoded[0] is encoded[3]


def test_relative_pose_and_projection_against_independent_numpy_reference():
    rig = parse_camera_metadata(orbit_camera_metadata())
    cameras = relative_cameras(rig)
    coordinates = torch.tensor([[0, 0, 0, 0], [0, 2, 1, 1], [0, 1, 2, 0]])
    for index in range(4):
        camera = {"camera_angle_x": float(rig.camera_angle_x[0, index]), "distance": float(rig.distance[0, index]),
                  "mesh_scale": rig.mesh_scale, "transform_matrix": cameras[index]}
        actual = project_grid(coordinates, 3, 512, camera).numpy()
        spatial = coordinates[:, 1:].numpy().astype(np.float64) - 1
        world = spatial @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]) / 2
        points = np.column_stack((world, np.ones(len(world)))) @ np.linalg.inv(cameras[index].astype(np.float64)).T
        focal = 256 / np.tan(camera["camera_angle_x"] / 2)
        pixels = np.column_stack((focal * points[:, 0] / (-points[:, 2] + 1e-8) + 256,
                                  -focal * points[:, 1] / (-points[:, 2] + 1e-8) + 256))
        np.testing.assert_allclose(actual, (pixels + .5) / 256 - 1, atol=1e-6, rtol=1e-5)
    simple = {key: value for key, value in camera.items() if key != "transform_matrix"}
    simple["distance"] = float(rig.distance[0, 0])
    explicit = {**simple, "transform_matrix": cameras[0]}
    torch.testing.assert_close(project_grid(coordinates, 3, 512, simple), project_grid(coordinates, 3, 512, explicit))


def test_relative_rig_uses_view_zero_frame():
    metadata = orbit_camera_metadata()
    rotation = np.array([[0, -1, 0, .5], [1, 0, 0, .2], [0, 0, 1, .1], [0, 0, 0, 1]])
    for frame in metadata["frames"]:
        frame["transform_matrix"] = (rotation @ np.array(frame["transform_matrix"])).tolist()
    rig = parse_camera_metadata(metadata)
    front = np.array([[1, 0, 0, 0], [0, 0, -1, -rig.distance[0, 0]], [0, 1, 0, 0], [0, 0, 0, 1]])
    expected = np.stack([front @ np.linalg.inv(rig.transform_matrix[0, 0].astype(np.float64)) @ pose
                         for pose in rig.transform_matrix[0]])
    np.testing.assert_allclose(relative_cameras(rig), expected, atol=1e-6)


def test_average_fusion_runs_one_view_at_a_time_and_preserves_texture_target(views_directory):
    views = prepare_views(read_views(views_directory), remove_background=False)
    encoded = tuple(torch.full((1, 9, 8), float(index)) for index in range(4))
    coordinates = torch.tensor([[0, 1, 1, 1], [0, 2, 1, 1]])
    calls = []
    class Projector:
        def project(self, image, features, grid, target):
            assert image.shape[0] == 1 and grid.shape == (2, 2)
            calls.append(target)
            return torch.full((len(grid), features.shape[1]), float(target))
    projector = Projector()
    actual = build_multiview_conditioning(encoded, views, coordinates, 3, 32, "cpu", projector, naf_target=1024)
    assert calls == [1024] * 4
    assert actual["global"].shape == (1, 5, 8) and actual["proj"].shape == (1, 2, 16)
    torch.testing.assert_close(actual["global"], torch.full((1, 5, 8), 1.5))
    torch.testing.assert_close(actual["proj"][..., :8], torch.full((1, 2, 8), 1.5))
    torch.testing.assert_close(actual["proj"][..., 8:], torch.full((1, 2, 8), 1024.))
    single = build_multiview_conditioning(encoded[:1], prepare_views(read_views(views_directory, 1), False),
                                         coordinates, 3, 32, "cpu")
    expected = build_conditioning(encoded[0], views.tensor(0, 32, "cpu"), coordinates, 3, views.camera(0))
    torch.testing.assert_close(single["proj"], expected["proj"])


@pytest.mark.parametrize("bad_path", ["../outside.png", "/tmp/outside.png"])
def test_view_paths_cannot_escape_root(views_directory, bad_path):
    metadata = orbit_camera_metadata()
    metadata["frames"][0]["file_path"] = bad_path
    (views_directory / "transforms.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="within"):
        read_views(views_directory)


@pytest.mark.parametrize("field,value", [("camera_angle_x", 0), ("camera_angle_x", 1e-300),
                                         ("mesh_scale", 0), ("mesh_scale", 1e-300)])
def test_camera_metadata_rejects_invalid_or_overflowing_projection(field, value):
    metadata = orbit_camera_metadata()
    metadata[field] = value
    with pytest.raises(ValueError):
        parse_camera_metadata(metadata)


def test_identity_covers_all_views_and_pose_order(views_directory):
    original = read_views(views_directory).identity
    Image.new("RGB", (23, 31), "blue").save(views_directory / "view03.png")
    assert read_views(views_directory).identity != original
    metadata = orbit_camera_metadata()
    metadata["frames"][1:3] = reversed(metadata["frames"][1:3])
    (views_directory / "transforms.json").write_text(json.dumps(metadata))
    changed = read_views(views_directory).identity
    assert changed["frames"][1]["transform_matrix"] != original["frames"][1]["transform_matrix"]


def test_dedicated_mv_flows_are_pinned_and_never_fall_back(monkeypatch):
    requests = []
    def download(repo, filename, revision):
        requests.append((repo, filename, revision))
        return filename
    monkeypatch.setattr("pixalcuda.weights.hf_hub_download", download)
    store = ModelStore(multiview=True)
    for name in CHECKPOINTS:
        path = store.checkpoint(name)
        assert path.name.endswith("_mv.safetensors") == (name in ("structure", "shape_lr", "shape_hr", "texture"))
    assert {record[2] for record in requests} == {MULTIVIEW_REVISION}


def test_missing_local_mv_checkpoint_does_not_substitute_single_view(tmp_path):
    (tmp_path / "ckpts").mkdir()
    (tmp_path / "ckpts" / (CHECKPOINTS["structure"] + ".safetensors")).touch()
    with pytest.raises(FileNotFoundError, match="_mv"):
        ModelStore(tmp_path, multiview=True).checkpoint("structure")


@pytest.mark.parametrize("arguments", [
    ["--views-dir", "views", "--camera", "fixed"],
    ["--views-dir", "views", "--fov-degrees", "20"],
    ["--views-dir", "views", "--num-views", "0"],
    ["--image", "image.png", "--num-views", "4"],
    ["--image", "image.png", "--views-dir", "views"],
])
def test_cli_rejects_invalid_modes_before_loading_any_models(arguments):
    from pixalcuda.cli import main

    with pytest.raises(SystemExit) as error:
        main(["generate", *arguments])
    assert error.value.code == 2
