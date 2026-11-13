import copy
import json

import numpy as np
from PIL import Image
import pytest

from pixalcuda.cameras import orbit_camera_metadata, orbit_camera_pose, uploaded_camera_metadata, view_filename
from pixalcuda.multiview import read_mapped_views, read_views


def camera_mapping(count):
    return {"mesh_scale": 1.0, "frames": [
        {"image_index": index, "azimuth_degrees": index * 360 / count, "elevation_degrees": 0,
         "distance": 3.119205, "fov_degrees": 20} for index in range(count)]}


@pytest.mark.parametrize("count", [1, 2, 3, 5, 16, 17])
def test_variable_view_counts_are_mapped_by_index_not_frame_array_order(tmp_path, count):
    images = []
    for index in range(count):
        directory = tmp_path / str(index)
        directory.mkdir()
        path = directory / "same-name.png"
        Image.new("RGBA", (12, 20), (index, 80, 100, 180)).save(path)
        images.append(path)
    metadata = camera_mapping(count)
    metadata["frames"].reverse()
    for frame in metadata["frames"]:
        frame["file_path"] = "../../never-read.png"
    manifest = read_mapped_views(images, metadata)
    assert manifest.paths == tuple(images)
    assert [frame["image_index"] for frame in manifest.rig.frames] == list(range(count))
    for index in range(count):
        np.testing.assert_allclose(manifest.rig.transform_matrix[0, index],
                                   orbit_camera_pose(index * 360 / count, 0, 3.119205), atol=1e-6)
    for index, path in enumerate(manifest.paths):
        with Image.open(path) as source:
            assert source.getpixel((0, 0))[0] == index and source.size == (12, 20)
    reordered = copy.deepcopy(metadata)
    reordered["frames"].reverse()
    assert read_mapped_views(images, reordered).identity == manifest.identity
    if count > 1:
        assert read_mapped_views(list(reversed(images)), metadata).identity != manifest.identity
        assert len(read_mapped_views(images, metadata, 1).paths) == 1


def test_directory_mapping_preserves_file_path_camera_pairs_and_prefix(tmp_path):
    metadata = camera_mapping(5)
    for index, frame in enumerate(metadata["frames"]):
        frame["file_path"] = f"photo-{index}.png"
        Image.new("RGBA", (16, 16), (index, 0, 0, 180)).save(tmp_path / frame["file_path"])
    metadata["frames"].reverse()
    (tmp_path / "transforms.json").write_text(json.dumps(metadata))
    manifest = read_views(tmp_path, 3)
    assert [path.name for path in manifest.paths] == [f"photo-{index}.png" for index in range(3)]
    assert manifest.identity["frames"][0]["file_path"] == "photo-0.png"


@pytest.mark.parametrize("indices", [[0, 0, 2], [0, 1, 3], [0, 1, -1], [0, 1, True],
                                     [0, 1, "2"], [0, 1, 2.0], [0, 1, None]])
def test_ambiguous_or_incomplete_image_camera_mapping_is_rejected(indices):
    metadata = camera_mapping(3)
    for frame, index in zip(metadata["frames"], indices):
        if index is None:
            del frame["image_index"]
        else:
            frame["image_index"] = index
    with pytest.raises(ValueError, match="image_index"):
        uploaded_camera_metadata(metadata, 3)


@pytest.mark.parametrize("azimuth,elevation", [(0, 0), (90, 0), (180, 0), (270, 0), (47, 18), (203, -12), (0, 90)])
def test_orbit_camera_is_right_handed_and_looks_at_origin(azimuth, elevation):
    pose = np.array(orbit_camera_pose(azimuth, elevation, 3.0))
    np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-7)
    np.testing.assert_allclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-7)
    np.testing.assert_allclose(pose[:3, 3], pose[:3, 2] * 3, atol=1e-7)


def test_matrices_and_orbits_can_be_mixed_but_not_on_the_same_frame():
    metadata = camera_mapping(2)
    front = orbit_camera_metadata()["frames"][0]
    metadata["frames"][0] = {**front, "image_index": 0, "camera_angle_x": .5}
    result = uploaded_camera_metadata(metadata, 2)
    assert result["frames"][0]["camera_angle_x"] == .5
    assert result["frames"][1]["file_path"] == view_filename(1)
    metadata["frames"][0]["azimuth_degrees"] = 0
    with pytest.raises(ValueError, match="not both"):
        uploaded_camera_metadata(metadata, 2)


@pytest.mark.parametrize("field,value", [("azimuth_degrees", True), ("elevation_degrees", 91),
                                         ("distance", 0), ("distance", float("inf")),
                                         ("fov_degrees", 180), ("fov_degrees", "20")])
def test_invalid_orbit_numbers_are_not_silently_coerced(field, value):
    metadata = camera_mapping(1)
    metadata["frames"][0][field] = value
    with pytest.raises(ValueError):
        uploaded_camera_metadata(metadata, 1)


def test_missing_cameras_count_mismatches_and_fov_ambiguity_fail():
    for metadata, count in (({}, 1), ({"frames": []}, 0), (camera_mapping(2), 1), (camera_mapping(1), True)):
        with pytest.raises(ValueError):
            uploaded_camera_metadata(metadata, count)
    metadata = camera_mapping(1)
    del metadata["frames"][0]["distance"]
    with pytest.raises(ValueError):
        uploaded_camera_metadata(metadata, 1)
    metadata = camera_mapping(1)
    metadata["frames"][0]["camera_angle_x"] = .5
    with pytest.raises(ValueError):
        uploaded_camera_metadata(metadata, 1)


def test_array_order_mapping_without_an_implicit_rig():
    metadata = camera_mapping(3)
    for frame in metadata["frames"]:
        del frame["image_index"]
    assert len(uploaded_camera_metadata(metadata)["frames"]) == 3
    with pytest.raises(ValueError, match="camera matrix or"):
        uploaded_camera_metadata({"frames": [{"file_path": "front.png"}]}, 1)


def test_cli_passes_the_explicit_image_list_and_camera_object(tmp_path, monkeypatch):
    from pixalcuda.cli import main
    from pixalcuda.pipeline import Pipeline

    images = [str(tmp_path / f"image-{index}.png") for index in range(3)]
    metadata = camera_mapping(3)
    transforms = tmp_path / "cameras.json"
    transforms.write_text(json.dumps(metadata))
    calls = []
    def generate(pipeline, image_path, *args, **options):
        calls.append((pipeline.store.multiview, image_path, options))
        return tmp_path / "not-generated.glb"
    monkeypatch.setattr(Pipeline, "generate", generate)
    main(["generate", "--images", *images, "--transforms", str(transforms)])
    assert len(calls) == 1 and calls[0][:2] == (True, None)
    assert calls[0][2]["images"] == images and calls[0][2]["camera_metadata"] == metadata
