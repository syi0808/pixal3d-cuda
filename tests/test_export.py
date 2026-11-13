import numpy as np
import pytest
import trimesh

from pixalcuda.bake import VoxelSampler
from pixalcuda.export import clean_mesh, export_pbr, validate_glb


@pytest.mark.parametrize("geometry_resolution", [512, 1024])
def test_actual_uv_bake_embeds_both_pbr_images(tmp_path, monkeypatch, geometry_resolution):
    grids = []
    def sampler(coordinates, attributes, grid_size=512):
        grids.append(grid_size)
        return VoxelSampler(coordinates, attributes, grid_size)
    monkeypatch.setattr("pixalcuda.export.VoxelSampler", sampler)
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.35)
    vertices, faces = clean_mesh(mesh.vertices, mesh.faces, 100)
    axis = np.linspace(0, geometry_resolution - 1, 8).astype(np.int32)
    coordinates = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    colors = coordinates.astype(np.float32) / (geometry_resolution - 1)
    attributes = np.column_stack((colors, np.full(len(colors), 0.3), np.full(len(colors), 0.7), np.ones(len(colors)))).astype(np.float32)
    destination = tmp_path / "result.glb"
    result = export_pbr(vertices, faces, coordinates, attributes, destination, texture_size=512,
                        geometry_resolution=geometry_resolution)
    assert result["embedded_pbr"]
    assert result["triangles"] > 0
    assert result["geometry_resolution"] == geometry_resolution
    assert grids == [geometry_resolution]
    assert validate_glb(destination) == result
    loaded = trimesh.load(destination, force="scene")
    material = next(iter(loaded.geometry.values())).visual.material
    assert material.baseColorTexture.size == (512, 512)
    roughness = np.asarray(material.metallicRoughnessTexture)
    assert abs(float(np.median(roughness[..., 1])) / 255 - 0.7) < 0.03
    assert abs(float(np.median(roughness[..., 2])) / 255 - 0.3) < 0.03


@pytest.mark.parametrize("geometry_resolution", [512, 1024])
def test_texture_sampling_uses_the_geometry_grid(geometry_resolution):
    axis = np.array([geometry_resolution // 2 - 1, geometry_resolution // 2])
    coordinates = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    attributes = coordinates.astype(np.float32) / (geometry_resolution - 1)
    sampler = VoxelSampler(coordinates, attributes, grid_size=geometry_resolution)
    assert np.allclose(sampler.sample([[0, 0, 0]]), 0.5, atol=1e-6)
