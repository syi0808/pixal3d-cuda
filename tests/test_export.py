import numpy as np
import trimesh

from pixalcuda.export import clean_mesh, export_pbr, validate_glb


def test_actual_uv_bake_embeds_both_pbr_images(tmp_path):
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.35)
    vertices, faces = clean_mesh(mesh.vertices, mesh.faces, 100)
    axis = np.linspace(0, 511, 8).astype(np.int32)
    coordinates = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    colors = coordinates.astype(np.float32) / 511
    attributes = np.column_stack((colors, np.full(len(colors), 0.3), np.full(len(colors), 0.7), np.ones(len(colors)))).astype(np.float32)
    destination = tmp_path / "result.glb"
    result = export_pbr(vertices, faces, coordinates, attributes, destination, texture_size=512)
    assert result["embedded_pbr"]
    assert result["triangles"] > 0
    assert validate_glb(destination) == result
    loaded = trimesh.load(destination, force="scene")
    material = next(iter(loaded.geometry.values())).visual.material
    assert material.baseColorTexture.size == (512, 512)
    roughness = np.asarray(material.metallicRoughnessTexture)
    assert abs(float(np.median(roughness[..., 1])) / 255 - 0.7) < 0.03
    assert abs(float(np.median(roughness[..., 2])) / 255 - 0.3) < 0.03
