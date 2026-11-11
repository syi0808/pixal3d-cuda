"""UV unwrap, PBR baking, and embedded glTF 2.0 validation."""

import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image
import trimesh

from .bake import VoxelSampler, inpaint_texture, rasterize_uv, uv_unwrap


def clean_mesh(vertices, faces, target_faces=100000):
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) > target_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
    if len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all():
        raise ValueError("Shape decoder did not produce a nonempty finite triangle mesh")
    mesh.fix_normals(multibody=True)
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int64)


def export_pbr(vertices, faces, voxel_coordinates, attributes, destination, texture_size=1024):
    if texture_size not in (512, 1024, 2048):
        raise ValueError("Supported texture sizes are 512, 1024, and 2048")
    if not np.isfinite(attributes).all() or len(attributes) == 0:
        raise ValueError("Texture attributes must be nonempty and finite")
    unwrapped, triangles, coordinates, mapping = uv_unwrap(vertices, faces)
    mask, face_indices, barycentric = rasterize_uv(coordinates, triangles, texture_size)
    if not mask.any():
        raise ValueError("UV atlas has no covered texels")
    covered = np.where(mask)
    positions = (unwrapped[triangles[face_indices[covered]]] * barycentric[covered][..., None]).sum(1)
    sampled = np.empty((len(positions), attributes.shape[1]), dtype=np.float32)
    sampler = VoxelSampler(voxel_coordinates, attributes)
    for start in range(0, len(positions), 32768):
        sampled[start:start + 32768] = sampler.sample(positions[start:start + 32768])
    rgba = np.zeros((texture_size, texture_size, 4), dtype=np.uint8)
    metallic_roughness = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    rgba[covered] = np.clip(sampled[:, [0, 1, 2, 5]] * 255, 0, 255).astype(np.uint8)
    metallic_roughness[covered[0], covered[1], 1] = np.clip(sampled[:, 4] * 255, 0, 255).astype(np.uint8)
    metallic_roughness[covered[0], covered[1], 2] = np.clip(sampled[:, 3] * 255, 0, 255).astype(np.uint8)
    rgba = inpaint_texture(rgba, mask)
    metallic_roughness = inpaint_texture(metallic_roughness, mask)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(rgba), baseColorFactor=[255, 255, 255, 255],
        metallicRoughnessTexture=Image.fromarray(metallic_roughness),
        metallicFactor=1.0, roughnessFactor=1.0, alphaMode="OPAQUE", doubleSided=True,
    )
    export_uvs = coordinates.copy()
    export_uvs[:, 1] = 1 - export_uvs[:, 1]
    mesh = trimesh.Trimesh(
        vertices=unwrapped * np.array([-1, 1, -1]), faces=triangles, process=False,
        vertex_normals=trimesh.Trimesh(vertices=vertices, faces=faces, process=False).vertex_normals[mapping] * np.array([-1, 1, -1]),
        visual=trimesh.visual.TextureVisuals(uv=export_uvs, material=material),
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.glb")
    try:
        mesh.export(temporary, file_type="glb")
        validation = validate_glb(temporary)
        temporary.replace(destination)
        return validation
    finally:
        temporary.unlink(missing_ok=True)


def validate_glb(path):
    data = Path(path).read_bytes()
    if len(data) < 20 or struct.unpack_from("<4sII", data) != (b"glTF", 2, len(data)):
        raise ValueError("Invalid GLB v2 header")
    length, kind = struct.unpack_from("<II", data, 12)
    if kind != 0x4E4F534A:
        raise ValueError("Missing glTF JSON chunk")
    document = json.loads(data[20:20 + length])
    primitive = document["meshes"][0]["primitives"][0]
    material = document["materials"][primitive["material"]]["pbrMetallicRoughness"]
    for name in ("baseColorTexture", "metallicRoughnessTexture"):
        texture = document["textures"][material[name]["index"]]
        if "bufferView" not in document["images"][texture["source"]]:
            raise ValueError("PBR texture is not embedded in the GLB")
    scene = trimesh.load(path, force="scene", process=False)
    meshes = list(scene.geometry.values())
    if not meshes or not all(len(mesh.faces) and np.isfinite(mesh.vertices).all() for mesh in meshes):
        raise ValueError("Exported GLB geometry is empty or non-finite")
    return {"glb_bytes": len(data), "triangles": sum(len(mesh.faces) for mesh in meshes), "embedded_pbr": True, "geometry_resolution": 512}
