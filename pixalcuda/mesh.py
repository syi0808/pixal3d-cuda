"""Extract triangle mesh from decoder output.

Takes the ShapeSLatDecoder's 7-channel output and produces vertices + faces
via the flexible dual grid algorithm (same as TRELLIS.2's o_voxel.convert).

Channels:
  [0:3] vertex offsets → sigmoid → scaled to [-margin, 1+margin]
  [3:6] edge intersection flags → threshold > 0
  [6:7] quad split weight → softplus

All computation is numpy (CPU) — no GPU needed for mesh extraction.
"""
import numpy as np
_EDGE_NEIGHBOR_OFFSETS = np.array([[[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]], dtype=np.int64)
_QUAD_SPLIT_1 = np.array([0, 1, 2, 0, 2, 3], dtype=np.int64)
_QUAD_SPLIT_2 = np.array([0, 1, 3, 3, 1, 2], dtype=np.int64)

def decoder_output_to_mesh(feats: np.ndarray, coords: np.ndarray, resolution: int=256, voxel_margin: float=0.5):
    """Convert decoder output to a triangle mesh.

    Args:
        feats: [N, 7] — the 7-channel decoder output
        coords: [N, 4] — voxel coordinates (batch_idx, z, y, x)
        resolution: grid resolution for coordinate scaling
        voxel_margin: margin for vertex offset scaling

    Returns:
        vertices: [V, 3] float32 mesh vertices
        faces: [F, 3] int64 triangle indices
    """
    vertex_offsets = _sigmoid(feats[:, 0:3])
    vertex_offsets = (1 + 2 * voxel_margin) * vertex_offsets - voxel_margin
    intersection_flags = feats[:, 3:6] > 0
    quad_weights = _softplus(feats[:, 6:7])
    spatial_coords = coords[:, 1:4].astype(np.int64)
    return flexible_dual_grid_to_mesh(coords=spatial_coords, dual_vertices=vertex_offsets, intersected_flag=intersection_flags, split_weight=quad_weights, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], grid_size=resolution)

def flexible_dual_grid_to_mesh(coords, dual_vertices, intersected_flag, split_weight, aabb, grid_size=None, voxel_size=None):
    """Extract triangle mesh from sparse voxel dual-grid.

    Pure numpy implementation — same algorithm as trellis-mac's
    o_voxel_override_convert.py but without torch dependency.
    """
    coords = np.asarray(coords, dtype=np.int64)
    dual_vertices = np.asarray(dual_vertices, dtype=np.float32)
    intersected_flag = np.asarray(intersected_flag, dtype=bool)
    aabb = np.array(aabb, dtype=np.float32)
    if grid_size is not None:
        if isinstance(grid_size, (int, float)):
            grid_size = np.array([grid_size] * 3, dtype=np.float32)
        else:
            grid_size = np.array(grid_size, dtype=np.float32)
        voxel_size = (aabb[1] - aabb[0]) / grid_size
    elif voxel_size is not None:
        if isinstance(voxel_size, (int, float)):
            voxel_size = np.array([voxel_size] * 3, dtype=np.float32)
        else:
            voxel_size = np.array(voxel_size, dtype=np.float32)
    sample_count = coords.shape[0]
    packed = coords[:, 0] << 42 | coords[:, 1] << 21 | coords[:, 2]
    coord_to_idx = {}
    for voxel_index in range(sample_count):
        coord_to_idx[packed[voxel_index]] = voxel_index
    edge_neighbors = coords[:, None, None, :] + _EDGE_NEIGHBOR_OFFSETS[None, :, :, :]
    connected = edge_neighbors[intersected_flag]
    voxel_count = connected.shape[0]
    if voxel_count == 0:
        return (np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64))
    flat = connected.reshape(-1, 3)
    flat_packed = flat[:, 0] << 42 | flat[:, 1] << 21 | flat[:, 2]
    MISSING = 4294967295
    indices = np.array([coord_to_idx.get(coordinate_key, MISSING) for coordinate_key in flat_packed], dtype=np.int64)
    indices = indices.reshape(voxel_count, 4)
    valid = np.all(indices != MISSING, axis=1)
    quad_indices = indices[valid]
    quad_count = quad_indices.shape[0]
    if quad_count == 0:
        return (np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64))
    mesh_vertices = (coords.astype(np.float32) + dual_vertices) * voxel_size + aabb[0]
    if split_weight is None:
        a1 = quad_indices[:, _QUAD_SPLIT_1]
        n0 = np.cross(mesh_vertices[a1[:, 1]] - mesh_vertices[a1[:, 0]], mesh_vertices[a1[:, 2]] - mesh_vertices[a1[:, 0]])
        n1 = np.cross(mesh_vertices[a1[:, 2]] - mesh_vertices[a1[:, 1]], mesh_vertices[a1[:, 3]] - mesh_vertices[a1[:, 1]])
        align0 = np.abs(np.sum(n0 * n1, axis=1, keepdims=True))
        a2 = quad_indices[:, _QUAD_SPLIT_2]
        n0 = np.cross(mesh_vertices[a2[:, 1]] - mesh_vertices[a2[:, 0]], mesh_vertices[a2[:, 2]] - mesh_vertices[a2[:, 0]])
        n1 = np.cross(mesh_vertices[a2[:, 2]] - mesh_vertices[a2[:, 1]], mesh_vertices[a2[:, 3]] - mesh_vertices[a2[:, 1]])
        align1 = np.abs(np.sum(n0 * n1, axis=1, keepdims=True))
        mesh_triangles = np.where(align0 > align1, a1, a2).reshape(-1, 3)
    else:
        split_weight = np.asarray(split_weight, dtype=np.float32)
        sw = split_weight[quad_indices]
        if sw.ndim == 3:
            sw = sw.squeeze(-1)
        sw_02 = sw[:, 0] * sw[:, 2]
        sw_13 = sw[:, 1] * sw[:, 3]
        cond = (sw_02 > sw_13)[:, None].repeat(6, axis=1)
        mesh_triangles = np.where(cond, quad_indices[:, _QUAD_SPLIT_1], quad_indices[:, _QUAD_SPLIT_2]).reshape(-1, 3)
    return (mesh_vertices, mesh_triangles)

def _sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -88, 88)))

def _softplus(values):
    return np.where(values > 20, values, np.log1p(np.exp(np.minimum(values, 20))))
