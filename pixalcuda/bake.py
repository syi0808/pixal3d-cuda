import numpy as np


class VoxelSampler:
    """Trilinear sparse lookup with one sorted index and one lazy KD-tree."""

    def __init__(self, coordinates, attributes, grid_size=512):
        self.coordinates = np.asarray(coordinates, dtype=np.int64)
        self.attributes = np.asarray(attributes, dtype=np.float32)
        self.grid_size = grid_size
        if self.coordinates.shape != (len(self.attributes), 3) or not len(self.attributes):
            raise ValueError("Voxel coordinates/attributes must be aligned and nonempty")
        if self.coordinates.min() < 0 or self.coordinates.max() >= 1 << 20:
            raise ValueError("Voxel fields must fit unsigned 20-bit values")
        keys = self.pack(self.coordinates)
        self.order = np.argsort(keys)
        self.keys = keys[self.order]
        self.tree = None

    @staticmethod
    def pack(coordinates):
        return (coordinates[..., 0] << 40) | (coordinates[..., 1] << 20) | coordinates[..., 2]

    def sample(self, positions):
        positions = np.asarray(positions, dtype=np.float32)
        voxel_positions = (positions + 0.5) * self.grid_size - 0.5
        lower = np.floor(voxel_positions).astype(np.int64)
        fraction = voxel_positions - lower
        output = np.zeros((len(positions), self.attributes.shape[1]), dtype=np.float32)
        total = np.zeros(len(positions), dtype=np.float32)
        for offset in np.ndindex(2, 2, 2):
            corners = lower + offset
            valid = ((corners >= 0) & (corners < 1 << 20)).all(-1)
            packed = self.pack(corners.clip(0, (1 << 20) - 1))
            indices = np.searchsorted(self.keys, packed).clip(0, len(self.keys) - 1)
            valid &= self.keys[indices] == packed
            weights = np.prod(np.where(np.asarray(offset), fraction, 1 - fraction), axis=1)
            output[valid] += self.attributes[self.order[indices[valid]]] * weights[valid, None]
            total[valid] += weights[valid]
        present = total > 0
        output[present] /= total[present, None]
        if not present.all():
            from scipy.spatial import cKDTree
            if self.tree is None:
                self.tree = cKDTree((self.coordinates + 0.5) / self.grid_size - 0.5)
            _, nearest = self.tree.query(positions[~present], k=1)
            output[~present] = self.attributes[nearest]
        return output

def uv_unwrap(vertices, faces):
    """UV unwrap a mesh using xatlas.

    Args:
        vertices: [V, 3] float32
        faces: [F, 3] int

    Returns:
        new_vertices: [V', 3] float32 (may have more vertices due to seam splits)
        new_faces: [F, 3] uint32
        uvs: [V', 2] float32 in [0, 1]
        vmapping: [V'] int — maps new vertex indices to original vertex indices
    """
    import xatlas
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices.astype(np.float32), faces.astype(np.uint32))
    chart_options = xatlas.ChartOptions()
    chart_options.max_iterations = 0
    atlas.generate(chart_options=chart_options)
    vmapping, new_faces, uvs = atlas[0]
    new_vertices = vertices[vmapping]
    return (new_vertices, new_faces, uvs, vmapping)

def rasterize_uv(uvs, faces, texture_size=1024):
    """Rasterize triangles in UV space to get per-pixel barycentric coords.

    For each pixel in the texture, determines which triangle covers it and
    computes barycentric weights for interpolation.

    Args:
        uvs: [V, 2] float32 UV coordinates in [0, 1]
        faces: [F, 3] uint32
        texture_size: output texture resolution

    Returns:
        pixel_mask: [H, W] bool — which pixels are covered
        pixel_face_idx: [H, W] int — which face covers each pixel (-1 if none)
        pixel_bary: [H, W, 3] float32 — barycentric weights
    """
    texture_height = texture_width = texture_size
    face_count = len(faces)
    uv_px = uvs * texture_size
    tri_uvs = uv_px[faces]
    bb_min = np.floor(tri_uvs.min(axis=1)).astype(np.int32)
    bb_max = np.ceil(tri_uvs.max(axis=1)).astype(np.int32)
    bb_min = np.clip(bb_min, 0, texture_size - 1)
    bb_max = np.clip(bb_max, 0, texture_size - 1)
    pixel_face_idx = np.full((texture_height, texture_width), -1, dtype=np.int32)
    pixel_bary = np.zeros((texture_height, texture_width, 3), dtype=np.float32)
    v0 = tri_uvs[:, 0]
    v1 = tri_uvs[:, 1]
    v2 = tri_uvs[:, 2]
    d00 = np.sum((v1 - v0) * (v1 - v0), axis=1)
    d01 = np.sum((v1 - v0) * (v2 - v0), axis=1)
    d11 = np.sum((v2 - v0) * (v2 - v0), axis=1)
    inv_denom = 1.0 / (d00 * d11 - d01 * d01 + 1e-10)
    BATCH = 10000
    for start in range(0, face_count, BATCH):
        end = min(start + BATCH, face_count)
        batch_indices = np.arange(start, end)
        for fi in batch_indices:
            x_min, y_min = bb_min[fi]
            x_max, y_max = bb_max[fi]
            if x_min == x_max or y_min == y_max:
                continue
            xs = np.arange(x_min, x_max + 1) + 0.5
            ys = np.arange(y_min, y_max + 1) + 0.5
            px, py = np.meshgrid(xs, ys)
            points = np.stack([px.ravel(), py.ravel()], axis=1)
            dp = points - v0[fi]
            dp_d0 = dp[:, 0] * (v1[fi, 0] - v0[fi, 0]) + dp[:, 1] * (v1[fi, 1] - v0[fi, 1])
            dp_d1 = dp[:, 0] * (v2[fi, 0] - v0[fi, 0]) + dp[:, 1] * (v2[fi, 1] - v0[fi, 1])
            second_weight = (d11[fi] * dp_d0 - d01[fi] * dp_d1) * inv_denom[fi]
            third_weight = (d00[fi] * dp_d1 - d01[fi] * dp_d0) * inv_denom[fi]
            inside = (second_weight >= 0) & (third_weight >= 0) & (second_weight + third_weight <= 1)
            if not inside.any():
                continue
            pxi = (points[inside, 0] - 0.5).astype(np.int32)
            pyi = (points[inside, 1] - 0.5).astype(np.int32)
            valid = (pxi >= 0) & (pxi < texture_width) & (pyi >= 0) & (pyi < texture_height)
            pxi = pxi[valid]
            pyi = pyi[valid]
            u_in = second_weight[inside][valid]
            v_in = third_weight[inside][valid]
            pixel_face_idx[pyi, pxi] = fi
            pixel_bary[pyi, pxi, 0] = 1.0 - u_in - v_in
            pixel_bary[pyi, pxi, 1] = u_in
            pixel_bary[pyi, pxi, 2] = v_in
    pixel_mask = pixel_face_idx >= 0
    return (pixel_mask, pixel_face_idx, pixel_bary)

def inpaint_texture(image, mask, radius=3):
    """Inpaint missing pixels in a texture map.

    Args:
        image: [H, W, C] uint8 texture
        mask: [H, W] bool — True where pixels are valid

    Returns:
        inpainted: [H, W, C] uint8
    """
    try:
        import cv2
        inpaint_mask = (~mask).astype(np.uint8)
        if image.ndim == 2 or image.shape[2] == 1:
            img_2d = image.squeeze() if image.ndim == 3 else image
            result = cv2.inpaint(img_2d, inpaint_mask, radius, cv2.INPAINT_TELEA)
            return result[:, :, None] if image.ndim == 3 else result
        elif image.shape[2] == 3:
            return cv2.inpaint(image, inpaint_mask, radius, cv2.INPAINT_TELEA)
        else:
            rgb = cv2.inpaint(image[:, :, :3], inpaint_mask, radius, cv2.INPAINT_TELEA)
            rest = []
            for channel_index in range(3, image.shape[2]):
                ch = cv2.inpaint(image[:, :, channel_index], inpaint_mask, radius, cv2.INPAINT_TELEA)
                rest.append(ch[:, :, None])
            return np.concatenate([rgb] + rest, axis=2)
    except ImportError:
        from scipy.ndimage import distance_transform_edt
        result = image.copy()
        for channel_index in range(image.shape[2]):
            channel = image[:, :, channel_index].astype(np.float32)
            _, indices = distance_transform_edt(~mask, return_indices=True)
            result[:, :, channel_index] = channel[indices[0], indices[1]]
        return result
