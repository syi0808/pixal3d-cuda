"""Topology-preserving CPU cleanup for sparse dual-grid decoder meshes."""

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def edge_groups(faces):
    edges = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    lower = edges.min(axis=1).astype(np.uint64)
    upper = edges.max(axis=1).astype(np.uint64)
    keys = (lower << 32) | upper
    order = np.argsort(keys, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(keys[order])) + 1]
    return order, starts, np.diff(np.r_[starts, len(order)])


def compact(vertices, faces):
    used, inverse = np.unique(faces, return_inverse=True)
    return vertices[used], inverse.reshape(-1, 3)


def sanitize(vertices, faces):
    triangles = vertices[faces].astype(np.float64)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    threshold = np.finfo(np.float64).eps * max(float(np.ptp(vertices, axis=0).max()) ** 2, 1e-30)
    faces = faces[np.linalg.norm(cross, axis=1) > threshold]
    if not len(faces):
        raise ValueError("Mesh contains no nondegenerate triangles")
    _, geometric_ids = np.unique(vertices, axis=0, return_inverse=True)
    _, unique = np.unique(np.sort(geometric_ids[faces], axis=1), axis=0, return_index=True)
    return compact(vertices, faces[np.sort(unique)])


def components(faces):
    order, starts, counts = edge_groups(faces)
    owners = order // 3
    anchors = np.repeat(owners[starts], counts)
    keep = np.ones(len(order), dtype=bool)
    keep[starts] = False
    adjacency = sparse.coo_matrix((np.ones(keep.sum(), dtype=np.int8),
                                   (anchors[keep], owners[keep])), shape=(len(faces), len(faces)))
    return connected_components(adjacency, directed=False)


def remove_floaters(vertices, faces, minimum_area=1e-5):
    count, labels = components(faces)
    triangles = vertices[faces].astype(np.float64)
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) * 0.5
    totals = np.bincount(labels, weights=areas, minlength=count)
    keep = (totals >= minimum_area) | (np.arange(count) == totals.argmax())
    return compact(vertices, faces[keep[labels]])


def split_nonmanifold(vertices, faces):
    order, starts, counts = edge_groups(faces)
    paired = starts[counts == 2]

    def edge_corners(occurrences):
        owners, local = occurrences // 3, occurrences % 3
        following = (local + 1) % 3
        forward = faces[owners, local] < faces[owners, following]
        return owners * 3 + np.where(forward, local, following), owners * 3 + np.where(forward, following, local)

    first_left, second_left = edge_corners(order[paired])
    first_right, second_right = edge_corners(order[paired + 1])
    rows, columns = np.r_[first_left, second_left], np.r_[first_right, second_right]
    adjacency = sparse.coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, columns)),
                                   shape=(faces.size, faces.size))
    count, labels = connected_components(adjacency, directed=False)
    representatives = np.full(count, faces.size, dtype=np.int64)
    np.minimum.at(representatives, labels, np.arange(faces.size))
    return vertices[faces.reshape(-1)[representatives]], labels.reshape(-1, 3)


def orient_faces(vertices, faces):
    order, starts, counts = edge_groups(faces)
    paired = starts[counts == 2]
    first, second = order[paired], order[paired + 1]
    owners_first, owners_second = first // 3, second // 3
    same = faces[owners_first, first % 3] == faces[owners_second, second % 3]
    rows, columns = np.r_[owners_first, owners_second], np.r_[owners_second, owners_first]
    parity = np.r_[same, same].astype(np.int8) + 1
    adjacency = sparse.csr_matrix((parity, (rows, columns)), shape=(len(faces), len(faces)))
    seen, flips = np.zeros(len(faces), dtype=bool), np.zeros(len(faces), dtype=bool)
    for root in range(len(faces)):
        if seen[root]:
            continue
        seen[root] = True
        stack = [root]
        while stack:
            current = stack.pop()
            for position in range(adjacency.indptr[current], adjacency.indptr[current + 1]):
                neighbor = adjacency.indices[position]
                expected = flips[current] ^ (adjacency.data[position] == 2)
                if not seen[neighbor]:
                    seen[neighbor], flips[neighbor] = True, expected
                    stack.append(neighbor)
    oriented = faces.copy()
    oriented[flips] = oriented[flips][:, [0, 2, 1]]
    return vertices, oriented


def fill_small_holes(vertices, faces, maximum_perimeter=0.03):
    order, starts, counts = edge_groups(faces)
    boundary = order[starts[counts == 1]]
    _, geometric_ids = np.unique(vertices, axis=0, return_inverse=True)
    geometric_order, geometric_starts, geometric_counts = edge_groups(geometric_ids[faces])
    multiplicity = np.empty(faces.size, dtype=np.int32)
    multiplicity[geometric_order] = np.repeat(geometric_counts, geometric_counts)
    boundary = boundary[multiplicity[boundary] == 1]
    owners, local = boundary // 3, boundary % 3
    edges = faces[owners[:, None], np.column_stack((local, (local + 1) % 3))]
    outgoing, incoming = {}, {}
    for start, end in edges:
        outgoing.setdefault(int(start), []).append(int(end))
        incoming.setdefault(int(end), []).append(int(start))
    visited, extra_vertices, extra_faces = set(), [], []
    for root in outgoing:
        if root in visited:
            continue
        loop, current = [], root
        while current not in visited and len(outgoing.get(current, ())) == len(incoming.get(current, ())) == 1:
            visited.add(current)
            loop.append(current)
            current = outgoing[current][0]
        if current != root or len(loop) < 3:
            continue
        points = vertices[loop]
        perimeter = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1).sum()
        if not 0 < perimeter <= maximum_perimeter:
            continue
        if len(loop) == 3:
            extra_faces.append([loop[2], loop[1], loop[0]])
        else:
            center = len(vertices) + len(extra_vertices)
            extra_vertices.append(points.mean(axis=0))
            extra_faces.extend([[following, start, center] for start, following in zip(loop, loop[1:] + loop[:1])])
    if extra_vertices:
        vertices = np.concatenate((vertices, np.asarray(extra_vertices, dtype=vertices.dtype)))
    if extra_faces:
        faces = np.concatenate((faces, np.asarray(extra_faces, dtype=np.int64)))
    return vertices, faces


def separate_conflicts(vertices, faces):
    order, starts, counts = edge_groups(faces)
    paired = starts[counts == 2]
    first, second = order[paired], order[paired + 1]
    conflicts = faces[first // 3, first % 3] == faces[second // 3, second % 3]
    isolated = np.unique(np.maximum(first[conflicts] // 3, second[conflicts] // 3))
    if not len(isolated):
        return vertices, faces
    extra = vertices[faces[isolated]].reshape(-1, 3)
    faces = faces.copy()
    faces[isolated] = len(vertices) + np.arange(len(extra)).reshape(-1, 3)
    print(f"    separated {len(isolated)} nonorientable conflict faces without moving or deleting triangles", flush=True)
    return np.concatenate((vertices, extra)), faces


def mesh_metrics(vertices, faces):
    order, starts, counts = edge_groups(faces)
    paired = starts[counts == 2]
    first, second = order[paired], order[paired + 1]
    conflicts = faces[first // 3, first % 3] == faces[second // 3, second % 3]
    _, geometric_ids = np.unique(vertices, axis=0, return_inverse=True)
    duplicates = len(faces) - len(np.unique(np.sort(geometric_ids[faces], axis=1), axis=0))
    return {"vertices": len(vertices), "triangles": len(faces), "boundary_edges": int((counts == 1).sum()),
            "nonmanifold_edges": int((counts > 2).sum()), "inconsistent_manifold_edges": int(conflicts.sum()),
            "coincident_triangles": duplicates}


def shading_geometry(vertices, faces, crease_degrees=75.0):
    triangles = vertices[faces].astype(np.float64)
    crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(crosses, axis=1)
    if not (lengths > 0).all() or not 0 < crease_degrees < 180:
        raise ValueError("Shading requires nondegenerate triangles and a valid crease angle")
    normals = crosses / lengths[:, None]
    order, starts, counts = edge_groups(faces)
    paired = starts[counts == 2]
    first, second = order[paired], order[paired + 1]
    first_owner, second_owner = first // 3, second // 3
    opposite = faces[first_owner, first % 3] != faces[second_owner, second % 3]
    smooth = opposite & (np.einsum("ij,ij->i", normals[first_owner], normals[second_owner]) >= np.cos(np.deg2rad(crease_degrees)))
    first, second = first[smooth], second[smooth]
    rows = np.r_[first, first // 3 * 3 + (first % 3 + 1) % 3]
    columns = np.r_[second // 3 * 3 + (second % 3 + 1) % 3, second]
    adjacency = sparse.coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, columns)), shape=(faces.size, faces.size))
    count, groups = connected_components(adjacency, directed=False)
    angles = np.empty(faces.shape, dtype=np.float64)
    for corner in range(3):
        first_side = triangles[:, (corner + 1) % 3] - triangles[:, corner]
        second_side = triangles[:, (corner + 2) % 3] - triangles[:, corner]
        angles[:, corner] = np.arctan2(np.linalg.norm(np.cross(first_side, second_side), axis=1),
                                      np.einsum("ij,ij->i", first_side, second_side))
    accumulated = np.zeros((count, 3), dtype=np.float64)
    parent_normals = np.repeat(normals, 3, axis=0)
    np.add.at(accumulated, groups, parent_normals * angles.reshape(-1, 1))
    normal_lengths = np.linalg.norm(accumulated, axis=1)
    accumulated /= np.maximum(normal_lengths[:, None], 1e-30)
    corner_normals = accumulated[groups]
    fallback = (normal_lengths[groups] < 1e-12) | (np.einsum("ij,ij->i", corner_normals, parent_normals) <= 0)
    groups[fallback] = count + np.arange(fallback.sum())
    corner_normals[fallback] = parent_normals[fallback]
    _, representatives, inverse = np.unique(groups, return_index=True, return_inverse=True)
    return vertices[faces.reshape(-1)[representatives]], inverse.reshape(-1, 3), corner_normals[representatives].astype(np.float32)


def repair(vertices, faces):
    vertices, faces = sanitize(vertices, faces)
    vertices, faces = split_nonmanifold(vertices, faces)
    vertices, faces = orient_faces(vertices, faces)
    vertices, faces = fill_small_holes(vertices, faces)
    vertices, faces = sanitize(vertices, faces)
    vertices, faces = split_nonmanifold(vertices, faces)
    vertices, faces = orient_faces(vertices, faces)
    return separate_conflicts(vertices, faces)


def clean_geometry(vertices, faces, target_faces=100000):
    import trimesh

    vertices, faces = np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Expected vertices [V, 3] and triangles [F, 3]")
    if not len(faces) or not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Mesh must contain nonempty finite geometry and valid indices")
    report = {"policy": "manifold_fans_v1", "before": mesh_metrics(vertices, faces)}
    vertices, faces = sanitize(vertices, faces)
    vertices, faces = remove_floaters(vertices, faces)
    vertices, faces = repair(vertices, faces)
    report["before_simplification"] = mesh_metrics(vertices, faces)
    if target_faces > 0:
        for target in (target_faces * 3, target_faces):
            if len(faces) > target:
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                mesh = mesh.simplify_quadric_decimation(face_count=target)
                vertices, faces = repair(np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces))
    report["after"] = mesh_metrics(vertices, faces)
    if report["after"]["nonmanifold_edges"] or report["after"]["inconsistent_manifold_edges"]:
        raise ValueError("Mesh cleanup left nonmanifold or inconsistently oriented edges")
    print(f"    mesh cleanup: {report['before']} -> {report['after']}", flush=True)
    return vertices.astype(np.float32), faces.astype(np.int64), report
