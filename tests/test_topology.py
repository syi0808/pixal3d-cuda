import numpy as np
import trimesh

from pixalcuda.export import clean_mesh
from pixalcuda.topology import fill_small_holes, mesh_metrics, orient_faces, remove_floaters, shading_geometry, split_nonmanifold


def test_nonmanifold_repair_preserves_triangle_positions_without_welding_fans():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
    assert mesh_metrics(vertices, faces)["nonmanifold_edges"] == 1
    repaired, triangles = split_nonmanifold(vertices, faces)
    np.testing.assert_array_equal(repaired[triangles], vertices[faces])
    assert mesh_metrics(repaired, triangles)["nonmanifold_edges"] == 0


def test_bow_tie_vertex_is_split_before_decimation():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 3, 4]])
    repaired, triangles = split_nonmanifold(vertices, faces)
    np.testing.assert_array_equal(repaired[triangles], vertices[faces])
    assert len(repaired) == 6


def test_small_attached_detail_is_not_filtered_as_detached_noise():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1e-6, 0],
                         [2, 0, 0], [2.001, 0, 0], [2, .001, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [1, 0, 3], [4, 5, 6]])
    retained, triangles = remove_floaters(vertices, faces)
    np.testing.assert_array_equal(retained[triangles], vertices[faces[:2]])


def test_winding_unification_follows_manifold_adjacency():
    mesh = trimesh.creation.icosphere(subdivisions=1)
    faces = mesh.faces.copy()
    faces[::2] = faces[::2, ::-1]
    assert mesh_metrics(mesh.vertices, faces)["inconsistent_manifold_edges"] > 0
    vertices, faces = orient_faces(mesh.vertices, faces)
    assert mesh_metrics(vertices, faces)["inconsistent_manifold_edges"] == 0


def test_hole_fill_is_limited_by_world_space_perimeter():
    mesh = trimesh.creation.icosphere(subdivisions=0)
    faces = mesh.faces[:-1]
    vertices, triangles = fill_small_holes(mesh.vertices * .001, faces)
    assert mesh_metrics(vertices, triangles)["boundary_edges"] == 0
    vertices, triangles = fill_small_holes(mesh.vertices, faces)
    assert len(triangles) == len(faces)


def test_cleanup_does_not_weld_coincident_disconnected_sheets():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                         [0, 0, 0], [1, 0, 0], [0, -1, 0],
                         [0, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [3, 5, 4], [6, 7, 8]])
    retained, triangles, report = clean_mesh(vertices, faces, 0, return_report=True)
    assert len(retained) == 9 and len(triangles) == 3
    assert report["after"]["nonmanifold_edges"] == 0
    assert np.allclose(np.linalg.norm(trimesh.Trimesh(retained, triangles, process=False).vertex_normals, axis=1), 1)


def test_cleanup_removes_duplicate_and_degenerate_faces():
    mesh = trimesh.creation.icosphere(subdivisions=1)
    faces = np.concatenate((mesh.faces, mesh.faces[:3, ::-1], [[0, 0, 0]]))
    vertices, triangles, report = clean_mesh(mesh.vertices, faces, 0, return_report=True)
    assert len(triangles) == len(mesh.faces)
    assert report["after"]["inconsistent_manifold_edges"] == 0


def test_geometric_duplicate_faces_are_removed_without_vertex_welding():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]] * 2, dtype=np.float32)
    faces = np.array([[0, 1, 2], [3, 5, 4]])
    retained, triangles, report = clean_mesh(vertices, faces, 0, return_report=True)
    assert len(triangles) == 1
    assert report["before"]["coincident_triangles"] == 1
    assert report["after"]["coincident_triangles"] == 0


def test_sharp_opposing_faces_cannot_cancel_export_normals():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [1, 0, 3]])
    unwrapped, triangles, normals = shading_geometry(vertices, faces)
    np.testing.assert_array_equal(unwrapped[triangles], vertices[faces])
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1)
    assert len(unwrapped) == 6
    np.testing.assert_allclose(normals[triangles[0]], [[0, 0, 1]] * 3)
    np.testing.assert_allclose(normals[triangles[1]], [[0, 0, -1]] * 3)


def test_smooth_normals_match_across_continuous_sphere_fans():
    mesh = trimesh.creation.icosphere(subdivisions=1)
    vertices, faces, normals = shading_geometry(mesh.vertices, mesh.faces)
    assert len(vertices) == len(mesh.vertices)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-6)
    assert (np.einsum("ij,ij->i", vertices, normals) > .99).all()
