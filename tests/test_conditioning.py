import torch

from pixalcuda.conditioning import Dino, NAF, neighborhood_indices, project_grid
from pixalcuda.ops import Limits
from pixalcuda.preprocess import camera_from_fov


def test_projection_center_and_axis_convention():
    coordinates = torch.tensor([[0, 1, 1, 1], [0, 2, 1, 1], [0, 1, 2, 1]])
    projected = project_grid(coordinates, 3, 512, camera_from_fov(49.13))
    torch.testing.assert_close(projected[0], torch.full((2,), 1 / 512))
    assert projected[1, 0] > projected[0, 0]
    assert projected[2, 1] < projected[0, 1]


def test_neighborhood_windows_shift_inward_at_borders():
    positions = torch.tensor([0, 63, 63 * 64, 64 * 64 - 1])
    neighbors = neighborhood_indices(positions, 64, 16)
    assert neighbors.shape == (4, 81)
    assert neighbors[0, 0] == 0
    assert neighbors[0, -1] == 8 * 16 + 8
    assert neighbors[-1, 0] == 7 * 16 + 7
    assert neighbors[-1, -1] == 255


def test_tiny_dino_outputs_nonaffine_normalized_features():
    model = Dino(channels=32, heads=4, layers=1, limits=Limits(attention_queries=3))
    with torch.inference_mode():
        output = model(torch.rand(1, 3, 32, 32))
    assert output.shape == (1, 9, 32)
    torch.testing.assert_close(output.mean(-1), torch.zeros(1, 9), atol=1e-6, rtol=0)


def test_sparse_naf_returns_finite_projected_features():
    model = NAF(Limits(naf_queries=2))
    with torch.inference_mode():
        output = model.project(torch.rand(1, 3, 64, 64), torch.rand(1, 16, 16, 16), torch.tensor([[-1., -1.], [0., 0.], [1., 1.]]), target=64)
    assert output.shape == (3, 16)
    assert output.isfinite().all()
