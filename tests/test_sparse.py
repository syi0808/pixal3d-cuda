import pytest
import torch
from torch.nn import functional as functional

from pixalcuda.ops import Limits
from pixalcuda.sparse import SparseConv, neighbor_map, requantize, subdivision


def test_submanifold_convolution_matches_dense():
    generator = torch.Generator().manual_seed(43)
    spatial = torch.cartesian_prod(torch.arange(3), torch.arange(3), torch.arange(3))
    coordinates = functional.pad(spatial[::2], (1, 0))
    features = torch.randn(len(coordinates), 4, generator=generator)
    module = SparseConv(4, 5, Limits(sparse_rows=3))
    dense = torch.zeros(1, 4, 3, 3, 3)
    dense[0, :, coordinates[:, 1], coordinates[:, 2], coordinates[:, 3]] = features.T
    expected = functional.conv3d(dense, module.weight.permute(0, 4, 1, 2, 3), module.bias, padding=1)
    expected = expected[0, :, coordinates[:, 1], coordinates[:, 2], coordinates[:, 3]].T
    with torch.inference_mode():
        actual = module(features, neighbor_map(coordinates))
    torch.testing.assert_close(actual, expected)


def test_subdivision_preserves_child_bit_order():
    coordinates = torch.tensor([[0, 2, 3, 4]])
    mask = torch.tensor([[True, False, True, False, False, False, False, True]])
    selection, children = subdivision(coordinates, mask, 8)
    assert selection.tolist() == [0, 2, 7]
    assert children.tolist() == [[0, 4, 6, 8], [0, 4, 7, 8], [0, 5, 7, 9]]
    with pytest.raises(MemoryError):
        subdivision(coordinates, mask, 2)


def test_requantize_512_has_explicit_32_grid_and_unique_tokens():
    coordinates = torch.tensor([[0, 0, 0, 0], [0, 511, 511, 511], [0, 1, 1, 1]])
    result = requantize(coordinates)
    assert result.tolist() == [[0, 0, 0, 0], [0, 31, 31, 31]]


def test_neighbor_map_rejects_duplicates():
    with pytest.raises(ValueError, match="unique"):
        neighbor_map(torch.zeros((2, 4), dtype=torch.int32))
