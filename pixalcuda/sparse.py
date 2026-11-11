"""Submanifold convolution using bounded gather/GEMM, without CUDA extensions."""

import itertools

import torch
from torch import nn

from .ops import Limits


def coordinate_keys(coordinates):
    coordinates = coordinates.to(torch.int64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise ValueError("Coordinates must have shape [N, 4]")
    return (coordinates[:, 0] << 48) | (coordinates[:, 1] << 32) | (coordinates[:, 2] << 16) | coordinates[:, 3]


def neighbor_map(coordinates):
    if len(coordinates) == 0:
        raise ValueError("Decoder has no active voxels")
    if (coordinates < 0).any() or (coordinates > 65535).any():
        raise ValueError("Sparse coordinate fields must be unsigned 16-bit values")
    keys = coordinate_keys(coordinates)
    sorted_keys, order = keys.sort()
    if (sorted_keys[1:] == sorted_keys[:-1]).any():
        raise ValueError("Sparse coordinates must be unique")
    groups = []
    for offset in itertools.product((-1, 0, 1), repeat=3):
        neighbors = coordinates.clone().long()
        neighbors[:, 1:] += coordinates.new_tensor(offset)
        valid = ((neighbors >= 0) & (neighbors <= 65535)).all(-1)
        neighbor_keys = coordinate_keys(neighbors.clamp(0, 65535))
        locations = torch.searchsorted(sorted_keys, neighbor_keys).clamp_max(len(keys) - 1)
        targets = (valid & (sorted_keys[locations] == neighbor_keys)).nonzero().flatten()
        groups.append((order[locations[targets]].to(torch.int32), targets.to(torch.int32)))
    return groups


class SparseConv(nn.Module):
    def __init__(self, in_channels, out_channels, limits=Limits()):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, 3, 3, 3, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.rows = limits.sparse_rows
        nn.init.normal_(self.weight, std=(2 / (in_channels * 27)) ** 0.5)

    def forward(self, features, topology):
        output = torch.zeros((len(features), len(self.bias)), device=features.device, dtype=torch.float32)
        kernels = self.weight.reshape(len(self.bias), 27, features.shape[1]).permute(1, 0, 2)
        for kernel, (sources, targets) in zip(kernels, topology):
            for start in range(0, len(sources), self.rows):
                source = sources[start:start + self.rows].long()
                target = targets[start:start + self.rows].long()
                contribution = torch.nn.functional.linear(features[source], kernel).float()
                output[target] += contribution
        return (output + self.bias.float()).to(features.dtype)


def subdivision(coordinates, mask, max_voxels):
    selection = mask.flatten().nonzero().flatten()
    if len(selection) == 0:
        raise ValueError("Subdivision predicted no active children")
    if len(selection) > max_voxels:
        raise MemoryError(f"Subdivision needs {len(selection):,} voxels; limit is {max_voxels:,}. No resolution fallback was applied.")
    parents = selection // 8
    child = selection % 8
    offsets = torch.stack((child % 2, (child // 2) % 2, child // 4), -1)
    children = coordinates[parents].clone()
    children[:, 1:] = children[:, 1:] * 2 + offsets
    return selection, children


def requantize(coordinates, output_resolution=512):
    result = coordinates.clone()
    result[:, 1:] = torch.round((coordinates[:, 1:].double() + 0.5) / 512 * (output_resolution // 16 - 1)).to(result.dtype)
    return torch.unique(result, dim=0, sorted=True)
