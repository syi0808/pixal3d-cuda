"""Dense occupancy and guided sparse shape/PBR decoders."""

import torch
from torch import nn
from torch.nn import functional as functional

from .ops import LayerNorm32, Limits, chunked_mlp, layer_norm
from .sparse import SparseConv, neighbor_map, subdivision


class ChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, features):
        return layer_norm(features.movedim(1, -1), 1e-5, self.weight, self.bias).movedim(-1, 1)


class DenseResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = ChannelNorm(channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = ChannelNorm(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, features):
        hidden = self.conv1(functional.silu(self.norm1(features)))
        return features + self.conv2(functional.silu(self.norm2(hidden)))


class DenseUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels * 8, 3, padding=1)

    def forward(self, features):
        features = self.conv(features)
        batch, channels, depth, height, width = features.shape
        return features.reshape(batch, channels // 8, 2, 2, 2, depth, height, width).permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(batch, channels // 8, depth * 2, height * 2, width * 2)


class StructureDecoder(nn.Module):
    def __init__(self, channels=(512, 128, 32)):
        super().__init__()
        self.input_layer = nn.Conv3d(8, channels[0], 3, padding=1)
        self.middle_block = nn.Sequential(DenseResidual(channels[0]), DenseResidual(channels[0]))
        blocks = []
        for level, width in enumerate(channels):
            blocks.extend((DenseResidual(width), DenseResidual(width)))
            if level + 1 < len(channels):
                blocks.append(DenseUpsample(width, channels[level + 1]))
        self.blocks = nn.Sequential(*blocks)
        self.out_layer = nn.Sequential(ChannelNorm(channels[-1]), nn.SiLU(), nn.Conv3d(channels[-1], 1, 3, padding=1))

    def forward(self, features):
        return self.out_layer(self.blocks(self.middle_block(self.input_layer(features))))


class SparseNext(nn.Module):
    def __init__(self, channels, limits):
        super().__init__()
        self.conv = SparseConv(channels, channels, limits)
        self.norm = LayerNorm32(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, channels * 4), nn.SiLU(), nn.Linear(channels * 4, channels))
        self.rows = limits.ffn_rows

    def forward(self, features, topology):
        hidden = self.norm(self.conv(features, topology))
        return features + chunked_mlp(hidden, self.mlp[0], self.mlp[2], self.rows, "silu")


class SparseUpsample(nn.Module):
    def __init__(self, channels, output_channels, predictive, limits):
        super().__init__()
        self.norm1 = LayerNorm32(channels)
        self.norm2 = LayerNorm32(output_channels, affine=False)
        self.conv1 = SparseConv(channels, output_channels * 8, limits)
        self.conv2 = SparseConv(output_channels, output_channels, limits)
        self.predictive = predictive
        self.limits = limits
        if predictive:
            self.to_subdiv = nn.Linear(channels, 8)

    def subdivide(self, features, coordinates, guide=None):
        mask = self.to_subdiv(features) > 0 if self.predictive else guide
        if mask is None or mask.shape != (len(features), 8):
            raise ValueError("Texture subdivision must use aligned shape masks")
        selection, children = subdivision(coordinates, mask, self.limits.max_voxels)
        return selection, children, mask

    def forward(self, features, coordinates, topology, guide=None):
        selection, children, mask = self.subdivide(features, coordinates, guide)
        hidden = self.conv1(functional.silu(self.norm1(features)), topology)
        hidden = hidden.reshape(-1, hidden.shape[-1] // 8)[selection]
        skip = features.reshape(-1, features.shape[-1] // 8)[selection]
        skip = skip.repeat_interleave(hidden.shape[-1] // skip.shape[-1], dim=1)
        topology = neighbor_map(children)
        hidden = self.conv2(functional.silu(self.norm2(hidden)), topology)
        return hidden + skip, children, topology, mask


class SparseDecoder(nn.Module):
    def __init__(self, texture=False, channels=(1024, 512, 256, 128, 64),
                 block_counts=(4, 16, 8, 4, 0), limits=Limits(), verbose=True):
        super().__init__()
        self.verbose = verbose
        self.from_latent = nn.Linear(32, channels[0])
        self.output_layer = nn.Linear(channels[-1], 6 if texture else 7)
        self.blocks = nn.ModuleList()
        for level, (width, count) in enumerate(zip(channels, block_counts)):
            blocks = [SparseNext(width, limits) for _ in range(count)]
            if level + 1 < len(channels):
                blocks.append(SparseUpsample(width, channels[level + 1], not texture, limits))
            self.blocks.append(nn.ModuleList(blocks))

    def forward(self, features, coordinates, guides=None, coordinates_only=False):
        features = self.from_latent(features.to(self.from_latent.weight.dtype))
        topology = neighbor_map(coordinates)
        masks = []
        for level_index, level in enumerate(self.blocks):
            for block in level:
                if isinstance(block, SparseNext):
                    features = block(features, topology)
                else:
                    guide = None if guides is None else guides[len(masks)].to(coordinates.device)
                    if coordinates_only and level_index == len(self.blocks) - 2:
                        _, children, _ = block.subdivide(features, coordinates, guide)
                        if self.verbose:
                            print(f"    decoder level {len(masks) + 1}: {len(children):,} voxels (coordinates only)", flush=True)
                        return children
                    features, coordinates, topology, mask = block(features, coordinates, topology, guide)
                    masks.append(None if coordinates_only else mask.cpu())
                    if self.verbose:
                        print(f"    decoder level {len(masks)}: {len(coordinates):,} voxels", flush=True)
        if coordinates_only:
            return coordinates
        return self.output_layer(layer_norm(features, 1e-5)), coordinates, masks
