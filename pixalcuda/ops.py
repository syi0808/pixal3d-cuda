"""Bounded global attention and checkpoint-compatible normalization."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class Limits:
    attention_queries: int = 256
    cross_attention_queries: int = 256
    ffn_rows: int = 1024
    sparse_rows: int = 4096
    naf_queries: int = 128
    conditioning_cache_mb: int = 128
    max_tokens: int = 16384
    max_voxels: int = 1500000
    sparse_prepack: int = 0

    def __post_init__(self):
        if type(self.cross_attention_queries) is not int or self.cross_attention_queries < 0:
            raise ValueError("cross_attention_queries must be a nonnegative integer; 0 means all queries")
        if type(self.sparse_prepack) is not int or self.sparse_prepack not in (0, 1):
            raise ValueError("sparse_prepack must be 0 or 1")
        if self.conditioning_cache_mb < 0 or any(value <= 0 for name, value in vars(self).items() if name not in ("conditioning_cache_mb", "sparse_prepack", "cross_attention_queries")):
            raise ValueError("All runtime limits must be positive")


def layer_norm(features, epsilon=1e-6, weight=None, bias=None):
    return functional.layer_norm(
        features.float(), (features.shape[-1],),
        None if weight is None else weight.float(),
        None if bias is None else bias.float(), epsilon,
    ).to(features.dtype)


class LayerNorm32(nn.Module):
    def __init__(self, channels, affine=True, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(channels)) if affine else None
        self.bias = nn.Parameter(torch.zeros(channels)) if affine else None

    def forward(self, features):
        return layer_norm(features, self.epsilon, self.weight, self.bias)


class HeadNorm(nn.Module):
    def __init__(self, heads, head_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, head_dim))
        self.scale = math.sqrt(head_dim)

    def forward(self, features):
        normalized = features.float() * torch.rsqrt(features.float().square().sum(-1, keepdim=True) + 1e-12)
        return (normalized * self.gamma.float() * self.scale).to(features.dtype)


def attention(queries, keys, values, query_chunk=256):
    output = torch.empty_like(queries, dtype=values.dtype)
    for start in range(0, queries.shape[-2], query_chunk):
        stop = start + query_chunk
        output[..., start:stop, :] = functional.scaled_dot_product_attention(
            queries[..., start:stop, :].contiguous(), keys, values, dropout_p=0.0,
        )
    return output


def rope_phases(coordinates, head_dim):
    frequency_dim = head_dim // 6
    frequencies = 10000.0 ** (-torch.arange(frequency_dim, device=coordinates.device).float() / frequency_dim)
    angles = (coordinates.float().unsqueeze(-1) * frequencies).flatten(1)
    angles = functional.pad(angles, (0, head_dim // 2 - angles.shape[-1]))
    return angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)


def apply_rope(features, phases):
    cosine, sine = phases
    paired = features.float().reshape(*features.shape[:-1], -1, 2)
    first, second = paired.unbind(-1)
    return torch.stack((first * cosine - second * sine, first * sine + second * cosine), -1).flatten(-2).to(features.dtype)


def chunked_mlp(features, first, second, rows, activation="gelu"):
    output = features.new_empty((*features.shape[:-1], second.out_features))
    flat_input = features.reshape(-1, features.shape[-1])
    flat_output = output.reshape(-1, output.shape[-1])
    for start in range(0, len(flat_input), rows):
        hidden = first(flat_input[start:start + rows])
        hidden = functional.silu(hidden) if activation == "silu" else functional.gelu(hidden, approximate="tanh")
        flat_output[start:start + rows] = second(hidden)
    return output
