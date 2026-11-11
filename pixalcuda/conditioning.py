"""Native DINOv3, sparse NAF projection, and camera-aligned sampling."""

import math

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as functional

from .ops import Limits, attention, layer_norm


def image_tensor(image, size, device):
    array = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32) / 255
    return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).to(device)


class LayerScale(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(channels))

    def forward(self, features):
        return features * self.lambda1


class DinoAttention(nn.Module):
    def __init__(self, channels, heads, limits):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels)
        self.o_proj = nn.Linear(channels, channels)
        self.limits = limits

    def forward(self, features, cosine, sine, prefix=5):
        batch, tokens, channels = features.shape
        projected = [layer(features).reshape(batch, tokens, self.heads, -1).transpose(1, 2)
                     for layer in (self.q_proj, self.k_proj, self.v_proj)]
        queries, keys, values = projected
        def rotate(tensor):
            patches = tensor[:, :, prefix:].float()
            first, second = patches.chunk(2, -1)
            rotated = patches * cosine + torch.cat((-second, first), -1) * sine
            return torch.cat((tensor[:, :, :prefix], rotated.to(tensor.dtype)), 2)
        result = attention(rotate(queries), rotate(keys), values, self.limits.attention_queries)
        return self.o_proj(result.transpose(1, 2).reshape(batch, tokens, channels))


class DinoMLP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up_proj = nn.Linear(channels, channels * 4)
        self.down_proj = nn.Linear(channels * 4, channels)


class DinoLayer(nn.Module):
    def __init__(self, channels, heads, limits):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, eps=1e-5)
        self.attention = DinoAttention(channels, heads, limits)
        self.layer_scale1 = LayerScale(channels)
        self.norm2 = nn.LayerNorm(channels, eps=1e-5)
        self.mlp = DinoMLP(channels)
        self.layer_scale2 = LayerScale(channels)
        self.rows = limits.ffn_rows

    def forward(self, features, cosine, sine):
        features = features + self.layer_scale1(self.attention(self.norm1(features), cosine, sine))
        normalized = self.norm2(features)
        hidden = torch.empty_like(features)
        for start in range(0, features.shape[1], self.rows):
            hidden[:, start:start + self.rows] = self.mlp.down_proj(functional.gelu(self.mlp.up_proj(normalized[:, start:start + self.rows])))
        return features + self.layer_scale2(hidden)


class DinoEmbeddings(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, channels))
        self.register_tokens = nn.Parameter(torch.zeros(1, 4, channels))
        self.patch_embeddings = nn.Conv2d(3, channels, 16, stride=16)


class Dino(nn.Module):
    def __init__(self, channels=1024, heads=16, layers=24, limits=Limits()):
        super().__init__()
        self.embeddings = DinoEmbeddings(channels)
        self.layer = nn.ModuleList([DinoLayer(channels, heads, limits) for _ in range(layers)])
        self.head_dim = channels // heads

    def forward(self, image):
        dtype = self.embeddings.patch_embeddings.weight.dtype
        mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        standard = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        patches = self.embeddings.patch_embeddings(((image - mean) / standard).to(dtype))
        height, width = patches.shape[-2:]
        features = torch.cat((self.embeddings.cls_token, self.embeddings.register_tokens, patches.flatten(2).transpose(1, 2)), 1)
        rows = (torch.arange(height, device=image.device).float() + 0.5) / height * 2 - 1
        columns = (torch.arange(width, device=image.device).float() + 0.5) / width * 2 - 1
        coordinates = torch.stack(torch.meshgrid(rows, columns, indexing="ij"), -1).reshape(-1, 2)
        frequencies = 100.0 ** (-torch.arange(self.head_dim // 4, device=image.device).float() * 4 / self.head_dim)
        angles = (coordinates.unsqueeze(-1) * frequencies * (2 * math.pi)).flatten(1).repeat(1, 2)
        for block in self.layer:
            features = block(features, angles.cos(), angles.sin())
        normalized = layer_norm(features, 1e-5)
        if not torch.isfinite(normalized).all():
            raise FloatingPointError("Non-finite DINOv3 features. The official backbone requires FP32 execution on T4; do not cache these features.")
        return normalized


class GuideBlock(nn.Module):
    def __init__(self, channels, kernel):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel, padding=kernel // 2, padding_mode="reflect")
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel, padding=kernel // 2, padding_mode="reflect")

    def forward(self, features):
        hidden = self.conv1(functional.silu(self.norm1(features)))
        return self.conv2(functional.silu(self.norm2(hidden)))


class GuideRoPE(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("periods", 100.0 ** (2 * torch.arange(16).float() / 32))

    def forward(self, features):
        batch, channels, height, width = features.shape
        rows = (torch.arange(height, device=features.device).float() + 0.5) / height * 2 - 1
        columns = (torch.arange(width, device=features.device).float() + 0.5) / width * 2 - 1
        coordinates = torch.stack(torch.meshgrid(rows, columns, indexing="ij"), -1).reshape(-1, 2)
        angles = (2 * math.pi * coordinates.unsqueeze(-1) / self.periods).flatten(1).repeat(1, 2)
        reshaped = features.reshape(batch, 4, 64, height * width).transpose(2, 3)
        first, second = reshaped.chunk(2, -1)
        return (reshaped * angles.cos() + torch.cat((-second, first), -1) * angles.sin()).transpose(2, 3).reshape_as(features)


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = self.branch(1)
        self.sem_encoder = self.branch(3)
        self.rope = GuideRoPE()

    @staticmethod
    def branch(kernel):
        return nn.Sequential(nn.Conv2d(3, 128, kernel, padding=kernel // 2, padding_mode="reflect"), GuideBlock(128, kernel), GuideBlock(128, kernel))

    def forward(self, image, target):
        branches = []
        for branch in (self.encoder, self.sem_encoder):
            encoded = branch(image)
            branches.append(functional.adaptive_avg_pool2d(encoded, (target, target)))
            del encoded
        return self.rope(torch.cat(branches, 1))


def neighborhood_indices(positions, target, source, kernel=9):
    dilation = target // source
    rows, columns = positions // target, positions % target
    start_row = (rows // dilation - kernel // 2).clamp(0, source - kernel)
    start_column = (columns // dilation - kernel // 2).clamp(0, source - kernel)
    offsets = torch.arange(kernel, device=positions.device)
    return ((start_row[:, None, None] + offsets[None, :, None]) * source +
            start_column[:, None, None] + offsets[None, None, :]).flatten(1)


class NAF(nn.Module):
    def __init__(self, limits=Limits()):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.limits = limits

    def project(self, image, features, grid, target=512):
        source = features.shape[-1]
        if target % source or source < 9:
            raise ValueError("NAF requires a divisible target and at least 9 source patches per axis")
        queries = self.image_encoder(image.float(), target)
        keys = functional.adaptive_avg_pool2d(queries, (source, source))
        queries = queries.reshape(4, 64, -1).transpose(1, 2)
        keys = keys.reshape(4, 64, -1).transpose(1, 2)
        values = features.float().reshape(4, features.shape[1] // 4, -1).transpose(1, 2)
        output = torch.empty((len(grid), features.shape[1]), device=image.device)
        for start in range(0, len(grid), self.limits.naf_queries):
            coordinates = (grid[start:start + self.limits.naf_queries] + 1) * target / 2 - 0.5
            lower = coordinates.floor().long()
            weights = coordinates - lower
            corner_offsets = lower.new_tensor([[0, 0], [1, 0], [0, 1], [1, 1]])
            corners = (lower[:, None] + corner_offsets).clamp(0, target - 1)
            positions = (corners[..., 1] * target + corners[..., 0]).flatten()
            neighbors = neighborhood_indices(positions, target, source)
            scores = (queries[:, positions, None, :] * keys[:, neighbors, :]).sum(-1) / 8
            probabilities = scores.softmax(-1)
            attended = torch.einsum("hqn,hqnc->hqc", probabilities, values[:, neighbors, :])
            attended = attended.reshape(4, len(coordinates), 4, -1)
            horizontal, vertical = weights.unbind(-1)
            bilinear = torch.stack(((1 - horizontal) * (1 - vertical), horizontal * (1 - vertical),
                                    (1 - horizontal) * vertical, horizontal * vertical), -1)
            output[start:start + len(coordinates)] = (attended * bilinear[None, :, :, None]).sum(2).transpose(0, 1).flatten(1)
        return output


def project_grid(coordinates, resolution, image_size, camera):
    spatial = coordinates[:, 1:].float() / (resolution - 1) * 2 - 1
    points = torch.stack((spatial[:, 0], -spatial[:, 2], spatial[:, 1]), -1) / camera["mesh_scale"] / 2
    depth = points[:, 1] + camera["distance"]
    focal = image_size / (2 * math.tan(camera["camera_angle_x"] / 2))
    pixels = torch.stack((focal * points[:, 0] / (depth + 1e-8) + image_size / 2,
                          -focal * points[:, 2] / (depth + 1e-8) + image_size / 2), -1)
    return (pixels + 0.5) / image_size * 2 - 1


def build_conditioning(encoded, image, coordinates, resolution, camera, naf=None):
    patch_count = image.shape[-1] // 16
    spatial = encoded[:, 5:].transpose(1, 2).reshape(1, -1, patch_count, patch_count).float()
    grid = project_grid(coordinates, resolution, image.shape[-1], camera)
    projected = functional.grid_sample(spatial, grid[None, None], mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T
    if naf is not None:
        projected = torch.cat((projected, naf.project(image, spatial, grid)), -1)
    return {"global": encoded[:, :5].float(), "proj": projected.unsqueeze(0)}
