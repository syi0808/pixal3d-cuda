"""Projection-conditioned DiTs with official checkpoint parameter names."""

import math

import torch
from torch import nn
from torch.nn import functional as functional

from .ops import HeadNorm, LayerNorm32, Limits, apply_rope, attention, chunked_mlp, layer_norm, rope_phases


class TimestepEmbedder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, channels), nn.SiLU(), nn.Linear(channels, channels))

    def forward(self, timestep):
        frequencies = torch.exp(-math.log(10000) * torch.arange(128, device=timestep.device).float() / 128)
        angles = timestep.float().unsqueeze(-1) * frequencies
        embedding = torch.cat((angles.cos(), angles.sin()), -1).to(self.mlp[0].weight.dtype)
        return self.mlp(embedding)


class Attention(nn.Module):
    def __init__(self, channels, heads, context_channels=None, limits=Limits()):
        super().__init__()
        self.heads = heads
        self.head_dim = channels // heads
        self.limits = limits
        if context_channels is None:
            self.to_qkv = nn.Linear(channels, channels * 3)
        else:
            self.to_q = nn.Linear(channels, channels)
            self.to_kv = nn.Linear(context_channels, channels * 2)
        self.to_out = nn.Linear(channels, channels)
        self.q_rms_norm = HeadNorm(heads, self.head_dim)
        self.k_rms_norm = HeadNorm(heads, self.head_dim)

    def prepare_context(self, context):
        projected = self.to_kv(context).reshape(-1, 2, self.heads, self.head_dim)
        return self.k_rms_norm(projected[:, 0]), projected[:, 1]

    def forward(self, features, context=None, phases=None, prepared=None):
        if context is None:
            projected = self.to_qkv(features).reshape(-1, 3, self.heads, self.head_dim)
            queries, keys, values = projected.unbind(1)
            keys = self.k_rms_norm(keys)
        else:
            queries = self.to_q(features).reshape(-1, self.heads, self.head_dim)
            keys, values = self.prepare_context(context) if prepared is None else prepared
        queries = self.q_rms_norm(queries)
        if phases is not None:
            queries, keys = apply_rope(queries, phases), apply_rope(keys, phases)
        attended = attention(
            queries.transpose(0, 1).unsqueeze(0), keys.transpose(0, 1).unsqueeze(0),
            values.transpose(0, 1).unsqueeze(0), self.limits.attention_queries,
        ).squeeze(0).transpose(0, 1).reshape(len(features), -1)
        return self.to_out(attended)


class ProjectAttention(nn.Module):
    def __init__(self, channels, heads, context_channels, projection_channels, limits):
        super().__init__()
        self.cross_attn_block = Attention(channels, heads, context_channels, limits)
        self.proj_linear = nn.Linear(projection_channels, channels)

    def forward(self, features, context, prepared=None):
        global_features, projection = context["global"], context["proj"]
        cached = prepared if isinstance(prepared, dict) else {"kv": prepared, "projection": None}
        projected = cached["projection"]
        if projected is None:
            projected = self.proj_linear(projection).reshape_as(features)
        return self.cross_attn_block(features, global_features, prepared=cached["kv"]) + projected

    def unconditional(self):
        inner = self.cross_attn_block
        values = inner.to_kv.bias.reshape(2, -1)[1]
        return inner.to_out(values) + self.proj_linear.bias


class FeedForward(nn.Module):
    def __init__(self, channels, hidden, limits):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, channels))
        self.rows = limits.ffn_rows

    def forward(self, features):
        return chunked_mlp(features, self.mlp[0], self.mlp[2], self.rows)


class Block(nn.Module):
    def __init__(self, channels, heads, context_channels, projection_channels, hidden, limits):
        super().__init__()
        self.self_attn = Attention(channels, heads, limits=limits)
        self.norm2 = LayerNorm32(channels)
        self.cross_attn = ProjectAttention(channels, heads, context_channels, projection_channels, limits)
        self.mlp = FeedForward(channels, hidden, limits)
        self.modulation = nn.Parameter(torch.zeros(6 * channels))

    def forward(self, features, modulation, context, phases, prepared, negative):
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (modulation + self.modulation).chunk(6)
        hidden = layer_norm(features) * (1 + scale_attn) + shift_attn
        features = features + self.self_attn(hidden, phases=phases) * gate_attn
        if negative:
            cross = prepared["negative"] if isinstance(prepared, dict) else self.cross_attn.unconditional()
        else:
            cross = self.cross_attn(self.norm2(features), context, prepared)
        features = features + cross
        hidden = layer_norm(features) * (1 + scale_ffn) + shift_ffn
        return features + self.mlp(hidden) * gate_ffn


class Flow(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, projection_channels=2048,
                 channels=1536, heads=12, blocks=30, hidden=8192,
                 context_channels=1024, dense=False, limits=Limits()):
        super().__init__()
        self.dense = dense
        self.head_dim = channels // heads
        self.limits = limits
        self.t_embedder = TimestepEmbedder(channels)
        self.input_layer = nn.Linear(in_channels, channels)
        self.out_layer = nn.Linear(channels, out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 6 * channels))
        self.blocks = nn.ModuleList([
            Block(channels, heads, context_channels, projection_channels, hidden, limits) for _ in range(blocks)
        ])

    def forward(self, sample, timestep, context, phases, prepared, negative=False, concat=None):
        original_shape = sample.shape
        dtype = self.input_layer.weight.dtype
        features = sample.movedim(1, -1).reshape(-1, sample.shape[1]) if self.dense else sample
        if concat is not None:
            features = torch.cat((features, concat), -1)
        features = self.input_layer(features.to(dtype))
        modulation = self.adaLN_modulation(self.t_embedder(timestep))[0]
        for block, cached in zip(self.blocks, prepared):
            features = block(features, modulation, context, phases, cached, negative)
        output = self.out_layer(layer_norm(features, 1e-5)).float()
        return output.reshape(original_shape[0], *original_shape[2:], -1).movedim(-1, 1) if self.dense else output


def cfg_rescale(positive, guided, sparse):
    dimensions = tuple(range(positive.ndim)) if sparse else tuple(range(1, positive.ndim))
    correction = 0 if sparse else 1
    std_positive = positive.std(dim=dimensions, correction=correction, keepdim=True)
    std_guided = guided.std(dim=dimensions, correction=correction, keepdim=True)
    ratio = torch.where(std_guided > 0, std_positive / std_guided.clamp_min(torch.finfo(torch.float32).tiny), 1.0)
    return guided * ratio


@torch.inference_mode()
def sample_flow(model, noise, context, coordinates, params, concat=None):
    dtype = model.input_layer.weight.dtype
    context = {name: value.to(device=noise.device, dtype=dtype) for name, value in context.items()}
    phases = rope_phases(coordinates, model.head_dim)
    prepared = []
    remaining = model.limits.conditioning_cache_mb * 1024 ** 2
    projection_bytes = context["proj"].shape[1] * model.input_layer.out_features * torch.empty((), dtype=dtype).element_size()
    for block in model.blocks:
        projection = None
        if projection_bytes <= remaining:
            projection = block.cross_attn.proj_linear(context["proj"]).reshape(-1, model.input_layer.out_features)
            remaining -= projection_bytes
        prepared.append({"kv": block.cross_attn.cross_attn_block.prepare_context(context["global"]),
                         "projection": projection, "negative": block.cross_attn.unconditional()})
    model.last_cache_report = {"budget_bytes": model.limits.conditioning_cache_mb * 1024 ** 2,
                               "projection_bytes_per_block": projection_bytes,
                               "cached_blocks": sum(item["projection"] is not None for item in prepared),
                               "total_blocks": len(prepared),
                               "used_bytes": model.limits.conditioning_cache_mb * 1024 ** 2 - remaining}
    steps = params.get("steps", 12)
    times = torch.linspace(1, 0, steps + 1, dtype=torch.float64)
    rescale_t = params.get("rescale_t", 3.0)
    times = (rescale_t * times / (1 + (rescale_t - 1) * times)).tolist()
    strength = params.get("guidance_strength", 7.5)
    rescale = params.get("guidance_rescale", 0.5)
    interval = params.get("guidance_interval", (0.6, 1.0))
    sample = noise.float()
    for step, (current, previous) in enumerate(zip(times, times[1:]), 1):
        timestep = torch.tensor([1000 * current], device=sample.device)
        positive = model(sample, timestep, context, phases, prepared, concat=concat)
        if strength != 1 and interval[0] <= current <= interval[1]:
            negative = model(sample, timestep, context, phases, prepared, negative=True, concat=concat)
            prediction = strength * positive + (1 - strength) * negative
            if rescale:
                start_positive = (1 - 1e-5) * sample - (1e-5 + (1 - 1e-5) * current) * positive
                start_guided = (1 - 1e-5) * sample - (1e-5 + (1 - 1e-5) * current) * prediction
                rescaled = cfg_rescale(start_positive, start_guided, sparse=not model.dense)
                start = rescale * rescaled + (1 - rescale) * start_guided
                prediction = ((1 - 1e-5) * sample - start) / (1e-5 + (1 - 1e-5) * current)
        else:
            prediction = positive
        sample = sample - (current - previous) * prediction
        if not torch.isfinite(sample).all():
            raise FloatingPointError(f"Non-finite flow state at step {step}")
        print(f"    step {step}/{steps}", flush=True)
    return sample
