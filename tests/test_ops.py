import pytest
import torch
from torch.nn import functional as functional

from pixalcuda.flow import Flow, ProjectAttention, cfg_rescale, sample_flow
from pixalcuda.ops import Limits, apply_rope, attention, rope_phases


@pytest.mark.parametrize("chunk", [1, 7, 100])
def test_query_chunking_preserves_global_attention(chunk):
    generator = torch.Generator().manual_seed(42)
    queries = torch.randn(1, 3, 23, 12, generator=generator)
    keys = torch.randn(1, 3, 29, 12, generator=generator)
    values = torch.randn(1, 3, 29, 12, generator=generator)
    expected = functional.scaled_dot_product_attention(queries, keys, values)
    torch.testing.assert_close(attention(queries, keys, values, chunk), expected)


def test_rope_matches_complex_rotation():
    coordinates = torch.tensor([[0, 1, 2], [4, 8, 3]])
    features = torch.randn(2, 3, 128)
    cosine, sine = rope_phases(coordinates, 128)
    complex_features = torch.view_as_complex(features.reshape(2, 3, 64, 2))
    expected = torch.view_as_real(complex_features * torch.complex(cosine, sine)).flatten(-2)
    torch.testing.assert_close(apply_rope(features, (cosine, sine)), expected)


def test_constant_negative_context_matches_explicit_zero_conditioning():
    module = ProjectAttention(24, 3, 16, 8, Limits())
    features = torch.randn(13, 24)
    context = {"global": torch.zeros(1, 5, 16), "proj": torch.zeros(1, 13, 8)}
    with torch.inference_mode():
        expected = module(features, context)
        actual = module.unconditional().expand_as(expected)
    torch.testing.assert_close(actual, expected)


def test_cfg_rescale_uses_population_std_for_sparse():
    positive = torch.tensor([[1., 2.], [3., 4.]])
    guided = positive * 3
    torch.testing.assert_close(cfg_rescale(positive, guided, True), positive)
    assert torch.isfinite(cfg_rescale(positive, torch.zeros_like(positive), True)).all()


def test_limits_reject_nonpositive_values():
    with pytest.raises(ValueError):
        Limits(attention_queries=0)


def test_cached_and_uncached_sampling_match():
    torch.manual_seed(12)
    model = Flow(8, 8, 16, channels=48, heads=4, blocks=2, hidden=96, context_channels=16, dense=True)
    coordinates = torch.cartesian_prod(torch.arange(2), torch.arange(2), torch.arange(2))
    noise = torch.randn(1, 8, 2, 2, 2)
    context = {"global": torch.randn(1, 5, 16), "proj": torch.randn(1, 8, 16)}
    params = {"steps": 3, "guidance_strength": 7.5, "guidance_interval": (0, 1)}
    expected = sample_flow(model, noise, context, coordinates, params)
    model.limits = Limits(conditioning_cache_mb=0)
    actual = sample_flow(model, noise, context, coordinates, params)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA GPU required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_cuda_attention_matches_fp32_reference(dtype):
    queries = torch.randn(1, 12, 513, 128, device="cuda", dtype=dtype)
    with torch.inference_mode():
        actual = attention(queries, queries, queries, 128)
        expected = functional.scaled_dot_product_attention(queries.float(), queries.float(), queries.float())
    torch.testing.assert_close(actual.float(), expected, atol=0.01, rtol=0.01)
