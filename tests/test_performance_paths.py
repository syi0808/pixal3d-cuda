import json

import pytest
from safetensors.torch import save_file
import torch

from pixalcuda.conditioning import Dino
from pixalcuda.decoder import SparseDecoder, SparseUpsample
from pixalcuda.flow import Attention, FeedForward, Flow, sample_flow
from pixalcuda.ops import Limits, attention
from pixalcuda.runtime import Runtime
from pixalcuda.sparse import SparseConv, neighbor_map
from pixalcuda.weights import load_model, validate_materialized_tensors


def test_cross_attention_has_independent_query_chunks_and_full_query_mode(monkeypatch):
    torch.manual_seed(21)
    limits = Limits(attention_queries=2, cross_attention_queries=0)
    features, context = torch.randn(7, 12), torch.randn(1, 5, 8)
    chunks = []
    def record(queries, keys, values, query_chunk):
        chunks.append((keys.shape[-2], query_chunk))
        return attention(queries, keys, values, query_chunk)
    monkeypatch.setattr("pixalcuda.flow.attention", record)
    cross = Attention(12, 3, 8, limits)
    self_attention = Attention(12, 3, limits=limits)
    with torch.inference_mode():
        full = cross(features, context)
        prepared = cross.prepare_context(context)
        torch.testing.assert_close(cross(features, context, prepared=prepared), full)
        cross.limits = Limits(attention_queries=2, cross_attention_queries=3)
        torch.testing.assert_close(cross(features, context), full)
        self_attention(features)
    assert chunks == [(5, 7), (5, 7), (5, 3), (7, 2)]


@pytest.mark.parametrize("rows", [4, 8, 32])
def test_larger_ffn_chunks_keep_all_rows_and_weights(rows):
    torch.manual_seed(22)
    model = FeedForward(12, 32, Limits(ffn_rows=1))
    features = torch.randn(11, 12)
    with torch.inference_mode():
        expected = model(features)
        model.rows = rows
        torch.testing.assert_close(model(features), expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("rows", [4, 16, 64])
def test_larger_sparse_chunks_preserve_tail_rows_and_contributions(rows):
    torch.manual_seed(23)
    model = SparseConv(4, 5, Limits(sparse_rows=1))
    coordinates = torch.nn.functional.pad(torch.cartesian_prod(torch.arange(3), torch.arange(3), torch.arange(3)), (1, 0))
    features = torch.randn(len(coordinates), 4)
    topology = neighbor_map(coordinates)
    with torch.inference_mode():
        expected = model(features, topology)
        model.rows = rows
        torch.testing.assert_close(model(features, topology), expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_cross_attention_query_limit_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="cross_attention_queries"):
        Limits(cross_attention_queries=value)


@pytest.mark.parametrize("texture", [False, True])
def test_coordinate_only_stops_before_final_convolutions_topology_and_blocks(monkeypatch, texture):
    torch.manual_seed(18)
    coordinates = torch.nn.functional.pad(torch.cartesian_prod(torch.arange(2), torch.arange(2), torch.arange(2)), (1, 0))
    features = torch.randn(len(coordinates), 32)
    model = SparseDecoder(texture=texture, channels=(64, 32, 16), block_counts=(1, 1, 1))
    guides = [torch.ones((len(coordinates), 8), dtype=torch.bool), torch.ones((len(coordinates) * 8, 8), dtype=torch.bool)] if texture else None
    with torch.inference_mode():
        _, expected, _ = model(features, coordinates, guides=guides)
    final_upsample = model.blocks[-2][-1]
    assert isinstance(final_upsample, SparseUpsample)
    def forbidden(*args, **kwargs):
        raise AssertionError("Coordinate-only path must not compute final child features")
    monkeypatch.setattr(final_upsample.conv1, "forward", forbidden)
    monkeypatch.setattr(final_upsample.conv2, "forward", forbidden)
    monkeypatch.setattr(model.blocks[-1][0], "forward", forbidden)
    calls = []
    def topology(value):
        calls.append(len(value))
        return neighbor_map(value)
    monkeypatch.setattr("pixalcuda.decoder.neighbor_map", topology)
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    with torch.inference_mode():
        actual = model(features, coordinates, guides=guides, coordinates_only=True)
    assert torch.equal(actual, expected)
    assert len(calls) == 2


def test_coordinate_fast_path_preserves_empty_and_voxel_guards():
    coordinates = torch.zeros((1, 4), dtype=torch.int32)
    features = torch.randn(1, 32)
    model = SparseDecoder(channels=(64, 32), block_counts=(0, 0), limits=Limits(max_voxels=1))
    with torch.no_grad():
        model.blocks[0][0].to_subdiv.weight.zero_()
        model.blocks[0][0].to_subdiv.bias.fill_(-1)
    with torch.inference_mode(), pytest.raises(ValueError, match="no active"):
        model(features, coordinates, coordinates_only=True)
    with torch.no_grad():
        model.blocks[0][0].to_subdiv.bias.fill_(1)
    with torch.inference_mode(), pytest.raises(MemoryError, match="limit"):
        model(features, coordinates, coordinates_only=True)


def test_dino_reuses_identical_rope_storage_across_layers():
    model = Dino(channels=32, heads=4, layers=3)
    phases = []
    handles = [block.register_forward_pre_hook(lambda module, args: phases.append(args[1:])) for block in model.layer]
    try:
        with torch.inference_mode():
            output = model(torch.rand(1, 3, 32, 32))
    finally:
        for handle in handles:
            handle.remove()
    assert torch.isfinite(output).all()
    assert len(phases) == 3
    assert len({phase[0].data_ptr() for phase in phases}) == 1
    assert len({phase[1].data_ptr() for phase in phases}) == 1


def test_finite_status_is_batched_without_per_tensor_host_reads(monkeypatch):
    checks = [(f"weight-{index}", torch.float32, torch.tensor(True)) for index in range(700)]
    original = torch.Tensor.__bool__
    reads = []
    def read(value):
        reads.append(value.numel())
        return original(value)
    monkeypatch.setattr(torch.Tensor, "__bool__", read)
    validate_materialized_tensors(checks)
    assert reads == [1]
    reads.clear()
    checks[37] = ("overflow-weight", torch.float16, torch.tensor(False))
    with pytest.raises(FloatingPointError, match="overflow-weight.*wider dtype"):
        validate_materialized_tensors(checks)
    assert reads == [1]


def test_opt_in_sparse_packing_keeps_names_shapes_and_values(tmp_path):
    torch.manual_seed(19)
    original = SparseConv(4, 5)
    checkpoint = tmp_path / "sparse.safetensors"
    save_file(original.state_dict(), checkpoint)
    packed = load_model(lambda: SparseConv(4, 5, Limits(sparse_prepack=1)), checkpoint, "cpu", torch.float32)
    assert packed.state_dict().keys() == original.state_dict().keys()
    torch.testing.assert_close(packed.weight, original.weight, rtol=0, atol=0)
    kernels = packed.weight.reshape(5, 27, 4).permute(1, 0, 2)
    assert kernels.is_contiguous() and all(kernel.is_contiguous() for kernel in kernels)
    storage = packed.weight.data_ptr()
    packed.prepack_weights_()
    assert packed.weight.data_ptr() == storage
    coordinates = torch.nn.functional.pad(torch.cartesian_prod(torch.arange(3), torch.arange(3), torch.arange(3)), (1, 0))
    features = torch.randn(len(coordinates), 4)
    topology = neighbor_map(coordinates)
    with torch.inference_mode():
        torch.testing.assert_close(packed(features, topology), original(features, topology))


def test_model_report_separates_load_callback_and_cleanup(tmp_path):
    checkpoint = tmp_path / "linear.safetensors"
    save_file(torch.nn.Linear(3, 2).state_dict(), checkpoint)
    runtime = Runtime(tmp_path / "run", "cpu", {"seed": 42})
    with runtime.stage("test_model"):
        output = runtime.run_model(lambda: torch.nn.Linear(3, 2), checkpoint, lambda model: model(torch.ones(1, 3)), dtype=torch.float32)
    assert output.shape == (1, 2)
    record = json.loads((runtime.directory / "report.json").read_text())["models"][0]
    assert record["stage"] == "test_model" and record["status"] == "complete"
    assert record["parameter_bytes"] == 32
    assert all(record[name] >= 0 for name in ("load_wall_seconds", "callback_wall_seconds", "cleanup_wall_seconds"))
    assert "callback_cuda_interval_ms" not in record


def test_stage_entry_does_not_repeat_model_boundary_allocator_flush(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("pixalcuda.runtime.release", lambda *args, **kwargs: calls.append(kwargs))
    runtime = Runtime(tmp_path / "run", "cpu", {"seed": 42})
    with runtime.stage("no_model"):
        pass
    assert not calls
    checkpoint = tmp_path / "linear.safetensors"
    save_file(torch.nn.Linear(3, 2).state_dict(), checkpoint)
    with runtime.stage("model"):
        runtime.run_model(lambda: torch.nn.Linear(3, 2), checkpoint, lambda model: model(torch.ones(1, 3)), dtype=torch.float32)
    assert calls == [{"synchronize": False}]


def test_cache_metrics_show_explicit_budget_without_changing_outputs():
    torch.manual_seed(20)
    model = Flow(4, 4, 8, channels=12, heads=2, blocks=2, hidden=24, context_channels=8)
    coordinates = torch.tensor([[0, 0, 0], [1, 1, 1]])
    context = {"global": torch.randn(1, 5, 8), "proj": torch.randn(1, 2, 8)}
    noise = torch.randn(2, 4)
    params = {"steps": 2, "guidance_strength": 1}
    cached = sample_flow(model, noise, context, coordinates, params)
    assert model.last_cache_report["cached_blocks"] == 2
    assert model.last_cache_report["used_bytes"] == 2 * 2 * 12 * 4
    model.limits = Limits(conditioning_cache_mb=0)
    uncached = sample_flow(model, noise, context, coordinates, params)
    assert model.last_cache_report["used_bytes"] == 0 and model.last_cache_report["cached_blocks"] == 0
    torch.testing.assert_close(cached, uncached)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA GPU required")
def test_cuda_loader_rejects_overflow_before_callback(tmp_path):
    model = torch.nn.Linear(3, 2)
    with torch.no_grad():
        model.weight.fill_(70000)
    checkpoint = tmp_path / "overflow.safetensors"
    save_file(model.state_dict(), checkpoint)
    with pytest.raises(FloatingPointError, match="weight.*wider dtype"):
        load_model(lambda: torch.nn.Linear(3, 2), checkpoint, "cuda", torch.float16)
