import ast
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import asdict
import inspect
from pathlib import Path
import subprocess
from types import MethodType, SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.nn import functional

from pixalcuda.cameras import orbit_camera_pose
from pixalcuda.optimizations import (ConditioningCache, FeatureCache, OptimizationOptions,
                                    OptimizationSession, active_projection, sampling_cache)
from pixalcuda.upstream_sources import lock


class SparseValue:
    def __init__(self, feats, coords):
        self.feats, self.coords = feats, coords

    def replace(self, feats):
        return SparseValue(feats, self.coords)

    def type(self, dtype):
        return self.replace(self.feats.type(dtype))


class Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(3, 8, 2, stride=2)
        with torch.no_grad():
            self.patch_embeddings.weight.copy_(torch.arange(96).reshape(8, 3, 2, 2) / 96)
            self.patch_embeddings.bias.zero_()

    def forward(self, image, bool_masked_pos=None):
        patches = self.patch_embeddings(image).flatten(2).transpose(1, 2)
        return torch.cat((patches.mean(1, keepdim=True).expand(-1, 5, -1), patches), 1)


class TinyBackbone(nn.Module):
    loads = 0
    forwards = 0

    def __init__(self):
        super().__init__()
        self.embeddings = Embeddings()
        self.layer = nn.ModuleList()
        self.config = SimpleNamespace(patch_size=2, hidden_size=8, num_register_tokens=4)

    @classmethod
    def from_pretrained(cls, path):
        cls.loads += 1
        return cls()

    def rope_embeddings(self, image):
        type(self).forwards += 1
        return None


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean)[None, :, None, None]
        self.std = torch.tensor(std)[None, :, None, None]

    def forward(self, image):
        return (image - self.mean) / self.std


class ToyUpsampler(nn.Module):
    def forward(self, image, features, target):
        return functional.interpolate(features, size=target, mode="bilinear", align_corners=False)


def compile_nodes(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


def compile_method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    node.decorator_list = [decorator for decorator in node.decorator_list if not isinstance(decorator, ast.Name)]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


@pytest.fixture(scope="module")
def upstream():
    root = Path(__file__).resolve().parents[1] / "outputs" / "upstream-Pixal3D-review"
    if not (root / "pixal3d").is_dir():
        pytest.skip("Pinned source checkout required; CI checks it out without downloading weights")
    assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == lock()["sources"]["pixal3d"]["revision"]
    assert not subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    namespace = {"torch": torch, "nn": nn, "F": functional, "np": np, "Image": Image,
                 "DINOv3ViTModel": TinyBackbone, "SparseTensor": SparseValue,
                 "transforms": SimpleNamespace(Compose=lambda layers: nn.Sequential(*layers), Normalize=Normalize)}
    feature_path = root / "pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py"
    compile_nodes(feature_path, ["project_points_to_image_batch", "sample_features", "ProjGrid", "ProjGridMV",
                                "compute_relative_calc_mat", "DinoV3ProjFeatureExtractor", "DinoV3ProjMultiViewFeatureExtractor"], namespace)
    compile_nodes(root / "pixal3d/modules/attention/proj_attention.py", ["ProjectAttention"], namespace)
    compile_nodes(root / "pixal3d/modules/sparse/attention/proj_attention.py", ["SparseProjectAttention"], namespace)
    namespace.update(abstractmethod=abstractmethod, contextmanager=contextmanager)
    compile_nodes(root / "pixal3d/utils/elastic_utils.py", ["ElasticModuleMixin"], namespace)
    namespace["sv_shape"] = compile_method(root / "pixal3d/pipelines/pixal3d_image_to_3d.py",
        "Pixal3DImageTo3DPipeline", "get_proj_cond_shape", dict(namespace))
    mv_path = root / "pixal3d/pipelines/pixal3d_mv_image_to_3d.py"
    for name in ("_views_of", "_image_for", "_run_extractor", "get_proj_cond_shape"):
        namespace["mv" + name] = compile_method(mv_path, "Pixal3DMVImageTo3DPipeline", name, dict(namespace))
    return SimpleNamespace(**namespace)


def extractor(upstream, multiview=False, image_size=16):
    kind = upstream.DinoV3ProjMultiViewFeatureExtractor if multiview else upstream.DinoV3ProjFeatureExtractor
    model = kind("pinned-model", image_size=image_size, grid_resolution=4, use_naf_upsample=True, naf_target_size=12)
    model.naf_model = ToyUpsampler()
    return model.eval()


def pipe(upstream, multiview, low_vram):
    result = SimpleNamespace(device=torch.device("cpu"), low_vram=low_vram)
    result.get_proj_cond_shape = MethodType(upstream.mvget_proj_cond_shape if multiview else upstream.sv_shape, result)
    if multiview:
        result._views_of = MethodType(upstream.mv_views_of, result)
        result._image_for = upstream.mv_image_for
        result._run_extractor = MethodType(upstream.mv_run_extractor, result)
    return result


@pytest.mark.parametrize("multiview", [False, True])
@pytest.mark.parametrize("override", [None, 6])
@pytest.mark.parametrize("low_vram", [False, True])
def test_active_projection_matches_pinned_dense_builder(upstream, monkeypatch, multiview, override, low_vram):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    model, pipeline = extractor(upstream, multiview), pipe(upstream, multiview, low_vram)
    resolution = override or 4
    coordinates = torch.tensor([[0, resolution - 1, 0, resolution - 1], [0, 0, 0, 0],
                                [0, 1, 2, 1], [0, 1, 2, 1]], dtype=torch.int32)
    generator = torch.Generator().manual_seed(7)
    options = {"grid_resolution_override": override}
    if multiview:
        image = {"images": {16: torch.rand(1, 3, 3, 16, 16, generator=generator)},
                 "transform_matrix": torch.tensor([[orbit_camera_pose(angle, 0, 2) for angle in (0, 90, 240)]])}
        options.update(camera_angle_x=torch.tensor([[.3, .7, 1.1]]), distance=torch.tensor([[2., 2., 2.]]), mesh_scale=.8)
    else:
        image = [Image.fromarray((torch.rand(16, 16, 3, generator=generator).numpy() * 255).astype(np.uint8))]
        options.update(camera_angle_x=.3, distance=1.2, mesh_scale=.8)
    original_grid = model.proj_grid
    points = original_grid.grid_points.clone()
    expected = pipeline.get_proj_cond_shape(model, image, coordinates, **options)
    records = []
    optimized = active_projection(pipeline, SparseValue, pipeline.get_proj_cond_shape, multiview, records)
    with torch.inference_mode():
        actual = optimized(model, image, coordinates, **options)
    for name in ("cond", "neg_cond"):
        torch.testing.assert_close(actual[name]["global"], expected[name]["global"], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual[name]["proj"].feats, expected[name]["proj"].feats, atol=1e-6, rtol=1e-6)
        assert torch.equal(actual[name]["proj"].coords, coordinates)
    assert model.grid_resolution == 4 and torch.equal(model.proj_grid.grid_points, points)
    assert records == [{"grid_resolution": resolution, "dense_points": resolution**3,
                        "projected_points": len(coordinates), "naf_target_size": [12, 12]}]


def test_shared_dino_and_features_match_pinned_extract_features(upstream):
    TinyBackbone.loads = TinyBackbone.forwards = 0
    images = {size: torch.rand(1, 3, size, size) for size in (16, 24)}
    with torch.inference_mode():
        expected = [extractor(upstream, image_size=size).extract_features(images[size]) for size in (16, 16, 24, 24)]
    assert TinyBackbone.loads == 4 and TinyBackbone.forwards == 4
    TinyBackbone.loads = TinyBackbone.forwards = 0
    original = inspect.getattr_static(TinyBackbone, "from_pretrained")
    with torch.inference_mode(), OptimizationSession(OptimizationOptions(), TinyBackbone, upstream.DinoV3ProjFeatureExtractor, SparseValue) as session:
        models = [extractor(upstream, image_size=size) for size in (16, 16, 24, 24)]
        actual = [model.extract_features(images[size]) for model, size in zip(models, (16, 16, 24, 24))]
        assert len({id(model.model) for model in models}) == 1
        assert session.features.stats["hits"] == 2 and session.features.stats["misses"] == 2
        for result, reference in zip(actual, expected):
            assert torch.equal(result, reference)
        changed = models[0].extract_features(images[16] + .1)
        assert not torch.equal(changed, actual[0])
    assert TinyBackbone.loads == 1 and TinyBackbone.forwards == 3
    assert inspect.getattr_static(TinyBackbone, "from_pretrained") is original
    assert session.features.stats["used_bytes"] == 0


class GlobalCross(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_kv = nn.Linear(6, 16)

    def forward(self, features, context):
        source = features.feats if isinstance(features, SparseValue) else features
        projected = self.to_kv(context).mean(-2)[..., :8]
        output = source + projected.unsqueeze(-2)
        return features.replace(output[0]) if isinstance(features, SparseValue) else output


class ToyFlow(nn.Module):
    image_attn_mode = "proj"

    def __init__(self, projector_type, sparse, dtype):
        super().__init__()
        self.sparse = sparse
        self.blocks = nn.ModuleList([nn.Module() for index in range(3)])
        for block in self.blocks:
            block.cross_attn = projector_type(GlobalCross(), 8, 6)
        self.to(dtype).eval()

    def forward(self, sample, timestep, cond):
        dtype = next(self.parameters()).dtype
        global_features = cond["global"].to(dtype)
        projection = cond["proj"].type(dtype)
        features = sample.replace(sample.feats + timestep) if self.sparse else sample + timestep
        for block in self.blocks:
            features = block.cross_attn(features, (global_features, projection))
        return features


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("elastic", [False, True])
def test_sampling_cache_matches_pinned_project_attention_and_is_scoped(upstream, sparse, dtype, elastic):
    torch.manual_seed(9)
    model_type = type("ElasticToyFlow", (upstream.ElasticModuleMixin, ToyFlow), {}) if elastic else ToyFlow
    model = model_type(upstream.SparseProjectAttention if sparse else upstream.ProjectAttention, sparse, dtype)
    coordinates = torch.zeros(7, 4, dtype=torch.int32)
    projection = torch.rand(7, 6) if sparse else torch.rand(1, 7, 6)
    context = {"global": torch.rand(1, 5, 6), "proj": SparseValue(projection, coordinates) if sparse else projection}
    negative = {"global": torch.zeros(1, 5, 6), "proj": SparseValue(torch.zeros_like(projection), coordinates) if sparse else torch.zeros_like(projection)}
    noise = torch.rand(7, 8, dtype=dtype) if sparse else torch.rand(1, 7, 8, dtype=dtype)
    original = model.forward
    with torch.inference_mode():
        expected = []
        for timestep in range(4):
            for conditioning in (context, negative):
                sample = SparseValue(noise, coordinates) if sparse else noise
                expected.append(model(sample, timestep, conditioning))
        with sampling_cache(model, 2048) as cache:
            actual = []
            for timestep in range(4):
                for conditioning in (context, negative):
                    sample = SparseValue(noise, coordinates) if sparse else noise
                    actual.append(model(sample, timestep, conditioning))
            assert cache.stats["hits"] > 0 and cache.stats["peak_bytes"] <= 2048
            for result, reference in zip(actual, expected):
                assert torch.equal(result.feats if sparse else result, reference.feats if sparse else reference)
            context["global"].add_(.5)
            sample = SparseValue(noise, coordinates) if sparse else noise
            changed = model(sample, 2, context)
            uncached = original(sample, 2, context)
            assert torch.equal(changed.feats if sparse else changed, uncached.feats if sparse else uncached)
    assert "forward" not in vars(model) and not cache.contexts and cache.used_bytes == 0


def test_feature_cache_budget_and_output_ownership():
    model = nn.Linear(3, 3).eval()
    cache = FeatureCache(48)
    image = torch.ones(2, 3)
    with torch.inference_mode():
        first = cache.get(model, image, lambda: image + 1)
        first.zero_()
        assert torch.equal(cache.get(model, image, lambda: image + 1), image + 1)
        cache.get(model, image * 2, lambda: image + 2)
        assert cache.stats["evictions"] == 1 and cache.stats["peak_bytes"] <= 48
        cache.get(model, torch.ones(4, 3), lambda: torch.ones(4, 3))
        assert cache.stats["oversized"] == 1


@pytest.mark.parametrize("values", [{"dino_cache_mb": -1}, {"conditioning_cache_mb": 2049},
                                    {"conditioning_cache_mb": True}, {"share_dino": 1}, {"sparse_projection": "yes"}])
def test_optimization_options_reject_invalid_values(values):
    with pytest.raises(ValueError):
        OptimizationOptions(**values)
    assert asdict(OptimizationOptions.disabled()) == {"share_dino": False, "dino_cache_mb": 0,
                                                    "sparse_projection": False, "conditioning_cache_mb": 0}


def test_active_projection_restores_original_grid_on_error(upstream):
    model, pipeline = extractor(upstream), pipe(upstream, False, True)
    original_grid = model.proj_grid
    points = original_grid.grid_points.clone()
    def fail(*arguments, **options):
        raise RuntimeError("controlled extractor failure")
    model.forward = fail
    optimized = active_projection(pipeline, SparseValue, pipeline.get_proj_cond_shape, False, [])
    with pytest.raises(RuntimeError, match="controlled"):
        optimized(model, [], torch.tensor([[0, 1, 2, 3]]), grid_resolution_override=6)
    assert model.proj_grid is original_grid and model.grid_resolution == 4
    assert torch.equal(points, model.proj_grid.grid_points)


def test_sampler_hooks_release_cache_before_offload_and_restore_on_error(upstream):
    model = ToyFlow(upstream.ProjectAttention, False, torch.float32)
    context = {"global": torch.rand(1, 5, 6), "proj": torch.rand(1, 7, 6)}
    noise = torch.rand(1, 7, 8)
    class Sampler:
        def sample(self, model, noise, fail=False):
            for timestep in range(3):
                result = model(noise, timestep, context)
            if fail:
                raise RuntimeError("controlled sampler failure")
            return result
    sampler = Sampler()
    pipeline = SimpleNamespace(models={"shape_hr": model}, sparse_structure_sampler=sampler,
                               shape_slat_sampler=sampler, tex_slat_sampler=sampler)
    options = OptimizationOptions(False, 0, False, 1)
    with torch.inference_mode(), OptimizationSession(options, TinyBackbone, upstream.DinoV3ProjFeatureExtractor, SparseValue) as session:
        session.install(pipeline, False)
        sampler.sample(model, noise)
        with pytest.raises(RuntimeError, match="controlled sampler"):
            sampler.sample(model, noise, fail=True)
        for record in session.report()["flow_caches"]:
            assert record["model"] == "shape_hr" and record["hits"] > 0
            assert record["snapshot_bytes"] == record["output_bytes"] == 0
        assert "forward" not in vars(model)
    assert "sample" not in vars(sampler)


def test_native_reference_comparison_checks_settings_structure_and_finiteness(tmp_path):
    from scripts.compare_native_references import compare
    reference = {"schema": "pixalcuda.native-reference.v1", "identity": {"seed": 42, "optimizations": {}},
                 "actual_geometry_resolution": 1024,
                 "shape": {"features": torch.ones(2, 3), "coordinates": torch.zeros(2, 4, dtype=torch.int32)},
                 "texture": {"features": torch.ones(2, 3), "coordinates": torch.zeros(2, 4, dtype=torch.int32)},
                 "mesh": {"vertices": torch.ones(2, 3), "faces": torch.zeros(2, 3, dtype=torch.int64),
                          "coords": torch.zeros(2, 3, dtype=torch.int32), "attrs": torch.ones(2, 6)}}
    baseline, candidate = tmp_path / "baseline.pt", tmp_path / "candidate.pt"
    torch.save(reference, baseline)
    reference["identity"]["optimizations"] = asdict(OptimizationOptions())
    torch.save(reference, candidate)
    assert compare(baseline, candidate)["passed"]
    reference["mesh"]["faces"][0, 0] = 1
    torch.save(reference, candidate)
    assert not compare(baseline, candidate)["passed"]
    reference["identity"]["seed"] = 1
    torch.save(reference, candidate)
    with pytest.raises(ValueError, match="settings differ"):
        compare(baseline, candidate)
