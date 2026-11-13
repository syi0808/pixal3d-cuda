import json
from pathlib import Path

from PIL import Image
import pytest
import torch

from pixalcuda.cameras import orbit_camera_metadata
from pixalcuda.mesh import decoder_output_to_mesh
from pixalcuda.pipeline import Pipeline, dense_coordinates, finish_export
from pixalcuda.runtime import Runtime


class ControlledStore:
    directory = None

    def checkpoint(self, name):
        return Path(name)

    def dino(self):
        return Path("dino")

    def config(self):
        normalization = {"mean": [0.] * 32, "std": [1.] * 32}
        return {"shape_slat_normalization": normalization, "tex_slat_normalization": normalization,
                "sparse_structure_sampler": {"params": {"steps": 12}},
                "shape_slat_sampler": {"params": {"steps": 12}},
                "tex_slat_sampler": {"params": {"steps": 12}}}


@pytest.mark.parametrize("input_mode", ["single", "directory", "mapped"])
@pytest.mark.parametrize("defer_export", [False, True])
@pytest.mark.parametrize("geometry_resolution", [512, 1024])
def test_complete_pipeline_wiring_and_resume_with_controlled_model_outputs(tmp_path, monkeypatch, input_mode, defer_export, geometry_resolution):
    multiview = input_mode != "single"
    calls = []
    conditioning_calls = []
    decoded_coordinates = dense_coordinates(3, "cpu")
    def extract(attributes, coordinates, resolution):
        assert resolution == geometry_resolution
        return decoder_output_to_mesh(attributes, coordinates, resolution=resolution)
    monkeypatch.setattr("pixalcuda.pipeline.decoder_output_to_mesh", extract)
    def run_model(runtime, factory, path, callback, **options):
        name = Path(path).name
        calls.append(name)
        if name != "structure_decoder":
            assert options["dtype"] == torch.float32
        if name == "dino":
            assert options["dtype"] == torch.float32
            return callback(lambda image: torch.zeros((1, (image.shape[-1] // 16) ** 2 + 5, 1024)))
        if name == "structure_decoder":
            def occupancy(latent):
                output = torch.full((1, 1, 64, 64, 64), -1.)
                output[:, :, 16:20, 16:20, 16:20] = 1
                return output
            return callback(occupancy)
        if name in ("shape_decoder", "texture_decoder"):
            def decode(features, coordinates, coordinates_only=False, guides=None):
                if coordinates_only:
                    output = coordinates.clone()
                    output[:, 1:] *= 16
                    return output
                if name == "texture_decoder":
                    assert guides is not None
                    return torch.zeros((len(decoded_coordinates), 6)), decoded_coordinates, guides
                output = torch.zeros((len(decoded_coordinates), 7))
                output[:, 3:6] = 1
                return output, decoded_coordinates, [torch.ones((len(coordinates), 8), dtype=torch.bool)]
            return callback(decode)
        def flow(sample, timestep, context, phases, prepared, negative=False, concat=None):
            return torch.zeros_like(sample)
        if name == "structure":
            return torch.zeros(1, 8, 16, 16, 16)
        return torch.zeros(8, 32)
    def condition(pipeline, encoded, image, size, coordinates, camera, naf_target=512, latent_resolution=32):
        conditioning_calls.append((size, naf_target, latent_resolution))
        if multiview:
            encoded = encoded[0]
        return {"global": encoded[:, :5], "proj": torch.zeros(1, len(coordinates), 2048)}
    monkeypatch.setattr(Runtime, "run_model", run_model)
    monkeypatch.setattr(Pipeline, "condition", condition)
    image = tmp_path / "input.png"
    Image.new("RGB", (32, 32), "red").save(image)
    output = tmp_path / "run"
    store = ControlledStore()
    store.multiview = multiview
    pipeline = Pipeline("cpu", store=store)
    options = {"texture_size": 512, "camera": "fixed", "remove_background": False, "defer_export": defer_export,
               "geometry_resolution": geometry_resolution}
    if multiview:
        metadata = orbit_camera_metadata()
        if input_mode == "mapped":
            metadata["frames"] = [
                {"file_path": f"view{index:02d}.png", "image_index": index, "azimuth_degrees": index * 72,
                 "elevation_degrees": 0, "distance": 3.119205, "fov_degrees": 20} for index in range(5)]
        for frame in metadata["frames"]:
            Image.new("RGBA", (32, 32), (255, 0, 0, 128)).save(tmp_path / frame["file_path"])
        (tmp_path / "transforms.json").write_text(json.dumps(metadata))
        options["camera"] = "auto"
        if input_mode == "mapped":
            options.update(images=[tmp_path / frame["file_path"] for frame in metadata["frames"]],
                           camera_metadata={**metadata, "frames": list(reversed(metadata["frames"]))})
        else:
            options["views_dir"] = tmp_path
    result = pipeline.generate(None if multiview else image, output, **options)
    assert result.is_file()
    if defer_export:
        assert not (output / "result.glb").exists()
        assert json.loads((output / "report.json").read_text())["status"] == "awaiting_export"
        def forbid_cuda(*args, **kwargs):
            raise AssertionError("CPU export must not initialize or synchronize CUDA")
        monkeypatch.setattr(torch.cuda, "synchronize", forbid_cuda)
        monkeypatch.setattr(torch.cuda, "empty_cache", forbid_cuda)
        result = finish_export(output)
        assert result.name == "result.glb"
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "complete"
    assert report["validation"]["embedded_pbr"]
    assert report["validation"]["geometry_resolution"] == geometry_resolution
    assert report["identity"]["geometry_resolution"] == geometry_resolution
    assert report["active_counts"]["shape_voxels"] == len(decoded_coordinates)
    assert report["active_counts"]["texture_voxels"] == len(decoded_coordinates)
    assert len(report["stages"]) == (12 if multiview else 11)
    assert report["identity"]["conditioning_mode"] == ("multiview_average" if multiview else "single")
    assert (512, 512, 32) in conditioning_calls
    assert (1024, 512, geometry_resolution // 16) in conditioning_calls
    assert ((1024, 1024, geometry_resolution // 16) in conditioning_calls) == multiview
    assert "shape_hr" in calls and "texture" in calls
    calls.clear()
    pipeline.generate(None if multiview else image, output, **options, resume=True)
    if defer_export:
        finish_export(output)
    assert not calls
    report = json.loads((output / "report.json").read_text())
    assert "mesh_cleanup" in report
    with pytest.raises(ValueError, match="changed"):
        pipeline.generate(None if multiview else image, output, **{**options, "geometry_resolution": 1536 - geometry_resolution}, resume=True)
    if multiview:
        Image.new("RGB", (32, 32), "blue").save(tmp_path / metadata["frames"][-1]["file_path"])
        with pytest.raises(ValueError, match="changed"):
            pipeline.generate(None, output, **options, resume=True)


@pytest.mark.parametrize("resolution", [256, 2048, True, 1024.0])
def test_unsupported_geometry_is_rejected_before_loading_models(tmp_path, resolution):
    with pytest.raises(ValueError, match="Geometry resolution"):
        Pipeline("cpu", store=ControlledStore()).generate(tmp_path / "missing.png", tmp_path / "run", geometry_resolution=resolution)
