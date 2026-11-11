import json
from pathlib import Path

from PIL import Image
import torch

from pixalcuda.pipeline import Pipeline, dense_coordinates
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


def test_complete_pipeline_wiring_and_resume_with_controlled_model_outputs(tmp_path, monkeypatch):
    calls = []
    decoded_coordinates = dense_coordinates(3, "cpu")
    def run_model(runtime, factory, path, callback, **options):
        name = Path(path).name
        calls.append(name)
        if name == "dino":
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
    def condition(pipeline, encoded, image, size, coordinates, camera):
        return {"global": encoded[:, :5], "proj": torch.zeros(1, len(coordinates), 2048)}
    monkeypatch.setattr(Runtime, "run_model", run_model)
    monkeypatch.setattr(Pipeline, "condition", condition)
    image = tmp_path / "input.png"
    Image.new("RGB", (32, 32), "red").save(image)
    output = tmp_path / "run"
    pipeline = Pipeline("cpu", store=ControlledStore())
    result = pipeline.generate(image, output, texture_size=512, camera="fixed", remove_background=False)
    assert result.is_file()
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "complete"
    assert report["validation"]["embedded_pbr"]
    assert len(report["stages"]) == 11
    assert "shape_hr" in calls and "texture" in calls
    calls.clear()
    pipeline.generate(image, output, texture_size=512, camera="fixed", remove_background=False, resume=True)
    assert not calls
