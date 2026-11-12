import copy
import json
from types import SimpleNamespace

from PIL import Image
import pytest

pytest.importorskip("modal")

from deploy import modal_app
from pixalcuda.cameras import orbit_camera_metadata
from pixalcuda.preprocess import prepare_foreground, prepare_image


def test_modal_request_cache_covers_all_images_cameras_settings_and_source(tmp_path, monkeypatch):
    metadata = orbit_camera_metadata()
    for frame in metadata["frames"]:
        Image.new("RGBA", (16, 16), (255, 0, 0, 180)).save(tmp_path / frame["file_path"])
    path = tmp_path / "transforms.json"
    path.write_text(json.dumps(metadata))
    options = dict(image_path="", views_dir=tmp_path, seed=42, texture_size=1024, target_faces=100000,
                   camera="auto", fov_degrees=49.13, no_rembg=True, attempt="")
    original, files = modal_app.build_request(**options)
    assert len(files) == 4
    assert original["multiview"]
    assert all(".." not in record["name"] for record in original["inputs"])
    assert modal_app.fingerprint(original) == modal_app.fingerprint(modal_app.build_request(**options)[0])
    for update in ({"seed": 1}, {"texture_size": 512}, {"attempt": "retry-1"}, {"no_rembg": False}):
        changed, _ = modal_app.build_request(**{**options, **update})
        assert modal_app.fingerprint(changed) != modal_app.fingerprint(original)
    altered = copy.deepcopy(metadata)
    altered["frames"][-1]["transform_matrix"][0][3] -= .1
    path.write_text(json.dumps(altered))
    assert modal_app.fingerprint(modal_app.build_request(**options)[0]) != modal_app.fingerprint(original)
    path.write_text(json.dumps(metadata))
    Image.new("RGB", (16, 16), "blue").save(files[-1][0])
    assert modal_app.fingerprint(modal_app.build_request(**options)[0]) != modal_app.fingerprint(original)
    monkeypatch.setattr(modal_app, "source_identity", lambda: {"changed": "source"})
    assert modal_app.fingerprint(modal_app.build_request(**options)[0]) != modal_app.fingerprint(original)


def test_modal_single_image_and_validation(tmp_path):
    path = tmp_path / "input.png"
    Image.new("RGBA", (40, 20), (20, 40, 100, 255)).save(path)
    options = dict(image_path=path, views_dir="", seed=42, texture_size=1024, target_faces=100000,
                   camera="fixed", fov_degrees=49.13, no_rembg=True, attempt="")
    request, files = modal_app.build_request(**options)
    assert not request["multiview"] and request["cameras"] is None
    assert files == [(path, "input.png")]
    for changed in ({"views_dir": tmp_path}, {"image_path": ""}, {"texture_size": 7}, {"target_faces": 0},
                    {"camera": "guess"}, {"fov_degrees": float("nan")}):
        with pytest.raises(ValueError):
            modal_app.build_request(**{**options, **changed})
    for invalid in ("../job", "z" * 64, "a" * 63):
        with pytest.raises(ValueError, match="Invalid job"):
            modal_app.job_path(invalid)


def test_cpu_matting_boundary_does_not_double_crop(tmp_path):
    source = tmp_path / "source.png"
    image = Image.new("RGBA", (80, 40), (0, 0, 0, 0))
    image.paste((200, 50, 10, 255), (8, 5, 35, 30))
    image.save(source)
    prepared = tmp_path / "prepared.png"
    prepare_foreground(source, False).save(prepared)
    direct = prepare_image(source, False)
    separated = prepare_image(prepared, False)
    assert direct.size == separated.size
    assert direct.tobytes() == separated.tobytes()


@pytest.mark.parametrize("multiview", [False, True])
@pytest.mark.filterwarnings("ignore:The .* function is executing locally:UserWarning")
def test_private_job_deduplication_and_phase_handoff(tmp_path, monkeypatch, multiview):
    volumes = SimpleNamespace(reload=lambda: None, commit=lambda: None)
    class Records(dict):
        def put(self, key, value, skip_if_exists=False):
            if skip_if_exists and key in self:
                return False
            self[key] = value
            return True
    records = Records()
    monkeypatch.setattr(modal_app, "records", records)
    monkeypatch.setattr(modal_app, "jobs", volumes)
    monkeypatch.setattr(modal_app, "weights", volumes)
    monkeypatch.setattr(modal_app, "JOBS_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(modal_app, "WEIGHTS_ROOT", tmp_path)
    source = tmp_path / "input.png"
    Image.new("RGBA", (32, 24), (20, 40, 100, 180)).save(source)
    if multiview:
        metadata = orbit_camera_metadata()
        for frame in metadata["frames"]:
            Image.new("RGBA", (32, 24), (20, 40, 100, 180)).save(tmp_path / frame["file_path"])
        (tmp_path / "transforms.json").write_text(json.dumps(metadata))
    request, files = modal_app.build_request("" if multiview else source, tmp_path if multiview else "",
                                            42, 1024, 100000, "auto", 49.13, True, "")
    job_id = modal_app.fingerprint(request)
    directory = modal_app.job_path(job_id)
    (directory / "upload").mkdir(parents=True)
    for path, name in files:
        (directory / "upload" / name).write_bytes(path.read_bytes())
    (tmp_path / ("ready-mv.json" if multiview else "ready-sv.json")).write_text(json.dumps(
        {"revision": request["revision"], "rembg": False, "camera": True}))
    calls = []
    infer = modal_app.infer_job
    export = modal_app.export_job
    monkeypatch.setattr(modal_app, "infer_job", SimpleNamespace(spawn=lambda identifier: calls.append(identifier) or SimpleNamespace(object_id="gpu-call")))
    result = modal_app.prepare_job.local(job_id, request)
    assert result["status"] == "queued_gpu" and calls == [job_id]
    assert modal_app.prepare_job.local(job_id, request)["status"] == "queued_gpu"
    assert calls == [job_id]
    pipeline_calls = []
    class Pipeline:
        def __init__(self, device):
            assert device == "cuda"

        def generate(self, image, output, **options):
            pipeline_calls.append((image, output, options))
            assert options["defer_export"] and not options["remove_background"]
            assert bool(options.get("views_dir")) == multiview
    monkeypatch.setattr("pixalcuda.pipeline.Pipeline", Pipeline)
    monkeypatch.setattr(modal_app, "export_job", SimpleNamespace(spawn=lambda identifier: calls.append(identifier) or SimpleNamespace(object_id="export-call")))
    assert infer.local(job_id)["status"] == "queued_export"
    assert len(pipeline_calls) == 1
    def finish(output):
        output.mkdir()
        result = output / "result.glb"
        result.write_bytes(b"controlled export output; not a real GLB")
        return result
    monkeypatch.setattr("pixalcuda.pipeline.finish_export", finish)
    assert export.local(job_id)["status"] == "complete"
    assert records[job_id]["bytes"] > 0
    assert modal_app.prepare_job.local(job_id, request)["status"] == "complete"
    assert len(pipeline_calls) == 1
