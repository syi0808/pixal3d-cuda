import copy
from dataclasses import replace
import json
from pathlib import Path
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
    for update in ({"seed": 1}, {"texture_size": 512}, {"attempt": "retry-1"}, {"no_rembg": False},
                   {"geometry_resolution": 1024}, {"conditioning_cache_mb": 512}, {"conditioning_cache_mb": 1024}):
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
                    {"camera": "guess"}, {"fov_degrees": float("nan")}, {"geometry_resolution": 256},
                    {"geometry_resolution": True}, {"conditioning_cache_mb": -1},
                    {"conditioning_cache_mb": 2049}, {"conditioning_cache_mb": True}):
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


@pytest.mark.parametrize("view_count", [0, 1, 3, 5, 16])
@pytest.mark.parametrize("submitted", [False, True])
@pytest.mark.filterwarnings("ignore:The .* function is executing locally:UserWarning")
def test_private_job_deduplication_and_phase_handoff(tmp_path, monkeypatch, view_count, submitted):
    multiview = view_count > 0
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
        metadata = {"frames": [
            {"file_path": f"image-{index}.png", "image_index": index, "azimuth_degrees": index * 360 / view_count,
             "elevation_degrees": 0, "distance": 3.119205, "fov_degrees": 20} for index in range(view_count)]}
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
    if submitted:
        records[job_id] = {"job_id": job_id, "status": "submitted"}
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
        def __init__(self, device, limits):
            assert device == "cuda"
            assert limits.conditioning_cache_mb == request["options"]["conditioning_cache_mb"]

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


def test_modal_geometry_profiles_keep_guards_explicit_and_cache_optional():
    low = modal_app.inference_limits(512, 0)
    high = modal_app.inference_limits(1024, 1024)
    assert low.max_tokens == 16384 and low.max_voxels == 1500000 and low.conditioning_cache_mb == 0
    assert high.max_tokens == modal_app.SETTINGS.geometry_1024_max_tokens
    assert high.max_voxels == modal_app.SETTINGS.geometry_1024_max_voxels
    assert high.conditioning_cache_mb == 1024


def test_modal_mapped_images_preserve_index_mapping_and_ignore_client_paths(tmp_path):
    images = []
    for index in range(3):
        path = tmp_path / f"photo-{index}.png"
        Image.new("RGBA", (20, 32), (index, 80, 120, 180)).save(path)
        images.append(path)
    metadata = {"frames": [
        {"image_index": index, "file_path": "../../ignored.png", "azimuth_degrees": index * 120,
         "elevation_degrees": 0, "distance": 3.119205, "fov_degrees": 20} for index in reversed(range(3))]}
    transforms = tmp_path / "cameras.json"
    transforms.write_text(json.dumps(metadata))
    options = dict(image_path="", views_dir="", seed=42, texture_size=1024, target_faces=100000,
                   camera="auto", fov_degrees=49.13, no_rembg=True, attempt="", images=images, transforms=transforms)
    request, files = modal_app.build_request(**options)
    assert [path for path, _ in files] == images
    assert [frame["image_index"] for frame in request["cameras"]["frames"]] == [0, 1, 2]
    assert [frame["file_path"] for frame in request["cameras"]["frames"]] == ["view00.png", "view01.png", "view02.png"]
    metadata["frames"].reverse()
    transforms.write_text(json.dumps(metadata))
    assert modal_app.fingerprint(modal_app.build_request(**options)[0]) == modal_app.fingerprint(request)
    assert modal_app.fingerprint(modal_app.build_request(**{**options, "images": list(reversed(images))})[0]) != modal_app.fingerprint(request)
    with pytest.raises(ValueError, match="together"):
        modal_app.build_request(**{**options, "transforms": None})


def test_modal_operational_limits_are_distinct_from_model_view_count(tmp_path, monkeypatch):
    files = []
    for index in range(17):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGBA", (20, 32), (index, 80, 120, 180)).save(path)
        files.append((path, f"view{index:02d}.png"))
    with pytest.raises(ValueError, match="1-16"):
        modal_app.validate_image_files(files, True)
    settings = modal_app.SETTINGS
    for limits in ({"max_image_pixels": 100}, {"max_upload_bytes": 1}, {"max_upload_bytes": 1000, "max_total_image_bytes": 1000}):
        monkeypatch.setattr(modal_app, "SETTINGS", replace(settings, **limits))
        with pytest.raises(ValueError, match="limit"):
            modal_app.validate_image_files(files[:16], True)


def test_local_plan_does_not_submit_jobs_or_disclose_secrets(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("Local plan must not look up or invoke remote functions")
    monkeypatch.setattr(modal_app.modal.Function, "from_name", forbidden)
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "never-print-this-value")
    monkeypatch.setenv("MODAL_ENVIRONMENT", "main")
    modal_app.run_client(action="plan")
    output = capsys.readouterr().out
    assert "never-print-this-value" not in output
    assert json.loads(output)["environment"] == "main"
    workflow = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "modal-deploy.yml").read_text()
    assert "workflow_dispatch:" in workflow and "push:" not in workflow and "schedule:" not in workflow
    assert "modal deploy --env main -m deploy.modal_app" in workflow
    assert "inputs.environment" not in workflow and "options: [dev, prod]" not in workflow
