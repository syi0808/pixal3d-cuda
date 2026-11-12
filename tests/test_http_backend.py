from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

from PIL import Image
import pytest

pytest.importorskip("modal")
pytest.importorskip("fastapi")

from deploy import modal_app
from deploy.http_api import DailyLimitExceeded, JobOptions


class Records(dict):
    def put(self, key, value, skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key] = value
        return True


def test_api_daily_quota_is_persistent_atomic_and_deduplicated(monkeypatch):
    records = Records()
    monkeypatch.setattr(modal_app, "records", records)
    monkeypatch.setattr(modal_app, "SETTINGS", replace(modal_app.SETTINGS, api_max_jobs_per_day=2))
    now = datetime(2026, 11, 12, 23, 59, 20, tzinfo=timezone.utc)
    modal_app.reserve_api_quota("job-a", now)
    modal_app.reserve_api_quota("job-a", now)
    assert len(records) == 1
    modal_app.reserve_api_quota("job-b", now)
    with pytest.raises(DailyLimitExceeded) as error:
        modal_app.reserve_api_quota("job-c", now)
    assert error.value.retry_after == 40
    modal_app.reserve_api_quota("job-c", now + timedelta(days=1))
    assert len(records) == 3


@pytest.mark.parametrize("multiview", [False, True])
def test_http_backend_commits_inputs_before_spawning_and_reuses_job(tmp_path, monkeypatch, multiview):
    records, events = Records(), []
    monkeypatch.setattr(modal_app, "records", records)
    monkeypatch.setattr(modal_app, "JOBS_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(modal_app, "jobs", SimpleNamespace(reload=lambda: events.append("reload"), commit=lambda: events.append("commit")))
    def spawn(job_id, request):
        assert events[-1] == "commit"
        assert records[job_id]["status"] == "submitted"
        assert request["multiview"] == multiview
        for record in request["inputs"]:
            assert (modal_app.job_path(job_id) / "upload" / record["name"]).exists()
        events.append("spawn")
    monkeypatch.setattr(modal_app, "prepare_job", SimpleNamespace(spawn=spawn))
    images = []
    for index in range(3 if multiview else 1):
        image = tmp_path / f"photo-{index}.png"
        Image.new("RGBA", (16, 24), (index, 80, 120, 180)).save(image)
        images.append(image)
    cameras = {"frames": [{"image_index": index, "azimuth_degrees": index * 120,
                            "elevation_degrees": 0, "distance": 3.119205, "fov_degrees": 20}
                           for index in reversed(range(len(images)))]} if multiview else None
    backend = modal_app.HttpBackend()
    options = JobOptions(remove_background=False).model_dump()
    first = backend.submit(images, cameras, options)
    assert first["status"] == "submitted"
    assert backend.submit(images, cameras, options) == first
    assert events.count("spawn") == 1
    assert len([key for key in records if key.startswith("api-quota:")]) == 1
    job_id = first["job_id"]
    directory = modal_app.job_path(job_id) / "run"
    directory.mkdir()
    (directory / "result.glb").write_bytes(b"controlled transport bytes; not a generated GLB")
    (directory / "report.json").write_text(json.dumps({"status": "complete"}))
    records[job_id] = {**first, "status": "complete"}
    destination = tmp_path / "download.glb"
    backend.copy_artifact(job_id, "result.glb", destination)
    assert destination.read_bytes() == (directory / "result.glb").read_bytes()
    with pytest.raises(ValueError, match="Unsupported"):
        backend.copy_artifact(job_id, "../request.json", destination)
    assert backend.status(job_id)["status"] == "complete"
