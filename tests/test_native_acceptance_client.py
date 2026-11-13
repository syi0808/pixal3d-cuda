import json

import httpx
import pytest
from PIL import Image

from scripts.validate_modal_native import main, proxy_headers


def test_native_acceptance_dry_run_does_not_read_credentials_or_contact_modal(tmp_path, monkeypatch, capsys):
    image = tmp_path / "input.png"
    Image.new("RGB", (16, 16), "red").save(image)
    def forbidden(*arguments, **options):
        raise AssertionError("Dry-run must not contact the app or read credentials")
    monkeypatch.setattr("scripts.validate_modal_native.proxy_headers", forbidden)
    monkeypatch.setattr("scripts.validate_modal_native.httpx.Client", forbidden)
    result = main(["--base-url", "https://example.modal.run", "--image", str(image), "--output-dir", str(tmp_path / "run")])
    assert not result["execute"] and not (tmp_path / "run").exists()
    assert "Modal-Secret" not in capsys.readouterr().out


def test_proxy_credentials_are_loaded_without_printing(tmp_path, monkeypatch, capsys):
    path = tmp_path / ".env"
    path.write_text("PIXAL_PROXY_KEY=fake-key\nPIXAL_PROXY_SECRET='fake-secret'\nUNRELATED=ignored\n")
    monkeypatch.delenv("PIXAL_PROXY_KEY", raising=False)
    monkeypatch.delenv("PIXAL_PROXY_SECRET", raising=False)
    assert proxy_headers(path) == {"Modal-Key": "fake-key", "Modal-Secret": "fake-secret"}
    assert not capsys.readouterr().out


@pytest.mark.parametrize("url", ["http://example.modal.run", "https://example.com", "https://example.modal.run@evil.test"])
def test_native_client_rejects_other_credential_destinations(url, tmp_path):
    with pytest.raises(SystemExit):
        main(["--base-url", url, "--image", "not-opened", "--output-dir", str(tmp_path)])


@pytest.mark.parametrize("failed", [False, True])
def test_native_acceptance_runs_sequentially_and_never_retries(tmp_path, monkeypatch, failed):
    image = tmp_path / "input.png"
    Image.new("RGB", (16, 16), "red").save(image)
    submissions = []
    completed = []
    validation = {"embedded_pbr": True, "sha256": "controlled"}
    def respond(request):
        if request.url.path == "/api/health":
            if "Modal-Key" not in request.headers:
                return httpx.Response(401)
            return httpx.Response(200, json={"runtime": "official", "gpu": "L4"})
        assert request.headers["Modal-Key"] == "fake-key"
        if request.method == "POST":
            assert len(submissions) == len(completed)
            reference = bool(submissions)
            assert f'"reference": {str(reference).lower()}'.encode() in request.content
            job_id = str(len(submissions)) * 64
            submissions.append(job_id)
            return httpx.Response(202, json={"job_id": job_id, "status_url": f"/api/jobs/{job_id}",
                "report_url": f"/api/jobs/{job_id}/report.json", "glb_url": f"/api/jobs/{job_id}/result.glb"})
        job_id = request.url.path.split("/")[3]
        if request.url.path.endswith("report.json"):
            completed.append(job_id)
            return httpx.Response(200, json={"status": "complete", "validation": validation,
                "actual_geometry_resolution": 1024, "stages": [], "optimizations": {}})
        if request.url.path.endswith("result.glb"):
            return httpx.Response(200, content=b"controlled-glb")
        return httpx.Response(200, json={"job_id": job_id, "status": "failed" if failed else "complete"})
    client_class = httpx.Client
    monkeypatch.setattr("scripts.validate_modal_native.httpx.Client",
                        lambda **options: client_class(transport=httpx.MockTransport(respond), **options))
    monkeypatch.setattr("scripts.validate_modal_native.proxy_headers", lambda path: {"Modal-Key": "fake-key"})
    monkeypatch.setattr("scripts.validate_modal_native.validate_pbr_glb", lambda path: validation)
    arguments = ["--base-url", "https://example.modal.run", "--image", str(image),
                 "--output-dir", str(tmp_path / "run"), "--execute"]
    if failed:
        with pytest.raises(RuntimeError, match="No retry"):
            main(arguments)
        assert len(submissions) == 1 and not completed
    else:
        results = main(arguments)
        assert [result["profile"] for result in results] == ["optimized", "reference"]
        assert submissions == completed and len(submissions) == 2
        assert len(json.loads((tmp_path / "run" / "acceptance.json").read_text())["results"]) == 2
