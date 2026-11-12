import ast
from dataclasses import replace
import io
import json
from pathlib import Path

from PIL import Image
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from deploy.http_api import DailyLimitExceeded, create_app
from deploy.settings import DeploySettings


JOB_ID = "a" * 64


def image_bytes(index=0):
    output = io.BytesIO()
    Image.new("RGBA", (16, 24), (index, 80, 120, 180)).save(output, format="PNG")
    return output.getvalue()


def cameras(count):
    return {"frames": [{"image_index": index, "azimuth_degrees": index * 360 / count,
                        "elevation_degrees": 0, "distance": 3.119205, "fov_degrees": 20}
                       for index in reversed(range(count))]}


class Backend:
    def __init__(self):
        self.calls, self.destinations = [], []
        self.records, self.artifacts = {}, {}

    def submit(self, images, transforms, options):
        self.calls.append({"names": [path.name for path in images], "images": [path.read_bytes() for path in images],
                           "transforms": transforms, "options": options})
        self.records[JOB_ID] = {"job_id": JOB_ID, "status": "submitted", "call_id": "internal-only"}
        return self.records[JOB_ID]

    def status(self, job_id):
        return self.records.get(job_id)

    def copy_artifact(self, job_id, filename, destination):
        self.destinations.append(destination)
        if filename not in self.artifacts:
            raise FileNotFoundError(filename)
        destination.write_bytes(self.artifacts[filename])


def test_http_submit_poll_and_download_are_separate_from_generation():
    backend = Backend()
    with TestClient(create_app(backend, DeploySettings())) as client:
        assert client.get("/api/health").json()["environment"] == "main"
        schema = client.get("/openapi.json").json()
        assert schema["security"] == [{"ModalKey": [], "ModalSecret": []}]
        body = schema["paths"]["/api/jobs"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        assert set(body["properties"]) == {"image", "images", "transforms", "options"}
        response = client.post("/api/jobs", files={"image": ("../../private.png", image_bytes(), "image/png")},
                               data={"options": json.dumps({"remove_background": False})})
        assert response.status_code == 202
        assert response.headers["location"] == f"/api/jobs/{JOB_ID}"
        assert response.json()["status"] == "submitted" and "call_id" not in response.json()
        assert backend.calls[0]["names"] == ["image-0.png"]
        assert backend.calls[0]["images"] == [image_bytes()]
        assert client.get(f"/api/jobs/{JOB_ID}").json()["status"] == "submitted"
        assert client.get(f"/api/jobs/{JOB_ID}/result.glb").status_code == 409
        backend.records[JOB_ID]["status"] = "complete"
        backend.artifacts = {"result.glb": b"controlled transport bytes; not a generated GLB", "report.json": b'{"status":"complete"}'}
        glb = client.get(f"/api/jobs/{JOB_ID}/result.glb")
        assert glb.status_code == 200 and glb.content == backend.artifacts["result.glb"]
        assert glb.headers["content-type"] == "model/gltf-binary"
        assert client.get(f"/api/jobs/{JOB_ID}/report.json").json()["status"] == "complete"
        assert not any(path.exists() for path in backend.destinations)
        backend.records[JOB_ID]["status"] = "failed"
        assert client.get(f"/api/jobs/{JOB_ID}/report.json").status_code == 200
        assert client.get(f"/api/jobs/{JOB_ID}/result.glb").status_code == 409
        del backend.artifacts["report.json"]
        assert client.get(f"/api/jobs/{JOB_ID}/report.json").status_code == 404
        backend.records[JOB_ID]["status"] = "complete"
        assert client.get("/api/jobs/not-a-job").status_code == 404
        assert client.get(f"/api/jobs/{'b' * 64}").status_code == 404
        del backend.artifacts["result.glb"]
        assert client.get(f"/api/jobs/{JOB_ID}/result.glb").status_code == 410


@pytest.mark.parametrize("count", [1, 3, 5, 16])
def test_http_images_and_camera_mapping_are_forwarded_in_upload_order(count):
    backend = Backend()
    with TestClient(create_app(backend, DeploySettings())) as client:
        files = [("images", ("same-name.png", image_bytes(index), "image/png")) for index in range(count)]
        metadata = cameras(count)
        response = client.post("/api/jobs", files=files, data={"transforms": json.dumps(metadata)})
        assert response.status_code == 202
        assert backend.calls[0]["images"] == [image_bytes(index) for index in range(count)]
        assert backend.calls[0]["transforms"] == metadata


@pytest.mark.parametrize("options", [{"seed": True}, {"seed": -1}, {"texture_size": 123}, {"target_faces": 0},
                                     {"resolution": 256}, {"remove_background": "false"}, {"attempt": "../retry"}])
def test_http_rejects_invalid_or_unsupported_settings_before_submission(options):
    backend = Backend()
    with TestClient(create_app(backend, DeploySettings())) as client:
        response = client.post("/api/jobs", files={"image": ("input.png", image_bytes(), "image/png")},
                               data={"options": json.dumps(options)})
        assert response.status_code == 422 and not backend.calls


def test_http_requires_unambiguous_single_or_mapped_inputs():
    backend = Backend()
    with TestClient(create_app(backend, DeploySettings())) as client:
        single = ("image", ("input.png", image_bytes(), "image/png"))
        mapped = ("images", ("input.png", image_bytes(), "image/png"))
        for files, data in (([single, mapped], {}), ([single, single], {}), ([mapped], {}),
                            ([single], {"transforms": "{}"}), ([mapped], {"transforms": ""}),
                            ([mapped], {"transforms": json.dumps(cameras(1)), "options": '{"fov_degrees":20}'})):
            assert client.post("/api/jobs", files=files, data=data).status_code == 422
        duplicate = [single, ("options", (None, "{}")), ("options", (None, "{}"))]
        assert client.post("/api/jobs", files=duplicate).status_code == 422
        assert client.post("/api/jobs", files=[single, ("unexpected", (None, "x"))]).status_code == 422
        assert not backend.calls


def test_http_has_bounded_body_uploads_and_daily_admission_errors():
    backend = Backend()
    settings = replace(DeploySettings(), max_upload_bytes=1)
    with TestClient(create_app(backend, settings)) as client:
        assert client.post("/api/jobs", files={"image": ("input.png", image_bytes(), "image/png")}).status_code == 413
        assert client.post("/api/jobs", content=b"", headers={"content-length": "999999999"}).status_code == 413
        assert client.post("/api/jobs", content=b"", headers={"content-length": "invalid"}).status_code == 400
        assert not backend.calls
    def quota(*args):
        raise DailyLimitExceeded(37)
    backend.submit = quota
    with TestClient(create_app(backend, DeploySettings())) as client:
        response = client.post("/api/jobs", files={"image": ("input.png", image_bytes(), "image/png")})
        assert response.status_code == 429 and response.headers["retry-after"] == "37"


def test_chunked_upload_cannot_bypass_total_body_limit():
    backend = Backend()
    settings = replace(DeploySettings(), max_upload_bytes=100, max_total_image_bytes=100)
    body = (b'--boundary\r\nContent-Disposition: form-data; name="image"; filename="image.png"\r\n\r\n'
            + b"x" * (settings.max_transforms_bytes + 128 * 1024 + 101) + b"\r\n--boundary--\r\n")
    with TestClient(create_app(backend, settings)) as client:
        response = client.post("/api/jobs", content=iter([body[:100], body[100:]]),
                               headers={"content-type": "multipart/form-data; boundary=boundary"})
        assert response.status_code == 413 and not backend.calls


@pytest.mark.parametrize("error", [ValueError("Invalid camera mapping"), OSError("Invalid image"),
                                  Image.DecompressionBombError("Image pixel limit exceeded")])
def test_invalid_content_is_not_classified_as_a_server_failure(error):
    backend = Backend()
    def reject(*args):
        raise error
    backend.submit = reject
    with TestClient(create_app(backend, DeploySettings())) as client:
        response = client.post("/api/jobs", files={"image": ("input.png", image_bytes(), "image/png")})
        assert response.status_code == 422


def test_deployed_http_function_requires_edge_auth_and_has_no_gpu():
    source = (Path(__file__).resolve().parents[1] / "deploy" / "modal_app.py").read_text()
    function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "api")
    calls = [node for node in function.decorator_list if isinstance(node, ast.Call)]
    edge = next(call for call in calls if isinstance(call.func, ast.Attribute) and call.func.attr == "asgi_app")
    assert any(keyword.arg == "requires_proxy_auth" and keyword.value.value is True for keyword in edge.keywords)
    worker = next(call for call in calls if isinstance(call.func, ast.Attribute) and call.func.attr == "function")
    assert not any(keyword.arg == "gpu" for keyword in worker.keywords)
