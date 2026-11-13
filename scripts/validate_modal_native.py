"""Explicit, bounded authenticated HTTP acceptance for an already deployed Modal app."""

import argparse
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse
import uuid

import httpx

from pixalcuda.pipeline import validate_pbr_glb
from pixalcuda.runtime import atomic_json
from pixalcuda.weights import sha256_file


def proxy_headers(env_file):
    values = {}
    path = Path(env_file)
    if path.exists():
        for line in path.read_text().splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() in ("PIXAL_PROXY_KEY", "PIXAL_PROXY_SECRET"):
                values[name.strip()] = value.strip().strip("\"'")
    for name in ("PIXAL_PROXY_KEY", "PIXAL_PROXY_SECRET"):
        values[name] = os.environ.get(name) or values.get(name)
        if not values[name]:
            raise ValueError(f"Missing {name}; configure credentials without printing them")
    return {"Modal-Key": values["PIXAL_PROXY_KEY"], "Modal-Secret": values["PIXAL_PROXY_SECRET"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--profiles", nargs="+", choices=("optimized", "reference"), default=["optimized", "reference"])
    parser.add_argument("--geometry-resolution", type=int, choices=(1024, 1536), default=1024)
    parser.add_argument("--texture-size", type=int, choices=(1024, 4096), default=4096)
    parser.add_argument("--remove-background", action="store_true", help="Requires server allow_rmbg and separate RMBG-2.0 license review")
    parser.add_argument("--timeout-seconds", type=int, default=2100)
    parser.add_argument("--execute", action="store_true", help="Actually upload and generate; may consume Modal credits")
    arguments = parser.parse_args(argv)
    url = urlparse(arguments.base_url)
    if url.scheme != "https" or not url.hostname or not url.hostname.endswith(".modal.run") or url.username or url.query or url.fragment:
        parser.error("Use the HTTPS modal.run URL returned by the deployed API function")
    if not 1 <= arguments.timeout_seconds <= 7200 or len(set(arguments.profiles)) != len(arguments.profiles):
        parser.error("Use unique profiles and a timeout between 1 and 7200 seconds")
    plan = {"execute": arguments.execute, "base_url": arguments.base_url.rstrip("/"),
            "image_sha256": sha256_file(arguments.image), "profiles": arguments.profiles,
            "geometry_resolution": arguments.geometry_resolution, "texture_size": arguments.texture_size,
            "target_faces": 1000000, "seed": 42, "sequential": True, "remove_background": arguments.remove_background}
    print(json.dumps(plan, indent=2), flush=True)
    if not arguments.execute:
        return plan
    if arguments.output_dir.exists():
        raise ValueError("Use a fresh acceptance output directory")
    headers = proxy_headers(arguments.env_file)
    arguments.output_dir.mkdir(parents=True)
    atomic_json(arguments.output_dir / "plan.json", plan)
    base = plan["base_url"]
    attempt = f"native-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    results = []
    with httpx.Client(timeout=120, follow_redirects=False) as client:
        unauthenticated = client.get(base + "/api/health")
        if unauthenticated.status_code == 200:
            raise RuntimeError("API is unexpectedly accessible without Proxy authentication")
        health = client.get(base + "/api/health", headers=headers)
        health.raise_for_status()
        metadata = health.json()
        if metadata.get("runtime") != "official" or metadata.get("gpu") != "L4":
            raise RuntimeError("Endpoint does not identify the declared official L4 runtime")
        if arguments.remove_background and not metadata.get("rmbg_enabled", False):
            raise RuntimeError("RMBG-2.0 is disabled on the server; no generation submitted")
        atomic_json(arguments.output_dir / "health.json", {"unauthenticated_status": unauthenticated.status_code, **metadata})
        for profile in arguments.profiles:
            destination = arguments.output_dir / profile
            destination.mkdir()
            options = {"seed": 42, "geometry_resolution": arguments.geometry_resolution, "texture_size": arguments.texture_size,
                       "target_faces": 1000000, "reference": profile == "reference", "attempt": attempt + "-" + profile,
                       "remove_background": arguments.remove_background}
            started = time.monotonic()
            with arguments.image.open("rb") as source:
                submitted = client.post(base + "/api/jobs", headers=headers,
                    files={"image": (arguments.image.name, source, "image/png")}, data={"options": json.dumps(options)})
            submitted.raise_for_status()
            record = submitted.json()
            atomic_json(destination / "submitted.json", record)
            previous = None
            while True:
                response = client.get(base + record["status_url"], headers=headers)
                response.raise_for_status()
                status = response.json()
                atomic_json(destination / "status.json", status)
                if status["status"] != previous:
                    print(json.dumps({"profile": profile, "job_id": status["job_id"], "status": status["status"]}), flush=True)
                    previous = status["status"]
                if status["status"] == "failed":
                    raise RuntimeError(f"Native job failed; inspect {destination / 'status.json'} and Modal logs. No retry submitted.")
                if status["status"] == "complete":
                    break
                if time.monotonic() - started > arguments.timeout_seconds:
                    raise TimeoutError("Client stopped waiting; inspect/cancel the existing job before any resubmission")
                time.sleep(10)
            wall_seconds = time.monotonic() - started
            for filename, field in (("report.json", "report_url"), ("result.glb", "glb_url")):
                with client.stream("GET", base + record[field], headers=headers) as artifact:
                    artifact.raise_for_status()
                    with (destination / filename).open("wb") as output:
                        for chunk in artifact.iter_bytes():
                            output.write(chunk)
            validation = validate_pbr_glb(destination / "result.glb")
            report = json.loads((destination / "report.json").read_text())
            if report.get("status") != "complete" or report.get("validation") != validation or report.get("actual_geometry_resolution") != arguments.geometry_resolution:
                raise ValueError("Downloaded GLB/report does not match the requested native completion")
            results.append({"profile": profile, "job_id": record["job_id"], "client_completion_seconds": wall_seconds,
                            "stages": report["stages"], "validation": validation, "optimizations": report["optimizations"]})
            atomic_json(arguments.output_dir / "acceptance.json", {"plan": plan, "results": results})
    return results


if __name__ == "__main__":
    main()
