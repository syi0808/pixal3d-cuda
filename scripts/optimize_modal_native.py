"""Explicit sequential L4 validation and 2/4/8-core frozen-fixture comparisons."""

import argparse
import json
from pathlib import Path
import sys
import time
import uuid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-job", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args(argv)
    if len(arguments.source_job) != 64 or any(character not in "0123456789abcdef" for character in arguments.source_job):
        parser.error("Expected completed native job ID")
    plan = {"execute": arguments.execute, "gpu": "L4", "cores": [2, 4, 8], "sequential": True,
            "timeout_seconds_per_call": 2100, "retries": 0, "chart_repeats": 2,
            "inpaint_workers": [1, 2, 4], "inpaint_repeats": 2, "source_job": arguments.source_job}
    print(json.dumps(plan, indent=2), flush=True)
    if not arguments.execute:
        return plan
    if arguments.output_dir.exists():
        raise ValueError("Use a fresh output folder; never duplicate an ambiguous paid call")
    import modal
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from deploy.modal_app import offline, jobs, weights
    from pixalcuda.runtime import atomic_json

    arguments.output_dir.mkdir(parents=True)
    atomic_json(arguments.output_dir / "plan.json", plan)
    app = modal.App("pixal3d-native-tuning")

    def remote_work(source_job, attempt, cores):
        import hashlib
        import json
        import os
        from pathlib import Path
        import sys

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE"] = "0"
        import numpy as np
        from PIL import Image
        from pixalcuda.cpu_tuning import validate_initialization, benchmark_cpu
        from pixalcuda.runtime import atomic_json
        from pixalcuda.upstream_sources import lock, verify_source
        from pixalcuda.upstream_weights import UpstreamStore
        from pixalcuda.upstream_worker import NativeReport, load_entry, pin_naf_loader, require_hardware

        jobs.reload()
        weights.reload()
        directory = Path("/jobs/tuning") / attempt
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / str(cores)
        output.mkdir(exist_ok=False)
        source = Path("/jobs") / source_job
        previous = json.loads((source / "run/report.json").read_text())
        image_path = source / "prepared/input.png"
        image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if previous['status'] != 'complete' or previous['identity']['upstream_lock'] != lock() or previous['identity']['multiview'] or image_hash != previous['identity']['inputs'][0]['sha256']:
            raise ValueError("Invalid native source job")
        report = NativeReport(output, {"upstream_lock": lock(), "image_sha256": image_hash,
            "camera": previous["camera"], "runtime_sources": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path("/opt/pixal3d/pixalcuda").glob("*.py"))}})
        try:
            report.data.update(require_hardware())
            for name in lock()['sources']:
                verify_source(name, '/opt/upstream')
            sys.path.insert(0, '/opt/upstream/pixal3d')
            if cores == 2:
                entry = load_entry('/opt/upstream', False)
                bundle = UpstreamStore().resolve(camera=False, rembg=False)
                pin_naf_loader('/opt/upstream', bundle['naf_weights'])
                for config in entry.IMAGE_COND_CONFIGS.values():
                    config['model_name'] = bundle['dino']
                with Image.open(image_path) as image:
                    rgba = np.asarray(image.convert('RGBA'), dtype=np.float32)
                    prepared = Image.fromarray((rgba[..., :3] * (rgba[..., 3:] / 255)).astype(np.uint8))
                validate_initialization(entry, bundle, prepared, previous['camera'], output, report)
            benchmark_cpu(directory / '2', output, cores, report)
            report.data['status'] = 'complete'
        except BaseException as error:
            report.data.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            report.write()
            jobs.commit()
        return report.data

    functions = {cores: app.function(image=offline, gpu="L4", cpu=(cores, cores), memory=(16384, 65536),
        timeout=2100, retries=0, min_containers=0, max_containers=1, buffer_containers=0,
        scaledown_window=2, include_source=False, serialized=True,
        name=f"tune_{cores}", volumes={"/weights": weights.with_mount_options(read_only=True), "/jobs": jobs})(remote_work)
        for cores in (2, 4, 8)}
    attempt = 'opt-' + uuid.uuid4().hex
    with modal.enable_output(), app.run(environment_name='main'):
        for cores in (2, 4, 8):
            stats = modal.Function.from_name('pixal3d-cuda', 'infer_job', environment_name='main').get_current_stats()
            if stats.backlog or stats.num_running_inputs or stats.num_total_runners:
                raise RuntimeError('Production GPU busy; refusing another allocation')
            call = functions[cores].spawn(arguments.source_job, attempt, cores)
            atomic_json(arguments.output_dir / f'call-{cores}.json', {'call_id': call.object_id, 'attempt': attempt})
            print(json.dumps({'cores': cores, 'call_id': call.object_id, 'attempt': attempt}), flush=True)
            try:
                result = call.get(timeout=7200)
            except BaseException:
                call.cancel(terminate_containers=True)
                raise
            atomic_json(arguments.output_dir / f'result-{cores}.json', result)
            print(json.dumps({'cores': cores, 'status': result['status']}), flush=True)
            deadline = time.monotonic() + 90
            while functions[cores].get_current_stats().num_total_runners:
                if time.monotonic() > deadline:
                    raise RuntimeError('Previous GPU has not scaled down; no next allocation submitted')
                time.sleep(2)


if __name__ == '__main__':
    main()
