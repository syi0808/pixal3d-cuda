# Official runtime on Modal main

**Deployed to main on 2026-11-13:** native SM89 image build and authenticated HTTP
smoke checks pass. On 2026-11-14 KST, both profiles completed actual single-image
1024/4K PBR generation on L4, but strict native tensor parity failed. See the
[performance pilot and limitations](performance.md). RMBG-2.0 is
disabled by default, so its previous access 403 is not a default prefetch
requirement. See [optional background removal](background-removal.md) and
[rollout evidence](validation.md#v02-modal-main-rollout).
Old T4 evidence
applies only to [the previous revision](https://github.com/syi0808/pixal3d-cuda/tree/a6ce8a3c4a68e377569b7965035a8f00d74ff9b0).

The subsequent [fixed-kernel full profiler](full-profiling.md) passes 149 neural
checkpoints exactly on the same shoe, under diagnostic-only backend controls.
It does not change production defaults or establish final GLB byte parity.

Subsequent [coverage and CPU/inpaint rollout](loading-and-cpu-tuning.md) enables
guarded no-init, `gpu_cpu_cores=8` and `inpaint_workers=4` in main. API/prepare
remain at `cpu_cores=2`; max GPU=1/min=0 are unchanged. CPU 4 is a cost-first
alternative to the latency-oriented CPU 8 default; cold-asset estimates are close.
Clients cannot override these operator budgets. `reference=true` disables no-init
and parallel inpaint alongside the reuse hooks, without a second runtime.

## Layout

- `deploy/modal.toml`: main only, L4/SM89, one GPU worker maximum, zero warm
  containers, no retries, short scaledown window, bounded HTTP admission.
- `pixalcuda/upstream-lock.json`: upstream/dependency/model revisions.
- `deploy/modal_app.py`: CUDA 12.4 devel image, Torch 2.6/cu124, native extension
  source builds; CPU prefetch/copy/HTTP workers and a native GPU worker.
- CPU prepare validates/copies uploads **without matting/cropping**. GPU worker
  defaults to supplied framing/alpha, with original camera/conditioning and CUDA
  remesh/PBR exporter. No separate CPU export worker is dispatched.

There is no dev environment and no automatic GPU upgrade/fallback. The new
declaration explicitly changes the hardware target from T4 to L4. Deployment
does not start asset generation. T4 is rejected by the BF16 baseline.
Native L4 model fit/inference passed for one prepared shoe, not every input,
mapped views or 1536. The 64 GiB host RAM limit
is not a VRAM allowance. Upstream loads all models on CPU even in low-VRAM mode.

## Manual rollout

Review the L4 cost and model licenses/access before deliberately submitting a
generation job. The deployed app remains scale-to-zero while idle.
Existing Modal authentication, `pixal3d-hf` secret (`HF_TOKEN`), Proxy Token,
`.env` Proxy credentials and weights/jobs volumes are reusable; do not print or
commit credentials. RMBG-2.0 is separately licensed and is not downloaded or
initialized unless explicitly enabled. No other matting model is substituted.

```bash
python -m deploy.modal_app --action plan
modal deploy --env main -m deploy.modal_app
python -m deploy.modal_app --action prefetch
python -m deploy.modal_app --action url
```

Deploy compiles CPU/CUDA code; it does not run asset generation. Prefetch downloads
pinned snapshots on CPU, including model JSONs, official DINO, NAF and
optional MoGe. Old ready markers fail the new runtime/lock checks and must be
replaced by the new prefetch. Immutable cached files can still be reused.

Native extensions have separate cached image layers, followed by the runtime
sources. Editing API/runtime code does not recompile CUDA. The image explicitly
uses GCC/G++ and CMake 3.31.6. MoGe and the runtime share the same pinned utils3d
1.3 source revision; installing the old 0.0.2 wheel alongside MoGe is a resolver
conflict. This dependency change does not add a second inference path.

HTTP routes and Proxy authentication remain unchanged:

- `GET /api/health`: `runtime=official`, revision, `[1024,1536]`, L4, SDPA/low-VRAM.
- `POST /api/jobs`: multipart `image` **or** mapped `images` + `transforms`; optional JSON `options`.
- `GET /api/jobs/{id}`, `/result.glb`, `/report.json`.

Background removal defaults to false in the HTTP and administrative clients.
The checked-in `allow_rmbg=false` rejects explicit removal requests before job
admission. Health reports this policy; see the [two-sided opt-in procedure](background-removal.md).

New defaults are `geometry_resolution=1024`, `texture_size=4096`,
`target_faces=1000000`. There is no `conditioning_cache_mb` compatibility field
and no alternate runtime. Old geometry 512/chunk/cache options are rejected.
Job schema 4 and source/lock fingerprints prevent reuse of legacy outputs as
native results. Failure/OOM is surfaced; there is no precision, quality or GPU
fallback. Input files and private reference tensors stay in private volumes.

Native DINO reuse/active projection/conditioning reuse defaults are declared in
`modal.toml` (128 MiB DINO / 256 MiB per-flow conditioning). This is a new native
optimization, not compatibility with the removed runtime. HTTP `reference=true`
disables the hooks in the same pipeline; clients cannot override server budgets.
See [controls and validation limits](optimizations.md).

After a deliberate single-image test, verify report stages include
`remesh_pbr_export`, embedded PBR images and actual geometry; inspect the GLB
visually before benchmarking. This passed for the L4 pilot's two GLBs; strict
tensor parity and repeated equivalent-output performance remain unaccepted.

## Bounded HTTP acceptance

After deployment and successful CPU prefetch, use a fresh output directory:

```bash
python scripts/validate_modal_native.py \
  --base-url https://YOUR-APP.modal.run --image object.png \
  --output-dir outputs/modal-native-acceptance
```

The default is a local dry-run. Add `--execute` to test unauthenticated rejection,
authenticated health, and sequential optimized/reference generation through the
actual HTTP API. Defaults are geometry 1024, texture 4096, 1M target faces and seed
42. The tool reads Proxy credentials from `.env` without printing them, validates
downloaded PBR GLBs and writes `acceptance.json`. A failed job stops the test;
there is no retry or hardware/quality fallback. A client timeout does not cancel
the remote job: inspect/cancel that job before submitting another attempt.

For private tensor comparisons only, temporarily set `save_reference = true` in
`modal.toml`, redeploy, then run the two profiles. Download each job's
`/<job_id>/run/reference.pt` with the Modal Volume CLI and compare with
`scripts/compare_native_references.py`. These tensors are not HTTP artifacts.
Restore `save_reference = false` and redeploy after diagnostics to avoid normal
requests accumulating large private files. Structural GLB checks and raw tensor
comparison do not replace visual inspection of the final remesh/UV/PBR output.
