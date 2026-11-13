# Official runtime on Modal main

**Not deployed or GPU-validated after the v0.2 rewrite.** These are declarations
and an untested native build recipe, not a working-server claim. Old T4 evidence
applies only to [the previous revision](https://github.com/syi0808/pixal3d-cuda/tree/a6ce8a3c4a68e377569b7965035a8f00d74ff9b0).

## Layout

- `deploy/modal.toml`: main only, L4/SM89, one GPU worker maximum, zero warm
  containers, no retries, short scaledown window, bounded HTTP admission.
- `pixalcuda/upstream-lock.json`: upstream/dependency/model revisions.
- `deploy/modal_app.py`: CUDA 12.4 devel image, Torch 2.6/cu124, native extension
  source builds; CPU prefetch/copy/HTTP workers and a native GPU worker.
- CPU prepare validates/copies uploads **without matting/cropping**. GPU worker
  uses the original BiRefNet/camera/conditioning pipeline and original CUDA
  remesh/PBR exporter. No separate CPU export worker is dispatched.

There is no dev environment and no automatic GPU upgrade/fallback. The new
declaration explicitly changes the hardware target from T4 to L4, but no live
resource changes happen until deployment. T4 is rejected by the BF16 baseline.
Native L4 fit/build/inference still need validation. The 64 GiB host RAM limit
is not a VRAM allowance. Upstream loads all models on CPU even in low-VRAM mode.

## Manual rollout

Review the L4 cost and model licenses/access first. Keep the old app stopped
until the image builds and a single reference job is deliberately authorized.
Existing Modal authentication, `pixal3d-hf` secret (`HF_TOKEN`), Proxy Token,
`.env` Proxy credentials and weights/jobs volumes are reusable; do not print or
commit credentials. RMBG-2.0 / camenduru DINO access may differ from old weights.

```bash
python -m deploy.modal_app --action plan
modal deploy --env main -m deploy.modal_app
python -m deploy.modal_app --action prefetch
python -m deploy.modal_app --action url
```

Deploy compiles CPU/CUDA code; it does not run asset generation. Prefetch downloads
pinned snapshots on CPU, including model JSONs, official DINO, BiRefNet, NAF and
optional MoGe. Old ready markers fail the new runtime/lock checks and must be
replaced by the new prefetch. Immutable cached files can still be reused.

HTTP routes and Proxy authentication remain unchanged:

- `GET /api/health`: `runtime=official`, revision, `[1024,1536]`, L4, SDPA/low-VRAM.
- `POST /api/jobs`: multipart `image` **or** mapped `images` + `transforms`; optional JSON `options`.
- `GET /api/jobs/{id}`, `/result.glb`, `/report.json`.

New defaults are `geometry_resolution=1024`, `texture_size=4096`,
`target_faces=1000000`. There is no `conditioning_cache_mb` compatibility field
and no alternate runtime. Old geometry 512/chunk/cache options are rejected.
Job schema 4 and source/lock fingerprints prevent reuse of legacy outputs as
native results. Failure/OOM is surfaced; there is no precision, quality or GPU
fallback. Input files and private reference tensors stay in private volumes.

After a deliberate single-image test, verify report stages include
`remesh_pbr_export`, embedded PBR images and actual geometry; inspect the GLB
visually before benchmarking. This has **not** been done for v0.2 yet.
