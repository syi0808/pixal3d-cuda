# Single-environment Modal deployment

**Workers and HTTP transport are locally contract-tested, not deployed or GPU-tested
on Modal yet.** Live URL/authentication verification remains pending. Running
the commands below provisions billable resources. Set
an account budget and approve a small benchmark before exposing a service.

## Infrastructure as code

Yes: use Modal's native **Python declarations plus a checked-in TOML config**,
not a separate VM or Kubernetes deployment. This repository now contains:

| File | Responsibility |
| --- | --- |
| `deploy/modal.toml` | App/resource names, GPU, CPU/RAM, timeouts, scale and upload limits |
| `deploy/settings.py` | Strict local config validation; rejects unknown keys and nonzero minimum replicas |
| `deploy/modal_app.py` | Image, persistent Volumes/Dict, Secret reference, CPU/GPU Functions and SDK client |
| `deploy/http_api.py` | Bounded asynchronous HTTP submission, status and artifact downloads |
| `.github/workflows/modal-deploy.yml` | Manual-only deployment to the single `main` environment |

Modal [updates an existing named App in place](https://modal.com/docs/guide/apps)
when you redeploy it. Resources are declared by stable names; the config and
code can be reviewed and versioned together. **This is IaC-like, not Terraform
state management**: our `plan` action only prints/validates local declarations,
not a remote state diff, price estimate, dry-run deployment or spending cap.
Secret values, account budgets and artifact retention
remain explicit operational setup, not silently provisioned by this code.

Default policy stays GPU minimum zero, maximum one, idle two seconds, no
automatic retries or GPU-type fallback. Only T4/L4 are accepted. `PIXAL_GPU`
optionally overrides the checked-in GPU for a benchmark; apply the same override
to deployment and submission. A config/source change invalidates job reuse.

Configuration targets **only the existing `main` Modal environment**. No separate
development environment is created. Resource and SDK lookups explicitly select
`main`, and stale `MODAL_ENVIRONMENT=dev` settings are rejected before contacting
Modal. Use `MODAL_ENVIRONMENT=main` and `modal deploy --env main`. Existing user
resources in other environments are not migrated or deleted automatically.

## Architecture

```text
HTTP/SDK client -> CPU API/prepare -> GPU infer -> CPU export -> persistent GLB
                         |               |             |
                  matting/validation   neural only   mesh/UV/PBR
                    weights Volume (CPU prefetch, GPU read-only)
                    jobs Volume + Dict (status/deduplication)
```

Deploying the App exposes a **Proxy Token-authenticated HTTPS API**, in
addition to private SDK workers. The CPU API persists uploads and returns
`202` plus a job ID after dispatching preparation; it does not wait for matting,
GPU inference or CPU export. Each phase commits before spawning the next one.
The GPU returns without waiting for CPU export; no orchestration worker stays
paid for the entire generation. HTTP downloads run only on CPU.

Inference stays outside Modal's
[150-second HTTP request window](https://modal.com/docs/guide/webhook-timeouts).
The API has its own 120-second function timeout. Poll job status rather than
holding one request open through generation. Both API and workers scale to
zero. Artifact copies finish with the Volume closed before HTTP streaming,
avoiding reloads while another request has that Volume file open.

## Cost controls implemented

- GPU defaults to T4; only explicit `PIXAL_GPU=T4` or `L4` is accepted. No
  fallback to a more expensive GPU.
- `min_containers=0`, `buffer_containers=0`, `max_containers=1`, idle window two
  seconds, one input at a time. This follows Modal's
  [scale-to-zero controls](https://modal.com/docs/guide/scale).
- CPU prefetch persists pinned weights/rembg data once. GPU runs offline with
  read-only weights; missing caches fail rather than downloading on GPU time.
- CPU matting preserves calibrated framing; CPU mesh extraction, repair,
  simplification, UV unwrap, texture bake and GLB validation follow inference.
  Single-image MoGe camera inference still runs as a separate model on GPU.
- Request hashes cover ordered image content, all poses, model revision, GPU,
  settings and runtime/deployment source. Atomic Dict reservation prevents
  duplicate submissions from launching another generation. Complete results
  are reused; failed/stuck requests require an explicit new `--attempt` value.
- GPU timeout 1800 s; explicit retries zero. CPU/GPU memory request/limits are
  starting settings, **not measured multi-view memory requirements**.
- Only runtime/deployment source and dependency metadata are copied into the
  image; local outputs, weights, `.env` and other checkouts are excluded.
- New HTTP jobs have a default **10 admissions per UTC day** shared across all
  authorized callers, configured by `api_max_jobs_per_day`. Atomic persistent
  quota slots deduplicate the same job; cache hits do not consume another slot.
  Failed admissions after reservation count conservatively. Exhaustion returns
  `429` with `Retry-After`. Administrative SDK calls do not use this HTTP quota.

Concurrency/timeouts are **not dollar caps**. Platform failure/preemption can
interrupt a job without running its Python error handler, leaving stale status.
Inspect Modal logs and ensure the old call has ended before retrying. The
adapter does not automatically retry/resume GPU jobs or delete old artifacts.

## Deploy and run

From the CUDA checkout, using the exact source used for deployment:

```bash
python -m pip install -e '.[modal,server]'
modal setup
export MODAL_ENVIRONMENT=main
python -m deploy.modal_app --action plan
```

In the Modal dashboard's **main environment**, create Secret `pixal3d-hf` with `HF_TOKEN`, a read token
whose account has DINOv3 access. Do not paste tokens into scripts or chat. The
secret is attached only to the CPU prefetch function.

The following **starts using your Modal account/credits**:

```bash
modal deploy --env main -m deploy.modal_app
python -m deploy.modal_app --action prefetch --multiview
python -m deploy.modal_app --images photo-a.png photo-b.png photo-c.png \
  --transforms cameras.json --no-rembg
```

For single-image generation prefetch without `--multiview`, then use
`--image-path object.png`. `--multiview` selects the prefetch mode only;
generation selects it from `--images` or `--views-dir`. Images/poses follow
[the mapped image/camera contract](multiview.md). The adapter's defaults are
1-16 mapped images, 16 MiB and 16 megapixels per image, 64 MiB combined images,
and 64 KiB camera JSON. PNG/JPEG/WebP only; no animation. Remove EXIF rotations
before supplying calibrated images. These limits are checked locally and on
CPU before GPU submission, and are configurable in `deploy/modal.toml`.

The deployed app continues after the client disconnects. Reconnect without
resubmitting inference:

```bash
python -m deploy.modal_app --action status --job-id PRINTED_SHA256
```

GLB, report and manifest download to `outputs/modal/JOB_ID/`. Raw inputs,
prepared inputs and intermediate checkpoints remain private in `pixal3d-jobs`;
weights stay in `pixal3d-weights`. Both use
[Volumes v2](https://modal.com/docs/guide/volumes). Drain pending/running jobs
before redeploying or updating the model cache: chained phases must use the
same source. Deploy and submit with the same GPU, environment, config and source.

Use `--attempt retry-1` to intentionally pay for a fresh attempt of a failed
request. Complete-cache hits do not regenerate. If you manually delete a job's
Volume directory, its cached status still points at the deleted artifacts.
Use a new explicit attempt to regenerate; do not remove admission/prepare
claims while a job might still be running. Establish a retention policy (for
example delete intermediate `.pt` checkpoints after approved results, retain
GLB/report seven days) before sustained use; no cleanup schedule is enabled.

## Remote HTTPS API

Yes, you can call the deployed server directly from `curl`, Python, another
backend or any HTTP client. A VM/SSH connection is not required. Modal assigns
the HTTPS URL at deployment; see its
[Web Function URL behavior](https://modal.com/docs/guide/webhook-urls). Obtain
the actual URL from the deploy output/dashboard, or as an administrator:

```bash
python -m deploy.modal_app --action url
```

Every route, including health, OpenAPI and docs, is protected by
`@modal.asgi_app(requires_proxy_auth=True)`. Create a **Modal Proxy Token** in
workspace settings. Send its ID/secret as `Modal-Key` and `Modal-Secret` headers,
as documented in [Proxy Token authentication](https://modal.com/docs/guide/webhook-proxy-auth).
This is distinct from the SDK/deployment token and from `HF_TOKEN`. No extra
application API-key Secret is needed. Keep Proxy Tokens on trusted clients or
backend servers, never in public frontend bundles, notebooks or Git.

| Method | Route | Behavior |
| --- | --- | --- |
| GET | `/api/health` | CPU-only service metadata; does not launch generation |
| POST | `/api/jobs` | Multipart upload; returns 202 and `job_id` without waiting for inference |
| GET | `/api/jobs/{job_id}` | Shared-owner status and relative artifact URLs |
| GET | `/api/jobs/{job_id}/result.glb` | Completed embedded-PBR GLB; 409 while unavailable |
| GET | `/api/jobs/{job_id}/report.json` | Runtime report for completed/failed jobs, if produced |

Set `BASE_URL` to the real deployed URL and load your Proxy Token into local
`PIXAL_PROXY_KEY` / `PIXAL_PROXY_SECRET` environment variables. Verify health
before spending credits on POST:

```bash
curl --fail-with-body "$BASE_URL/api/health" \
  -H "Modal-Key: $PIXAL_PROXY_KEY" -H "Modal-Secret: $PIXAL_PROXY_SECRET"
```

For mapped images, repeat the `images` file field in the order addressed by
`image_index`. Send camera JSON as **text**, using `<`, not the `@` file field:

```bash
curl --fail-with-body "$BASE_URL/api/jobs" \
  -H "Modal-Key: $PIXAL_PROXY_KEY" -H "Modal-Secret: $PIXAL_PROXY_SECRET" \
  -F "images=@photo-a.png" -F "images=@photo-b.png" -F "images=@photo-c.png" \
  -F "transforms=<cameras.json" -F 'options={"remove_background":false}'
```

For unposed single-image mode, use one `image=@object.png` field and omit
`transforms`. Never mix `image` and `images`. Options accept seed, texture size,
face target, background removal and an explicit retry `attempt` tag. Single-image
camera/FOV overrides must not accompany mapped images. Geometry remains 512;
resolution, precision, step count and GPU-type overrides are not HTTP options.
Uploads are validated on CPU before any GPU dispatch; URLs are not fetched.

Use the returned `job_id` to poll every few seconds, not a long-lived request:

```bash
curl --fail-with-body "$BASE_URL/api/jobs/$JOB_ID" \
  -H "Modal-Key: $PIXAL_PROXY_KEY" -H "Modal-Secret: $PIXAL_PROXY_SECRET"
```

After status becomes `complete`, download the actual GLB:

```bash
curl --fail "$BASE_URL/api/jobs/$JOB_ID/result.glb" \
  -H "Modal-Key: $PIXAL_PROXY_KEY" -H "Modal-Secret: $PIXAL_PROXY_SECRET" \
  --output result.glb
```

Exact duplicate inputs/settings reuse the same job and do not start another
GPU run, even if the HTTP response was lost. A failed request is not silently
retried; use a new `options.attempt` only after inspecting its status/logs.
Deleted complete artifacts return 410; failed jobs with no runtime report
return 404 for that report. There are no HTTP prefetch, arbitrary-file, or
model-cache modification endpoints.

This is a **shared-owner API**, not a multi-tenant SaaS authorization system.
All valid Proxy Token callers share job visibility and the admission quota.
The quota is not a dollar cap or a request-rate limit: authenticated polling,
downloads and invalid uploads can still consume CPU. Configure account budgets;
add per-user ownership/rate controls before offering this to untrusted users.

Local tests exercise multipart validation, asynchronous dispatch, deduplication,
quota, status and artifact transport. They do **not** emulate Modal's edge
authentication. After actual deployment, verify that no/invalid Proxy Token gets
401, a valid token can read health, and one approved real input produces a
visually reviewed GLB. Those live checks have not been run yet.

## Manual GitHub deployment

Add repository Actions Secrets `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` for a
dedicated Modal deployment identity. These are SDK/deployment credentials, not
the HF token or HTTP Proxy Token. The HF Secret lives only in Modal `main`.
No separate GitHub or Modal development environment is required.

Select **Actions -> Deploy Modal manually -> Run workflow** and choose a source
revision. The target is always `main`. The workflow runs CPU tests and the local
declaration printout, then `modal deploy`. It uses Modal's documented
[GitHub Actions deployment pattern](https://modal.com/docs/guide/continuous-deployment).
It never runs on push/PR/schedule and never invokes prefetch or inference, so
merging code alone cannot start a paid generation. Deployment/image building
can still use account resources; review the declaration and budget first.

Reapplying the same environment/App name updates the deployment rather than
creating a new named service. Review config changes and drain jobs first.
Rollback can be done by redeploying a reviewed earlier source/config revision;
this does not roll back or erase persisted inputs, checkpoints or model caches.
There is no automated destroy step or data-deletion workflow. Changing a Volume
name selects new storage; it does not migrate/delete the previous named Volume.

## Choosing T4 versus L4

As checked **2026-11-12**, Modal's
[published rates](https://modal.com/pricing) list:

| Resource | Rate |
| --- | ---: |
| T4 GPU only | $0.000164/s = $0.5904/h |
| L4 GPU only | $0.000222/s = $0.7992/h |
| CPU physical core (two vCPU equivalent) | $0.0000131/s |
| RAM | $0.00000222/GiB/s |
| Volumes | $0.09/GiB/month, published included allowance applies |
| Network egress | $0.04/GiB, plan allowances apply |

T4 is cheaper per GPU-second, not necessarily per asset. L4 needs about **1.354x
speedup** to break even on the GPU component alone. Include requested/used
CPU/RAM, image startup, weight loading, CPU export, storage and egress when
comparing total cost. The previous single-image Colab run included downloads;
its 1083 s must not be used as a warm Modal cost forecast.

Next benchmark: same four posed images, seed, 512 geometry, 12 steps, textures
and face target; one cold and at least three warm runs on T4 and L4. Compare
actual billed cost per **valid visually reviewed PBR GLB**, not latency alone.
Keep FP32 correctness until model/stage-level precision changes are validated.
