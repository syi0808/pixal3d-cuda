# Low-credit Modal deployment

**Adapter implemented and locally contract-tested, not deployed or GPU-tested
on Modal yet.** Running the commands below provisions billable resources. Set
an account budget and approve a small benchmark before exposing a service.

## Infrastructure as code

Yes: use Modal's native **Python declarations plus a checked-in TOML config**,
not a separate VM or Kubernetes deployment. This repository now contains:

| File | Responsibility |
| --- | --- |
| `deploy/modal.toml` | App/resource names, GPU, CPU/RAM, timeouts, scale and upload limits |
| `deploy/settings.py` | Strict local config validation; rejects unknown keys and nonzero minimum replicas |
| `deploy/modal_app.py` | Image, persistent Volumes/Dict, Secret reference, CPU/GPU Functions and SDK client |
| `.github/workflows/modal-deploy.yml` | Manual-only, tested dev/prod deployment through GitHub Actions |

Modal [updates an existing named App in place](https://modal.com/docs/guide/apps)
when you redeploy it. Resources are declared by stable names; the config and
code can be reviewed and versioned together. **This is IaC-like, not Terraform
state management**: our `plan` action only prints/validates local declarations,
not a remote state diff, price estimate, dry-run deployment or spending cap.
Secret values, account budgets, environment creation and artifact retention
remain explicit operational setup, not silently provisioned by this code.

Default policy stays GPU minimum zero, maximum one, idle two seconds, no
automatic retries or GPU-type fallback. Only T4/L4 are accepted. `PIXAL_GPU`
optionally overrides the checked-in GPU for a benchmark; apply the same override
to deployment and submission. A config/source change invalidates job reuse.

Use native [Modal Environments](https://modal.com/docs/guide/environments) to
isolate dev and prod Apps, Secrets, Volumes and Dicts without changing their
names. Set `MODAL_ENVIRONMENT` consistently for deploy, prefetch and client
requests; it is not an entry in the resource TOML.

## Architecture

```text
authenticated client -> CPU prepare -> GPU infer -> CPU export -> persistent GLB
                         |               |             |
                  matting/validation   neural only   mesh/UV/PBR
                    weights Volume (CPU prefetch, GPU read-only)
                    jobs Volume + Dict (status/deduplication)
```

`deploy/modal_app.py` deploys private Modal Functions, **not a public HTTP API**.
The local client uploads inputs, submits work, polls from the user's machine,
and downloads artifacts. Each phase commits its Volume before spawning the
next phase; the GPU returns without waiting for CPU export. No paid CPU worker
waits through the whole GPU job. Outputs outlive FunctionCall result retention.

For a frontend, later add a thin **CPU** API: authenticated POST returns 202 +
job ID, GET polls status, and GET/download uses an expiring object-storage URL.
Keep inference off the HTTP request path; Modal's
[HTTP timeout is 150 seconds](https://modal.com/docs/guide/webhook-timeouts).
Do not expose a GPU-triggering public endpoint without authentication, per-user
quotas, upload limits, and rate limiting. Those production API controls are
**not** implemented by this private adapter.

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

Concurrency/timeouts are **not dollar caps**. Platform failure/preemption can
interrupt a job without running its Python error handler, leaving stale status.
Inspect Modal logs and ensure the old call has ended before retrying. The
adapter does not automatically retry/resume GPU jobs or delete old artifacts.

## Deploy and run

From the CUDA checkout, using the exact source used for deployment:

```bash
python -m pip install -e '.[modal]'
python -m deploy.modal_app --action plan
modal setup
modal environment create dev
export MODAL_ENVIRONMENT=dev
```

In the Modal dashboard's **dev environment**, create Secret `pixal3d-hf` with `HF_TOKEN`, a read token
whose account has DINOv3 access. Do not paste tokens into scripts or chat. The
secret is attached only to the CPU prefetch function.

The following **starts using your Modal account/credits**:

```bash
modal deploy -m deploy.modal_app
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
Volume directory, also remove its Dict record; otherwise its cached status
still points at the deleted artifacts. Establish a retention policy (for
example delete intermediate `.pt` checkpoints after approved results, retain
GLB/report seven days) before sustained use; no cleanup schedule is enabled.

## Manual GitHub deployment

Create Modal environments `dev` and `prod` and the HF Secret in each environment
you intend to use. In GitHub, create corresponding **Environments** and add
`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` for a dedicated Modal deployment identity.
These are Modal credentials, not the HF token. Configure required reviewers for
the GitHub `prod` environment before using it.

Select **Actions -> Deploy Modal manually -> Run workflow**, choose a source
revision and target environment. The workflow runs CPU tests and the local
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
