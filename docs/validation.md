# Validation status

> The pretrained GPU results below apply only to the archived v0.1.x runtime.
> The official v0.2 rewrite has CPU contract tests but no native GPU acceptance yet.

## Optional RMBG default policy

Subsequent 2026-11-13 update: RMBG-2.0 is excluded from default snapshot downloads,
pipeline initialization, CLI generation and HTTP requests. The server additionally
uses `allow_rmbg=false`: clients cannot force its use or enable this operator flag
through HTTP. Explicit licensed opt-in remains possible, without a second 3D
runtime or a silent replacement model. Existing cached matting weights are not
automatically used or deleted.

**190 CPU tests passed.** Tests run the pinned original `from_pretrained` and
RGBA `preprocess_image` method bodies with tiny sampler/model doubles, verify
zero eager matting construction, device/offload behavior, factory restoration,
SV/MV optional downloads, accurate ready markers and pre-admission rejection.
These are not pretrained GPU or full-asset quality tests.

Main was redeployed in 44.458 seconds using the cached native layers. Observed:
authenticated health 200, unauthenticated health 401,
`background_removal_default=false`, `rmbg_enabled=false`, and an explicit matting
request rejected with 422. Default **CPU prefetch passed** with `rembg=false` and
automatic-camera weights ready. The earlier RMBG access 403 is therefore no
longer a default cache-readiness blocker. No GPU generation was submitted in
this update, and zero active containers were observed afterward.

Default single-image RGB/RGBA keeps supplied framing and composites alpha on
black, without automatic matting/crop; mapped views retain calibrated framing.
Use already prepared images, not raw photos expecting automatic extraction.
This is an intentional preprocessing-default change, not an original default
preprocessing-parity claim. The previous GPU/kernel-validation limitations below
still apply; no new GLB or speedup is claimed.

[Policy and license opt-in](background-removal.md),
[sanitized evidence](evidence/optional-rembg-2026-11-13.json).

## v0.2 Modal main rollout

The initial checkpoint below predates the optional-RMBG policy above.

2026-11-13: the CUDA 12.4/Python 3.12/Torch 2.6 SM89 image built successfully,
including nvdiffrast, CuMesh, FlexGEMM, o_voxel and NATTEN 0.21.0. Build failures
identified a utils3d/MoGe resolver conflict, missing G++ selection in the Modal
Python toolchain, and missing CMake. The recipe now shares MoGe's pinned utils3d
1.3 revision, selects GCC/G++, installs CMake and caches each extension separately
from runtime sources. NATTEN's first source build took 1,970.35 seconds on CPU;
the subsequent cached main deployment took about 40 seconds.

The deployed API rejected unauthenticated health (401), returned authenticated
health/OpenAPI (200), rejected unsupported geometry 512 (422), and returned 404
for an unknown job. These checks submitted no generation jobs. Local CPU tests:
**170 passed**; the Starlette/httpx deprecation warning remains unrelated.

**Pretrained generation remains blocked:** the actual main `pixal3d-hf` token
receives 403 from the pinned `briaai/RMBG-2.0/config.json`. The account must accept
the model's access conditions and the token must allow gated-model reads. No
alternate matting model, old weights, old GLB or automatic hardware upgrade was
substituted. The 1024/4096/1M-face optimized/reference pair was planned, not run;
there is no new GLB, model VRAM measurement, speedup or quality-parity result.

The separate, model-free L4 kernel diagnostic is **not accepted** either. Its first
attempt omitted the pipeline's upstream Python search path and failed before
kernel calls. The corrected attempt produced no function-body output within a
300-second client deadline and was cancelled with container termination. This
does not establish a kernel bug, GPU-capacity problem or successful kernel test.
Both temporary Apps were observed stopped. Main stays deployed with zero active
containers observed, min=0, one GPU worker maximum and `save_reference=false`.

The o_voxel metadata resolves unversioned CuMesh/FlexGEMM git references even
when locked local builds are already installed. In this observed build those
references matched the lock and no replacement wheel was installed. This is
evidence for this image, not a fully hermetic future dependency-resolution claim.

[Sanitized rollout evidence](evidence/modal-native-2026-11-13.rollout.json).
See [bounded HTTP acceptance instructions](modal.md#bounded-http-acceptance).
Credentials, input-derived tensors and large artifacts are not committed.

## v0.2 official-only rebase

Subsequent optimization contracts: **160 CPU tests passed**, including comparisons
against pinned upstream projection/attention/elastic-wrapper method bodies with
small DINO/NAF/sparse-tensor doubles. Python wheel build and dry-run plans passed.
This does not validate native kernels, pretrained GLB parity or speed. See
[optimization controls and limits](optimizations.md) and
[the CPU evidence summary](evidence/upstream-optimizations-2026-11-13.contracts.json).

2026-11-13: **129 CPU contracts passed**, and the Python wrapper wheel was built
with its packaged upstream lock and no former neural/exporter modules. Native
single/mapped wiring uses controlled doubles, not pretrained GPU execution.
At that initial rebase checkpoint, the native benchmark and Modal declaration
plans were run without execution and no paid GPU/app was started. The rollout
above supersedes its native build and HTTP-deployment status only. Pretrained
generation, upstream parity, capacity and performance remain unverified.
See [the current migration/investigation](upstream-runtime.md).

## Scope

Initial implementation date: 2026-11-11 (Asia/Seoul). Development host: macOS/Apple Silicon, Python 3.12, PyTorch 2.10.0, CPU tests. The separate MLX repository is reference-only and is not a runtime dependency.

The Colab notebook pins PyTorch 2.6.0 / torchvision 0.21.0 / CUDA 12.4. Linux CPU CI also uses PyTorch 2.6.0. Cross-platform installation and GPU execution are separate validation domains.

## Verified locally

- Native model parameter names/shapes match the pinned official SS flow, LR/HR shape flows, texture flow, SS/shape/texture decoders, DINOv3, and NAF checkpoints. The unused dense SS RoPE buffer and DINO's mask/final learned norm are explicitly excluded.
- CPU global query-chunk attention, complex RoPE, CFG rescale, and algebraic zero-negative-context folding contracts.
- Sparse submanifold convolution against dense PyTorch Conv3d, plus subdivision ordering and 512 requantization.
- Strict tensor-at-a-time loading, checkpoint identity validation, and failure report preservation.
- Actual synthetic-mesh UV unwrap, trilinear PBR sampling, embedded base-color/metallic-roughness textures, GLB validation and reload. This is an exporter test, not pretrained image-to-3D output.
- Colab notebook JSON/schema and Python cell syntax. No saved outputs or credentials.
- Optional pinned MoGe-2 dependency installation/import and real CPU camera inference complete successfully. This is not a T4 camera benchmark.

Initial local suite: **23 passed, 2 skipped** (the two CUDA-only tests). A controlled-output pipeline wiring/resume test exercises all stages through a real PBR export, but deliberately does not claim pretrained full-model inference.

The same **23 passed, 2 skipped** suite also passes on GitHub Actions Linux with the notebook's PyTorch 2.6.0 version. CUDA tests remain skipped on that CPU-only runner.

## Frozen-input MLX reference comparisons

The reference checkout was read/executed without writing files into it (`PYTHONDONTWRITEBYTECODE=1`). Comparison artifacts remain ignored under `outputs/`; no user image or model weights are published.

| Component | Conditions | Maximum absolute difference |
| --- | --- | ---: |
| NAF | Official weights, synthetic 64 guide, 16×16 source features, 23 queries including borders | 4.95e-6 |
| Flow | Two-block/48-channel dense model, identical FP16-rounded random weights, FP32 execution | 2.98e-7 |
| Sparse decoder | One-level 16→8-channel model, identical FP16-rounded random weights, FP32 execution | 1.00e-6 |
| DINOv3 | Official ViT-L/16 weights, identical synthetic 128×128 image, FP32 execution | 7.82e-5 |

The sparse decoder's output coordinates match exactly. These are scoped numerical comparisons, not quality equivalence of large-model CUDA flow sampling, full-resolution DINO features, or a generated mesh. Random tiny-model tests do not establish pretrained generation quality.

## Actual T4 acceptance history

The first actual Colab T4 attempt ran on 2026-11-11, Python 3.13.16, PyTorch 2.6.0+cu124, driver 580.82.07, Tesla T4 (SM 7.5), with 15,637,086,208 GPU bytes and 13,605,818,368 system RAM bytes. The pinned public shoe sample is identified by SHA-256 `abf395cc52d81c26dadae9f024072d6c7301679be4e8fc08d572723d7ae32a21` in `scripts/validate_colab_t4.py`.

At revision `02aa162`, preprocessing/camera and DINO stages returned, but SS sampling rejected a non-finite state at step 1. This was **not an OOM**. Both cached DINO feature tensors were entirely non-finite (1,053,696 values at 512, 4,199,424 at 1024). Official-weight FP16 DINO inference also reproduced non-finite output at layer 0 on the CPU development host. The runtime now explicitly loads/executes DINO in FP32, rejects non-finite features before caching, and rejects non-finite tensors introduced during checkpoint dtype conversion. This precision correction needs a new output directory rather than reuse of the poisoned feature cache.

This failed initial attempt did not establish end-to-end completion.

The second actual T4 attempt at `4274bc4` used FP32 DINO and completed all 12 SS steps, but FP16 LR shape sampling produced a non-finite state at step 10 (288.01 s including initial downloads). There was no reported OOM. Flow models and sparse decoders now explicitly execute in FP32 while retaining one-model residency, bounded attention/FFN/convolution intermediates, and all 12-step schedules. The dense occupancy decoder retains its separately exercised FP16 path. This changes the precision identity, so a fresh output directory was used for the next run.

### Completed pretrained run

Revision **`dacb3a11f99e6ebd8d2b904d92294b96b69a5f09`** completed on that actual Colab T4, starting **2026-11-11 06:09:51 UTC**. The command was:

```bash
python -u scripts/validate_colab_t4.py --output-dir outputs/colab-t4-shoe-stable-fp32
```

Conditions: seed 42, geometry grid 512, texture size 1024, automatic MoGe-2 camera, background removal enabled, default bounded chunks/guards, and all four official 12-step flow schedules. No quantization, token pruning, resolution fallback, or shortened schedule was used. The final shape/texture decoder expanded to **777,675 voxels**.

| Result | Observed value |
| --- | ---: |
| End-to-end wall time | 1,083.051943175 s (18 min 3 s) |
| Peak PyTorch CUDA allocated | 5,855,294,464 bytes (5.453 GiB) |
| Peak PyTorch CUDA reserved | 5,888,802,816 bytes (5.484 GiB) |
| Peak sampled process RSS | 7,743,610,880 bytes (7.212 GiB) |
| GLB size | 12,276,444 bytes |
| Exported vertices, including UV seam splits | 246,499 |
| Exported triangles | 150,306 |
| Embedded base-color texture | 1024×1024 |
| Embedded metallic/roughness texture | 1024×1024 |

**Timing includes downloads into a partially populated model cache**, notably the HR shape flow, texture flow, and sparse decoders. It is not pure inference latency or a warm-cache benchmark. CUDA figures are PyTorch allocator measurements, not total `nvidia-smi` device usage or hard limits. RSS is sampled every 100 ms and can miss short peaks. The 100,000-face simplification target is not a strict cap.

| Stage | Seconds | Allocated GiB | Reserved GiB | Sampled RSS GiB |
| --- | ---: | ---: | ---: | ---: |
| preprocess_camera | 19.51 | 2.19 | 2.38 | 3.29 |
| dino_features | 6.91 | 1.31 | 1.36 | 3.49 |
| sparse_structure | 109.93 | 5.37 | 5.40 | 7.21 |
| shape_lr | 73.97 | 5.45 | 5.48 | 7.05 |
| coordinate_subdivision | 43.64 | 2.86 | 2.96 | 3.54 |
| hr_conditioning | 0.97 | 2.68 | 2.72 | 2.64 |
| shape_hr | 360.70 | 5.45 | 5.48 | 6.82 |
| shape_decode | 36.84 | 2.93 | 3.02 | 3.41 |
| texture_flow | 312.54 | 5.45 | 5.48 | 6.59 |
| texture_decode | 76.57 | 2.93 | 3.02 | 4.15 |
| pbr_export | 38.90 | 0.01 | 0.04 | 3.28 |

The actual GLB rendered as the red shoe in the notebook's PBR model-viewer. Surface noise and dark shading artifacts are visible; this is an execution/material-packaging acceptance result, **not production-quality or reference-parity validation**. The downloaded GLB was independently reloaded on the development host: geometry, normals, and UVs are finite, both textures are embedded and 1024×1024, and the file hash matches the T4 record:

```text
35710dcdd27f00f3b7fd8e1ad7d10af4d83aea6bd96a0c4b9341a319aed307a8
```

On that same T4, **29 tests passed in 11.03 s**, including both FP16/FP32 CUDA attention tests. Strict metadata verification matched all nine pinned checkpoints. Raw, credential-free records are retained in [acceptance.json](evidence/t4-shoe-2026-11-11.acceptance.json), [report.json](evidence/t4-shoe-2026-11-11.report.json), and [test output](evidence/t4-shoe-2026-11-11.tests.txt). No weights, input images, or generated assets are committed. Release `v0.1.1` adds documentation/release metadata and pins the notebook; its inference implementation is the one exercised at `dacb3a1`.

### Modal T4 current runtime

On **2026-11-13**, revision **`f30dfc4f3db7c4debbf1a748754a91780b187ae3`**
completed the same public shoe through the deployed `main` HTTP API. CPU prefetch
persisted pinned single-image, DINOv3, NAF, MoGe-2 and rembg weights before
submission. The GPU worker then ran offline on **Tesla T4, SM 7.5,
PyTorch 2.6.0+cu124**, without resuming previous stage checkpoints, OOM or a
reported non-finite stage. Seed 42, 512 geometry, 1024 textures, automatic camera,
all four 12-step schedules, default limits/128 MiB conditioning cache and disabled
sparse prepacking were retained. The source image SHA-256 matches the historical
case; the runtime input hash identifies the separately CPU-matted PNG.

| Result | Observed value |
| --- | ---: |
| Client-observed complete status | 464.97 s (includes cold workers, CPU preparation and polling delay) |
| Sum of GPU-worker stages | 246.923774154 s |
| CPU export stage | 60.69261714 s |
| Peak PyTorch CUDA allocated | 5,866,810,880 bytes (5.464 GiB) |
| Peak PyTorch CUDA reserved | 5,909,774,336 bytes (5.504 GiB) |
| Peak sampled GPU-worker process RSS | 10.452 GiB |
| GLB size | 6,099,084 bytes |
| Exported triangles | 100,695 |
| Embedded base-color and metallic/roughness PNGs | 1024×1024 each |
| Coincident triangles / nonmanifold / inconsistent indexed edges after cleanup | 0 / 0 / 0 |

GPU stage totals include checkpoint loading, transfers, callbacks, cleanup and
camera estimation, not just kernel execution or total billed GPU-container time.
Weight prefetch time is excluded. These are single-run observations, **not a
controlled speedup comparison** with the older Colab run, whose cache,
preprocessing environment and source differ. PyTorch peaks are not total device
usage, and sampled RSS can miss short spikes.

HTTP submission returned 202, completed-status and artifact downloads returned
200, and the downloaded SHA-256 matched the worker record:

```text
744dbf89f46b7955798b92341658cb53c0711e43b165875aa851a92a77c92455
```

An identical second POST returned the same completed job, without dispatching a
second generation. Both embedded images were independently extracted/decoded.
Blender 5.1.2 imported the actual GLB with its normals, confirmed linked base
color/roughness/metallic channels and rendered three viewpoints with no mesh
modifiers. Small visible cracks/openings remain; zero nonmanifold indexed edges
does not establish watertightness or production quality. The 100,000-face target
is advisory, not a strict cap. An initial local acceptance assertion mistakenly
treated it as a cap; correcting that assertion did not change or rerun inference.

Credential-free records: [acceptance](evidence/modal-t4-shoe-2026-11-13.acceptance.json),
[runtime report](evidence/modal-t4-shoe-2026-11-13.report.json),
[Blender witness](evidence/modal-t4-shoe-2026-11-13.render.json), and
[CPU regression output](evidence/modal-t4-shoe-2026-11-13.tests.txt).
The local CPU suite passed **174 tests, with 3 CUDA-only skips**; those tests were
not rerun on the Modal GPU. Model weights, raw inputs, intermediate checkpoints,
GLB and render PNGs are not committed. After verification, the App was explicitly
stopped with zero active tasks; persistent model/job Volumes were retained.

### Modal T4 1024 geometry cache comparison

Revision **`8c54fbff9a2a35dab5e36eebb1b922b4fb2617c3`** added opt-in 1024
geometry and explicit per-flow projection-cache options to the CLI, SDK and HTTP
API. This is not a metadata-only resolution change: subdivision requantizes to
a 64³ latent grid, camera projection uses that 64-grid resolution, and CPU mesh
extraction/PBR voxel sampling use 1024. GLB mesh extras store the grid value;
legacy GLBs without it now report an unknown grid during standalone validation.

Two fresh full runs completed on **2026-11-13** in Modal `main`, in the order
**512 MiB then 1024 MiB**. Both used the same CPU-matted PNG from the original
public shoe: SHA-256 `84f4bc53f07a26a8e649a2db906d6c725ef8fd729765afd03d5301630fc68b12`.
Only CPU matting was reused; all neural stages were regenerated with seed 42,
automatic camera, geometry 1024, textures 1024, target 100,000 faces, FP32
flows/sparse decoders, unchanged chunks and all four 12-step schedules. Pinned
weights were already cached and GPU workers remained offline. Configured 1024
guards were 32,768 tokens / 6,000,000 voxels, not automatic token pruning or
resolution fallback. Actual refinement tokens were **10,305** and decoded
shape/texture voxels were **2,771,717** in both runs.

| Observed metric | Cache 512 MiB | Cache 1024 MiB |
| --- | ---: | ---: |
| GPU-stage sum, seconds | 653.653725966 | 644.017488603 |
| Model callbacks, seconds | 614.596067416 | 604.837014239 |
| Model loading, seconds | 21.527340181 | 22.199428048 |
| CPU export, seconds | 161.261708888 | 157.809413647 |
| Client-observed completion, seconds | 890.568008166 | 849.308429083 |
| Peak PyTorch allocated, GiB | 6.333 | 6.804 |
| Peak PyTorch reserved, GiB | 6.477 | 6.945 |
| Exported triangles | 99,123 | 99,123 |
| GLB bytes | 6,504,616 | 6,504,616 |

Both devices reported Tesla T4 / SM 7.5, PyTorch 2.6.0+cu124 and
15,651,373,056 total CUDA-device bytes (about 14.58 GiB). No OOM or failed
pipeline stage was observed. Downloaded GLBs passed independent finite geometry,
unit-normal, normalized-bounds and embedded 1024×1024 PNG checks. Both final
GLBs have **the identical SHA-256**:

```text
0e403c00c27a72e904f4cfa31a68294f5d0bfd3839373bddeb79d50c13396475
```

Structure, refinement coordinates, HR-shape latent and texture latent tensors
were also exactly equal, with maximum absolute difference zero. Blender 5.1.2
rendered the actual GLBs with imported normals, linked base-color/roughness/metallic
maps and no mesh modifiers. Small visible surface cracks remain. The cleaned
mesh reports 26,933 indexed boundary edges, including fan-cut seams: it is not
validated watertight or production-ready.

**Interpretation:** the observed GPU-stage reduction is **1.47%**, callback
reduction **1.59%**, and extra peak allocation about **0.472 GiB**. Each budget
was sampled once, sequentially, on separate scale-to-zero workers. There is no
variance estimate or causal/statistical speedup guarantee. Client completion
includes differing cold-start/CPU/transport/polling delays; its full difference
must not be attributed to the projection cache. GPU-stage/callback timing is not
pure kernel time or total billed GPU-container time. No 2048 MiB, chunk-size,
prepack or mixed-precision experiment was run, and the default cache stays 128 MiB.

Credential-free records: [comparison](evidence/modal-t4-geometry1024-cache-2026-11-13.comparison.json),
[512 MiB runtime](evidence/modal-t4-geometry1024-cache-2026-11-13.cache512.report.json),
[1024 MiB runtime](evidence/modal-t4-geometry1024-cache-2026-11-13.cache1024.report.json),
[512 MiB render](evidence/modal-t4-geometry1024-cache-2026-11-13.cache512.render.json),
[1024 MiB render](evidence/modal-t4-geometry1024-cache-2026-11-13.cache1024.render.json), and
[CPU tests](evidence/modal-t4-geometry1024-cache-2026-11-13.tests.txt).
The local suite and Linux CPU CI passed **203 tests, with 3 CUDA-only skips**;
no CUDA unit suite was run on Modal in this experiment. The App was explicitly
stopped afterward, with zero active tasks and persistent weights/jobs retained.
No weights, prepared image, raw tensors, GLB or render PNGs are committed.

### Modal T4 chunk profile trials

The CUDA implementation now separates flow cross-attention query chunks from
self-attention and provides independent server declarations for attention, FFN
and sparse rows. Chunk 0 is supported only for cross-attention and means all
queries. Self-attention still sees every key and remains chunked. CLI defaults
stay conservative; Modal's configurable candidate is **1024 / 0 / 8192 / 16384**
for self queries / cross queries / FFN rows / sparse rows.

Two bounded one-shot T4 probes in `main` recorded 36 passing warm synthetic
operator/positive-block cases, using two warmups and five timed repeats outside
profiler captures. CUDA activity confirmed automatic FP32 memory-efficient
attention on SM75. Forced math SDPA consumed more temporary VRAM and was slower;
it was not enabled in generation. Sparse geometry came from a prefix of the
previous decoded shoe, with random features, not a full pretrained decoder run.
These measurements select candidates; they do not establish asset-level speedup.

Revision **`9a1a9ca891829d83d3feabcd583bd0fad9f236cc`** deployed the selected
profile and completed another fresh single-image 1024/PBR generation through the
private administrative SDK, with authenticated HTTP health/download verification.
The CPU-matted input was again SHA-256
`84f4bc53f07a26a8e649a2db906d6c725ef8fd729765afd03d5301630fc68b12`.
Seed 42, automatic camera, 1024 texture size, 100,000-face advisory target, all
four 12-step schedules, FP32 and a 1024 MiB conditioning cache were retained.
No old neural checkpoint was resumed; no OOM or failed pipeline stage occurred.

| Observed metric | Conservative historical run | Candidate full run |
| --- | ---: | ---: |
| GPU-stage sum, seconds | 644.017488603 | 668.173351617 |
| Model-callback sum, seconds | 604.837014239 | 615.487143261 |
| Peak PyTorch allocated, GiB | 6.804 | 7.010 |
| Peak PyTorch reserved, GiB | 6.945 | 7.139 |
| Exported triangles | 99,123 | 99,051 |
| GLB bytes | 6,504,616 | 6,532,824 |

The candidate is **3.75% slower** in this single GPU-stage observation, on a
different worker/time from the historical reference. **No full-generation
speedup is established.** Model loading, camera and other worker/IO differences
also affect stage totals. Client completion definitions differ (SDK return after
downloads versus HTTP status polling), so their end-to-end figures are not an
equivalent comparison and must not be attributed to chunking.

The final candidate GLB has SHA-256:

```text
0eb1e2d8bdceb05256d53b77499f651daebeec1711253b1761c3920c64e8ba46
```

Both 1024×1024 PNG material maps were independently decoded. Finite/unit-normal
geometry, normalized bounds and zero coincident/nonmanifold/inconsistent indexed
cleanup metrics passed, and Blender 5.1.2 rendered three views with linked
base-color/roughness/metallic maps and no mesh modifiers. Small surface cracks
remain. Structure and refinement-coordinate tensors match the reference exactly;
HR-shape and texture latent relative L2 differences were 1.52e-5 and 3.83e-5.
The decoded voxel count changed from 2,771,717 to 2,771,719 and the GLBs are not
bit-identical. Chunking can change floating-point execution paths; this is not
quality parity or watertightness validation.

To separate configuration effects from different workers, a third bounded probe
loaded the actual pinned HR model once on one T4 with frozen conditioning and
initial noise. One warmup per profile preceded three interleaved timed rounds
(forward/reverse/forward order). Median positive forward times at timestep 1000
were 14,157.56 ms baseline, 14,141.40 ms cross-full, 12,990.76 ms with self queries
1024, and 12,957.29 ms with FFN rows 8192: **8.48% lower** for the final profile.
The warm candidate/reference relative L2 difference was 3.93e-7. This is not the
full schedule, negative CFG passes, model loading or a decoder benchmark, and
does not override the lack of full-generation improvement above.

Records: [operator summary](evidence/modal-t4-chunks-2026-11-13.operators.json),
[paired pretrained forwards](evidence/modal-t4-chunks-2026-11-13.pretrained-forwards.json),
[acceptance](evidence/modal-t4-chunks-2026-11-13.acceptance.json),
[runtime](evidence/modal-t4-chunks-2026-11-13.report.json),
[Blender witness](evidence/modal-t4-chunks-2026-11-13.render.json), and
[CPU tests](evidence/modal-t4-chunks-2026-11-13.tests.txt).
The final CPU suite passed 221 tests with 3 CUDA-only skips; GPU unit tests were
not run in these Apps. Production and all three temporary tuning Apps were
observed stopped with zero active tasks; weight/job Volumes remain. No raw
weights, tensors, prepared image, GLB, PNGs or Chrome traces are committed.

### Remaining validation

- Repeat with populated caches and controlled matching inputs/source before reporting warm-worker latency or optimization speedups.
- Interrupt and resume on actual GPU hardware and compare artifacts; current restart coverage is the controlled-output CPU pipeline test, not a real interrupted T4 run.
- Inspect additional viewpoints and representative complex/thin/reflective inputs; investigate the visible surface/shading artifacts before claiming quality parity.
- Measure more inputs before claiming general 16 GB fit, robustness, or speedup over upstream CUDA.

**Established:** real pretrained single-image runs on Colab and Modal T4 complete through embedded PBR GLB, with measured allocator/RSS peaks. The historical Colab run also passed its GPU tests; Modal verifies transport, cache reuse and the configurable chunk profile. Warm operator and paired positive-HR-forward timings exist. **Not established:** a full-generation speedup, broad input success rate, general memory bounds, quality parity, repeated warm-worker latency, multi-view Modal generation or real-GPU interruption recovery. No performance or capacity guarantee is made.
