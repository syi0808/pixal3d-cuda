# Validation status

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

### Remaining validation

- Repeat with populated caches and controlled matching inputs/source before reporting warm-worker latency or optimization speedups.
- Interrupt and resume on actual GPU hardware and compare artifacts; current restart coverage is the controlled-output CPU pipeline test, not a real interrupted T4 run.
- Inspect additional viewpoints and representative complex/thin/reflective inputs; investigate the visible surface/shading artifacts before claiming quality parity.
- Measure more inputs before claiming general 16 GB fit, robustness, or speedup over upstream CUDA.

**Established:** real pretrained single-image runs on Colab and Modal T4 complete through embedded PBR GLB, with measured allocator/RSS peaks. The historical Colab run also passed its GPU tests; the Modal run verifies the HTTP/CPU/GPU/export/download chain and completed-job reuse. **Not established:** broad input success rate, general memory bounds, quality parity, repeated warm-worker latency, controlled speedup, multi-view Modal generation or real-GPU interruption recovery. No performance or capacity guarantee is made.
