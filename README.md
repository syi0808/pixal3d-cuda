# pixal3d-cuda

[![CPU contracts](https://github.com/syi0808/pixal3d-cuda/actions/workflows/tests.yml/badge.svg)](https://github.com/syi0808/pixal3d-cuda/actions/workflows/tests.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/syi0808/pixal3d-cuda/blob/main/notebooks/colab_t4_512.ipynb)

A separate, memory-bounded **PyTorch/CUDA Pixal3D runtime targeting Colab T4**, with **512 geometry by default, opt-in 1024 geometry**, and **embedded PBR-textured GLB** output. No dependency on the MLX checkout; no training or replacement model weights.

**Experimental, with real Colab and Modal T4 acceptance:** the 512-geometry Modal run generated the pinned shoe through the authenticated HTTP API, offline T4 inference, CPU export and GLB download. With weights prefetched, GPU stages took **246.92 s**, CPU export **60.69 s**, and the client observed completion after **464.97 s**, including cold workers and preprocessing. Peak PyTorch allocation/reservation was **5.46/5.50 GiB**. The GLB embeds 1024×1024 base-color and metallic/roughness textures. A later 1024-geometry cache comparison also completed, as recorded below. Small visible surface cracks remain; this is not production-quality, general 16 GB fit or quality-parity validation. See [full evidence and limitations](docs/validation.md).

The historical Colab run used `dacb3a1`; the first fresh Modal run used `f30dfc4`, including [mesh-artifact fixes](docs/mesh-artifacts.md) and the current default optimizations. These runs are not a controlled performance comparison. The single-image Colab notebook remains pinned to its previously validated release; a fresh Colab-specific rerun is still pending.

## Colab quick start

1. Open the notebook using the badge; select **Runtime → Change runtime type → T4 GPU** and use a fresh session.
2. Run installation and GPU/operator preflight cells. The notebook pairs PyTorch 2.6.0, torchvision 0.21.0, and CUDA 12.4.
3. Obtain access to [official DINOv3 weights](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m). Add a read-only `HF_TOKEN` to Colab Secrets, enable notebook access, and authenticate in the notebook.
4. Download/validate pinned models, upload one image, and run generation. Allow roughly 25–30 GB for model downloads and start with 40 GiB free disk space.
5. Preview/download `result.glb` and inspect `report.json`. Enable the notebook's optional Google Drive storage for durable stage checkpoints.

[한국어 실행 안내](docs/colab-ko.md)

## What is implemented

- Native PyTorch DINOv3 ViT-L/16 with the **non-affine final normalization** expected by Pixal3D.
- Exact pixel projection and sparse NAF neighborhood queries, including shifted border windows. No bilinear-only substitute for NAF.
- Full global self-attention with **query chunking**, not local attention or token dropping. PyTorch selects a compatible SDPA backend; its math fallback is bounded by the query chunk.
- FP32 DINOv3, flow models, sparse decoders, CFG/sampler state and NAF; FP16 dense occupancy decoder. The first real T4 attempts exposed FP16 overflows in DINO and LR sampling, so the bounded runtime prioritizes a stable FP32 baseline rather than claiming unvalidated mixed-precision quality.
- Tensor-at-a-time safetensors loading into meta-initialized modules. No full state-dict copy and no all-model CPU/GPU residency.
- Sampling-scoped global K/V and bounded projected-conditioning caches; exact zero-negative-context folding.
- Reused sparse topology and bounded gather/GEMM submanifold convolution. Guided texture subdivision uses the shape decoder's masks.
- CPU dual-grid extraction, simplification, xatlas UV unwrap, trilinear PBR sampling/seam filling, and embedded base-color plus metallic/roughness images.
- Atomic stage checkpoints, strict resume fingerprints, finite-state checks, explicit token/voxel guards, and per-stage GPU/RSS telemetry.

No required FlashAttention, NATTEN, Triton, FlexGEMM, spconv, nvdiffrast, or locally compiled CUDA extensions. The portable sparse backend prioritizes compatibility and bounded memory; it is **not claimed faster than optimized CUDA sparse libraries**.

## The 512 cascade

```text
image → foreground crop → MoGe-2 camera
      → DINOv3 features at 512 and 1024
      → sparse structure flow + occupancy decode
      → LR shape flow (512 checkpoint)
      → decoder subdivision → requantize to 32³ latent grid
      → shape refinement (1024 checkpoint on the 512 output grid)
      → shape decode → mesh
      → texture flow (1024 checkpoint on that same latent grid)
      → guided texture decode → UV/PBR bake → result.glb
```

512 denotes the geometry grid, **not** a dense 512³ allocation, input resolution, or texture size. The implementation preserves the MLX fork's established 512 cascade, including 1024-named refinement/texture weights; it does not simply remove the refinement stage. Single-image texture conditioning reuses shape HR's NAF 512 setting, unlike upstream's dedicated NAF 1024 texture conditioning; mapped multi-view uses a separate NAF 1024 texture context. Export/remeshing defaults also differ. End-to-end quality parity with upstream has not been established. See the [upstream/MLX/CUDA architecture audit](docs/architecture-comparison.md) for model residency, dense-versus-sparse features, memory accounting and correctness gaps.

All four flow schedules retain the official 12 steps. No quantization, automatic resolution fallback, geometry-token pruning, or step reduction. On an explicit resource guard, generation fails with a report rather than silently changing the requested output.

The opt-in `--geometry-resolution 1024` uses a 64³ refinement latent grid,
64-grid camera projection, and 1024-grid mesh extraction/texture sampling with
the same pinned refinement/texture checkpoints. Geometry and texture size are
independent options. The exported GLB records its geometry grid in mesh metadata;
validation no longer assumes every file came from 512 geometry. Legacy GLBs
without that metadata report an unknown grid rather than an inferred value.
For local CUDA generation, explicitly review/increase guards for the larger grid:

```bash
pixal3d-cuda generate --image object.png --geometry-resolution 1024 \
  --conditioning-cache-mb 512 --max-tokens 32768 --max-voxels 6000000 \
  --output-dir outputs/geometry1024-cache512
```

Larger guards are not a memory guarantee. Use a fresh output directory for each
cache/geometry setting; resume fingerprints include both. On Modal T4, the same
1024-geometry shoe completed with 512/1024 MiB caches: GPU stages took
653.65/644.02 s and peak allocation was 6.33/6.80 GiB. Both GLBs have identical
SHA-256. This is one sequential observation per budget, not a statistical
speedup guarantee or a general memory bound. See
[the cache comparison](docs/validation.md#modal-t4-1024-geometry-cache-comparison).

## Linux CLI

Use a Python 3.11/3.12 environment with a matching NVIDIA driver. For the same stack as the notebook:

```bash
git clone https://github.com/syi0808/pixal3d-cuda.git
cd pixal3d-cuda
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[dev,preprocess,camera]'
hf auth login
pixal3d-cuda doctor
pixal3d-cuda download
python scripts/verify_checkpoints.py
pixal3d-cuda generate --image object.png --output-dir outputs/first
```

`hf auth login` is provided by recent `huggingface-hub`; the notebook also supports its interactive login. The token and model weights are never committed or embedded in reports. Background removal uses a CPU ONNX session; MoGe-2 executes separately from the flow models. Camera code is pinned before MoGe-3's FlexGEMM dependencies.

To repeat the public single-image acceptance case on a real T4 after installation/authentication:

```bash
python -u scripts/validate_colab_t4.py --output-dir outputs/t4-acceptance
```

It checks T4 hardware and the pinned input hash, then writes `acceptance.json` plus `run/result.glb` and `run/report.json`. Use a fresh output directory when upgrading from the earlier FP16 runtime; old checkpoints have a different precision/source fingerprint.

For an already isolated RGBA input, use `--no-rembg`. To explicitly skip MoGe and choose a horizontal FOV in degrees, use `--camera fixed --fov-degrees 49.13`; fixed camera can change output quality and is never an automatic fallback.

Resume with the same original input and settings:

```bash
pixal3d-cuda generate --image object.png --output-dir outputs/first --resume
```

Settings/input mismatch is rejected. Use a new output directory for a different seed or setting. Resume checkpoints contain the input-derived tensors, so treat them as private. Put the output directory on Drive if it must survive Colab runtime loss. Interrupted model downloads use the normal Hugging Face cache; the cache also needs persistent storage to survive runtime deletion.

## Mapped image/camera input and Modal

The experimental CUDA multi-view path accepts N images mapped one-to-one to
explicit cameras (indexed frames; orbit parameters or c2w matrices), uses the dedicated pinned MV flow weights, and averages
camera-projected conditioning without running four separate generations.
Framing must match the supplied calibration; arbitrary four photos are not
automatically calibrated. See [MV input and validation](docs/multiview.md) and
[the experimental Colab T4 MV notebook](notebooks/colab_t4_multiview_512.ipynb).
The old `--views-dir` format remains supported; the new form is
`--images a.png b.png --transforms cameras.json`. There is no four-view model
limit; Modal separately defaults to a 16-view operational limit.
Real pretrained multi-view CUDA generation remains **unvalidated**.

[The Modal adapter](docs/modal.md) separates CPU prefetch/matting,
offline GPU inference, and CPU mesh/UV/PBR export, with scale-to-zero and
content-addressed job reuse. A single-image `main` deployment and real T4 PBR
generation were verified on 2026-11-13; multi-view Modal generation remains unvalidated.
Infrastructure names, GPU/memory/timeouts and upload limits live in
`deploy/modal.toml`; `--action plan` prints the local declarations without
deploying. A manual-only GitHub Actions workflow deploys only to `main`, with no
separate development environment. Its Proxy Token-authenticated HTTPS API
supports asynchronous job submission, status polling and GLB/report downloads.
For the same GPU/CPU split outside Modal:

```bash
pixal3d-cuda generate --views-dir inputs/object --output-dir outputs/mv --defer-export
pixal3d-cuda export --output-dir outputs/mv
```

Transfer the entire output directory for CPU export, using the same runtime
source. Intermediate checkpoints are private input-derived data.

## Memory and performance controls

| Option | Default | Effect |
| --- | ---: | --- |
| `--attention-queries` | 256 | Bound query rows; all keys remain visible |
| `--cross-attention-queries` | 256 | Independent flow cross-attention query rows; 0 handles all queries together |
| `--ffn-rows` | 1024 | Bound FFN intermediate rows |
| `--sparse-rows` | 4096 | Bound convolution gather/GEMM rows |
| `--naf-queries` | 128 | Bound projected NAF queries |
| `--conditioning-cache-mb` | 128 | Bound static projection cache; 0 disables it |
| `--sparse-prepack` | 0 | Experimental 1=contiguous per-offset weights at load; extra temporary memory |
| `--max-tokens` | 16384 | Fail before flow on oversized sparse sets |
| `--max-voxels` | 1500000 | Fail before oversized subdivision |
| `--texture-size` | 1024 | 512/1024/2048 PBR output textures |
| `--geometry-resolution` | 512 | Geometry grid: 512 or experimental 1024; independent of texture size |
| `--target-faces` | 100000 | Simplification target, not a strict triangle cap |

Smaller chunks can reduce temporary allocations but take longer. Guards are not allocator limits; increasing them may OOM. Neither PyTorch memory reports nor this runtime promise a hard GPU/process memory cap. `report.json` separates per-stage peak allocated/reserved CUDA bytes from sampled process RSS. RSS sampling can miss brief peaks. Different devices/backends are not bitwise reproducible.

Correctness-first optimizations now skip unused final coordinate-decoder work,
reuse DINO RoPE, batch checkpoint finite-status decisions, and remove duplicate
stage-entry allocator flushes. Per-model load/callback/cleanup timing and flow
cache usage are recorded. Sparse prepacking remains opt-in; precision, sampling
and default cache budgets are unchanged. See [performance scope and warm GPU
profiling](docs/performance.md) and the [Colab operator notebook](notebooks/colab_t4_performance.ipynb).
No full-generation speedup or pretrained quality-parity result is claimed.

The Modal chunk profile now uses self-attention queries 1024, cross-attention
queries 0 (all queries), FFN rows 8192 and sparse rows 16384. The CLI retains
conservative defaults; use `--attention-queries 1024 --cross-attention-queries 0
--ffn-rows 8192 --sparse-rows 16384` to opt in locally. On one T4 with one loaded
pretrained model and interleaved repeats, a frozen positive HR forward improved
by 8.48%. **A full 1024/PBR generation did not improve against the historical
different-worker run**: GPU stages were 668.17 versus 644.02 s, with allocation
7.01 versus 6.80 GiB. The candidate generated a valid GLB but not an identical
file. These are scoped experiments, not a universal fast preset. See
[the chunk trials](docs/validation.md#modal-t4-chunk-profile-trials).

## Tests

```bash
python -m pytest -q
```

CPU tests cover global-attention equivalence, RoPE, CFG, negative-context folding, sparse-vs-dense convolution, subdivision/coordinate order, checkpoint strictness, resume/failure reports, NAF/projection, notebook schema, and actual UV bake/PBR GLB reload. CUDA attention tests run when a GPU is present and otherwise skip. These are not substitutes for pretrained generation/visual validation.

## Attribution and licenses

Runtime/model semantics follow [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D), [TRELLIS.2](https://github.com/microsoft/TRELLIS.2), [DINOv3](https://github.com/facebookresearch/dinov3), [NAF](https://github.com/valeoai/NAF), and [MoGe-2](https://github.com/microsoft/MoGe). The 512 cascade, native NAF/DINO port reference, CPU mesh extraction and texture-baking helpers derive from [Noah Lyons' MLX port](https://github.com/lyonsno/pixal3d-mlx) and [syi0808's fork](https://github.com/syi0808/pixal3d-mlx). The original MIT notice is retained in [LICENSE](LICENSE).

Code licensing does not grant additional rights to model weights. Review each upstream checkpoint license, including DINOv3 and any background-removal model, before use or redistribution. No weights, credentials, user inputs, or generated assets are shipped in this repository.
