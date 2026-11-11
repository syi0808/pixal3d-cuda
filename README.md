# pixal3d-cuda

[![CPU contracts](https://github.com/syi0808/pixal3d-cuda/actions/workflows/tests.yml/badge.svg)](https://github.com/syi0808/pixal3d-cuda/actions/workflows/tests.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/syi0808/pixal3d-cuda/blob/main/notebooks/colab_t4_512.ipynb)

A separate, memory-bounded **PyTorch/CUDA Pixal3D runtime targeting Colab T4**, with fixed **512 geometry** and **embedded PBR-textured GLB** output. No dependency on the MLX checkout; no training or replacement model weights.

**Experimental:** operator/CPU/export tests are available. Full pretrained image-to-GLB generation on an actual T4 has **not yet been measured or quality-validated**. Do not interpret this repository as a guaranteed 16 GB fit, a benchmark, or demonstrated quality parity. The Colab notebook is the reproducible hardware validation path. See [validation status](docs/validation.md).

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

512 denotes the geometry grid, **not** a dense 512³ allocation, input resolution, or texture size. The implementation preserves the MLX fork's established 512 cascade, including 1024-named refinement/texture weights; it does not simply remove the refinement stage. NAF output sampling uses the reference fork's bounded 512 feature setting. Image preprocessing and the FP16 CUDA path still require end-to-end quality comparison.

All four flow schedules retain the official 12 steps. No quantization, automatic resolution fallback, geometry-token pruning, or step reduction. On an explicit resource guard, generation fails with a report rather than silently changing the requested output.

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

For an already isolated RGBA input, use `--no-rembg`. To explicitly skip MoGe and choose a horizontal FOV in degrees, use `--camera fixed --fov-degrees 49.13`; fixed camera can change output quality and is never an automatic fallback.

Resume with the same original input and settings:

```bash
pixal3d-cuda generate --image object.png --output-dir outputs/first --resume
```

Settings/input mismatch is rejected. Use a new output directory for a different seed or setting. Resume checkpoints contain the input-derived tensors, so treat them as private. Put the output directory on Drive if it must survive Colab runtime loss. Interrupted model downloads use the normal Hugging Face cache; the cache also needs persistent storage to survive runtime deletion.

## Memory and performance controls

| Option | Default | Effect |
| --- | ---: | --- |
| `--attention-queries` | 256 | Bound query rows; all keys remain visible |
| `--ffn-rows` | 1024 | Bound FFN intermediate rows |
| `--sparse-rows` | 4096 | Bound convolution gather/GEMM rows |
| `--naf-queries` | 128 | Bound projected NAF queries |
| `--conditioning-cache-mb` | 128 | Bound static projection cache; 0 disables it |
| `--max-tokens` | 16384 | Fail before flow on oversized sparse sets |
| `--max-voxels` | 1500000 | Fail before oversized subdivision |
| `--texture-size` | 1024 | 512/1024/2048 PBR output textures |
| `--target-faces` | 100000 | Simplification target, not a strict triangle cap |

Smaller chunks can reduce temporary allocations but take longer. Guards are not allocator limits; increasing them may OOM. Neither PyTorch memory reports nor this runtime promise a hard GPU/process memory cap. `report.json` separates per-stage peak allocated/reserved CUDA bytes from sampled process RSS. RSS sampling can miss brief peaks. Different devices/backends are not bitwise reproducible.

## Tests

```bash
python -m pytest -q
```

CPU tests cover global-attention equivalence, RoPE, CFG, negative-context folding, sparse-vs-dense convolution, subdivision/coordinate order, checkpoint strictness, resume/failure reports, NAF/projection, notebook schema, and actual UV bake/PBR GLB reload. CUDA attention tests run when a GPU is present and otherwise skip. These are not substitutes for pretrained generation/visual validation.

## Attribution and licenses

Runtime/model semantics follow [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D), [TRELLIS.2](https://github.com/microsoft/TRELLIS.2), [DINOv3](https://github.com/facebookresearch/dinov3), [NAF](https://github.com/valeoai/NAF), and [MoGe-2](https://github.com/microsoft/MoGe). The 512 cascade, native NAF/DINO port reference, CPU mesh extraction and texture-baking helpers derive from [Noah Lyons' MLX port](https://github.com/lyonsno/pixal3d-mlx) and [syi0808's fork](https://github.com/syi0808/pixal3d-mlx). The original MIT notice is retained in [LICENSE](LICENSE).

Code licensing does not grant additional rights to model weights. Review each upstream checkpoint license, including DINOv3 and any background-removal model, before use or redistribution. No weights, credentials, user inputs, or generated assets are shipped in this repository.
