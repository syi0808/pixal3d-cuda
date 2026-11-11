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

## Required T4 acceptance run

Run all notebook cells on an allocated T4, not merely on a CPU or another GPU:

1. Record GPU name, compute capability, driver, torch/CUDA versions, and available RAM/disk.
2. Pass FP16 and FP32 CUDA attention tests and strict checkpoint metadata verification.
3. Generate from at least one clean single-object input with the default 12-step schedules and requested 512 geometry.
4. Verify a nonempty finite GLB with both embedded PBR textures and inspect geometry/materials from multiple viewpoints.
5. Retain report, input hash, source revision, seed, timing, allocated/reserved CUDA peaks and sampled CPU RSS.
6. Interrupt and resume a second run without changing settings; compare final artifacts/visuals.
7. Repeat representative complex/thin/reflective inputs before claiming general 16 GB fit or quality parity.

**Not yet established:** end-to-end pretrained T4 completion, latency, peak VRAM, robustness, quality parity with the reference MLX fork, or speedup over upstream CUDA. Passing contracts/metadata checks does not establish these claims. No performance or capacity guarantee is made.
