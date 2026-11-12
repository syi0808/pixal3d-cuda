# Correctness-first CUDA performance work

This work follows the supplied low-level analysis of baseline `a4cb2b2`.
**No new T4 or Modal performance result is claimed.** The development machine
is Apple ARM64 with CPU-only PyTorch 2.10. GPU speed, VRAM and pretrained GLB
quality must still be measured on real hardware. The MLX checkout is untouched.

## Implemented changes

| Change | Default | Evidence and limits |
| --- | --- | --- |
| Coordinates-only decoder early exit | On | Same predicted child coordinates; skips final two convolutions, child-feature gathers, child topology and trailing feature blocks |
| Coordinates-only guide transfers | Removed | Earlier subdivision masks are not copied to CPU when only coordinates are returned |
| DINO RoPE | Reused | One cosine/sine pair per forward/image resolution, shared by all 24 layers instead of recomputing it in every layer |
| Checkpoint finite checks | Batched | Every materialized floating tensor is still checked after conversion, with one final scalar decision; failures identify the first bad tensor |
| Allocator cleanup | Model boundaries retained | Removes duplicate stage-entry GC/synchronize/empty-cache; still releases each model and synchronizes stage timing |
| Sparse weight prepack | **Off** | `--sparse-prepack 1` rearranges storage once at load; each offset matrix is contiguous, while checkpoint names/shapes/values stay unchanged |
| Conditioning projection cache | **128 MiB unchanged** | Reports actual cached blocks/bytes; larger existing budgets are explicit benchmark candidates, not an automatic VRAM assumption |

FP32 DINO, flows, NAF, sparse decoders, the 512 geometry path, 12-step schedules,
foreground framing and PBR export are unchanged. There is no automatic precision,
resolution, token or step fallback. PyTorch's normal automatic SDPA dispatch
is preserved; explicitly forced benchmark backends never fall back. Dense occupancy remains
FP16 as before. The full shape/texture paths still produce the masks/features
needed for guided texture decoding.

The coordinate fast path skips execution, not checkpoint materialization: the
full decoder checkpoint is still strictly loaded/validated. Its stage cannot
be assumed to save the historical whole-stage time, which also included loading
and downloads.

The finite-check change removes per-tensor **scalar-result reads**, not all
loading synchronization: H2D copies remain blocking. Pinned/asynchronous copies
have not been enabled without lifetime/VRAM validation. The failure path copies
the small vector of finite statuses to CPU to name the bad tensor. No callback
runs before validation succeeds.

Prepacking is an experiment, not a measured speedup. Its temporary allocation
can be as large as one sparse convolution's weight tensor. Include packing/load
time and peak memory in any comparison; it can lose on short-lived workloads.
Keeping it off preserves the previously used GEMM layout until GPU validation.

## Model-level observability

`report.json` now includes a `models` list in addition to `stages`:

- `load_wall_seconds`: local checkpoint materialization, conversion, H2D,
  validation and optional packing, synchronized before the callback.
- `callback_wall_seconds`: the whole callback, including sampling, host gaps,
  transfers and its ending synchronization; **not pure kernel time**.
- `callback_cuda_interval_ms`: CUDA-event elapsed interval for that callback,
  also including host idle gaps between enqueued kernels; absent on CPU.
- `cleanup_wall_seconds`, parameter storage bytes, status and source stage.
- `conditioning_cache`: budget, per-block projection bytes, total/cached blocks
  and actual projection-cache usage for flow models.

CUDA stages also record device-wide free-memory snapshots before/after the
stage. These are not a minimum-free-memory measurement; other processes and
brief allocations can change the available capacity between snapshots.

Downloads occur while resolving the path **before** `run_model`; they are not
included in `load_wall_seconds`. Stage totals still include them. Stage-minus-
model time also contains preprocessing, IO and other work, not just downloads.
Profiler ranges `pixalcuda.load/*`, `pixalcuda.callback/*` and
`pixalcuda.model_cleanup` identify these boundaries when profiling is enabled.

## Warm operator profiling on Colab T4

Use [the operator notebook](../notebooks/colab_t4_performance.ipynb), or run:

```bash
python scripts/benchmark_cuda_ops.py \
  --device cuda --tokens 4096 --channels 1536 --heads 12 \
  --query-chunks 256 512 1024 2048 \
  --warmup 2 --repeats 5 \
  --output outputs/operators-t4.json --trace-dir outputs/operator-traces
```

No model download, HF token or real input image is needed. It tests:

- FP32 global attention chunk/layout sweeps, including one-time Q/K/V layout
  conversion **inside** each candidate's timed call.
- Sparse topology creation separately from gather/GEMM/scatter convolution;
  original versus prepacked weights use the same features and topology.
- A synthetic final decoder upsample: the previous coordinate path versus the
  early exit, using the same weights/input and exact coordinate comparison.
- One static projection reused over 12 evaluations versus 12 recomputations,
  **including the precompute** in cached timing. This is not a full flow run.

Timings run after warmup and outside the profiler. Each optional trace captures
**one representative call** per case, not four complete 12-step flows. The
report stores device/capability, precision controls, source hashes, individual
wall/CUDA-event intervals, allocation baselines/peaks and numerical errors.
All preallocated case inputs coexist, so use incremental allocation as well as
absolute peaks; neither predicts the complete pipeline's VRAM requirement.

`sdpa_operators` records observed dispatch, not hardware capability guesses.
Check `cuda_device_events` / `cuda_activity_captured` before interpreting a trace
as CUDA kernel evidence; CUPTI permissions can prevent device collection even
when CUDA-event timings work. Open Chrome traces with a trace viewer or use
Nsight Systems separately. PyTorch traces do not supply Nsight Compute's L1/L2,
DRAM, occupancy or register-spill counters.

To explicitly test backend availability without silently enabling another one:

```bash
python scripts/benchmark_cuda_ops.py --cases attention \
  --sdpa-backends auto math efficient flash --query-chunks 512 \
  --output outputs/sdpa-backends.json --trace-dir outputs/sdpa-traces
```

Unsupported forced backends are labeled `unsupported`, never promoted to a
timing or replaced with FP16/math. Numerical mismatch or other errors fail the
run and preserve the report. The default device refuses CPU-only machines.
`--device cpu` is explicitly a smoke test and must not be presented as a T4 run.

## Pretrained end-to-end acceptance still required

Prefetch official weights, then use identical input/poses, seed, limits, output
textures and face target. **Warm cache is not `--resume`**: each measurement
needs a fresh output directory and a full generation, otherwise it skips work.
The following baseline is the **current default policy**, not a measured
pre-optimization `a4cb2b2` run.
For example, after installing/authenticating on a real T4:

```bash
pixal3d-cuda download
HF_HUB_OFFLINE=1 pixal3d-cuda generate --image object.png \
  --output-dir outputs/warm-baseline --conditioning-cache-mb 128
HF_HUB_OFFLINE=1 pixal3d-cuda generate --image object.png \
  --output-dir outputs/warm-cache1024 --conditioning-cache-mb 1024
HF_HUB_OFFLINE=1 pixal3d-cuda generate --image object.png \
  --output-dir outputs/warm-prepacked --conditioning-cache-mb 128 --sparse-prepack 1
```

Also sweep 512/2048/3072 MiB if observed free memory permits. Do not infer safe
budgets from the older single-shoe 5.45 GiB figure. Report active tokens/voxels,
load/callback/cleanup, total time, allocated/reserved/free memory, and actual
geometry/PBR differences. Compare small, medium and large inputs; reject any
OOM, invalid GLB or visible quality regression before changing service defaults.
For Modal, keep CPU preprocessing/export separate; HTTP has no new profiling
or client-controlled GPU/precision flags. Paid cloud benchmarks are not launched
automatically by these tools or CI.

## Current validation

CPU tests verify exact child-coordinate equivalence, preservation of empty/
voxel guards, skipped final convolutions/topology/CPU mask copies, shared DINO
phases, 700 finite flags with one scalar decision, sparse storage/value/output
equivalence, bounded cache metrics and model telemetry. CUDA-only loader and
attention tests run only when hardware is available. A small synthetic CPU
operator smoke was also run locally; its timings are not GPU estimates.

Tiled/fused sparse kernels, packed neighbor maps, selective FP16, CUDA graphs,
default attention-layout changes and CPU rasterizer rewrites remain deferred
until real GPU profiles identify the next bottleneck.

References: [PyTorch 2.6 profiler](https://docs.pytorch.org/docs/2.6/profiler.html),
[SDPA backend/numerical behavior](https://docs.pytorch.org/docs/2.6/generated/torch.nn.functional.scaled_dot_product_attention.html).
