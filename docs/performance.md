# Correctness-first CUDA performance work

This work follows the supplied low-level analysis of baseline `a4cb2b2`.
The development machine is Apple ARM64 with CPU-only PyTorch 2.10. Real Modal
T4 acceptance, cache comparisons, warm operator probes and a paired pretrained
forward study are now recorded below. **No full-generation speedup or broad
capacity guarantee is established.** Broader GPU
speed, VRAM and pretrained GLB quality still require measurement. The MLX
checkout is untouched.

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

## Controlled full-generation comparison still required

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

On 2026-11-13, revision `f30dfc4` completed a fresh pretrained single-image
Modal T4 run with default limits, FP32 flows/sparse decoders, all 12-step schedules,
128 MiB conditioning cache and sparse prepacking disabled. Weights were prefetched
on CPU; GPU workers were offline. GPU stage totals were 246.92 s, including local
checkpoint loading/cleanup and GPU-side camera inference, not just kernels.
CPU export took 60.69 s; HTTP completion was observed after 464.97 s, including
cold workers/preprocessing and polling delay. PyTorch allocator peaks were
5.46 GiB allocated / 5.50 GiB reserved; sampled GPU-worker process RSS peaked at
10.45 GiB. This is one acceptance case, not repeated warm-worker latency or a
controlled comparison with the download-inclusive historical Colab run.
See [acceptance and quality limits](validation.md#modal-t4-current-runtime)
and the per-model load/callback/cleanup records linked there. Synthetic CUDA
operator sweeps, prepack alternatives and larger-cache measurements remain pending.

### 1024 geometry: 512 versus 1024 MiB cache

Revision `8c54fbf` completed two fresh runs on Modal T4 using the identical
CPU-matted public shoe, seed 42, 1024 geometry, 1024 PBR textures, 100,000-face
advisory target and unchanged chunk sizes/precision/12-step schedules. No neural
checkpoint was resumed. The 1024 profile explicitly uses 32,768-token and
6,000,000-voxel guards; actual refinement tokens were 10,305 and final shape/texture
voxels were 2,771,717. Both completed without OOM or resolution fallback.

| Observed metric | 512 MiB | 1024 MiB |
| --- | ---: | ---: |
| GPU-stage sum, seconds | 653.65 | 644.02 |
| Model-callback sum, seconds | 614.60 | 604.84 |
| Peak PyTorch allocated, GiB | 6.333 | 6.804 |
| Peak PyTorch reserved, GiB | 6.477 | 6.945 |
| Cached SS / LR / HR / texture blocks, out of 30 each | 21 / 30 / 8 / 8 | 30 / 30 / 16 / 16 |

The observed GPU-stage difference is **1.47%**, with about 0.472 GiB additional
peak allocation. Each budget was sampled once, sequentially, with separate
scale-to-zero workers: this is not a statistically established speedup. Both GLBs
are byte-identical, and structure/coordinate/HR-shape/texture checkpoint tensors
are exactly equal. The HR/texture projection cache would need about 1811.43 MiB
to cover all 30 blocks for this input; that larger budget was **not tested**.

The larger cache does not remove self-attention/FFN work. HR-shape and texture
flows still account for most GPU time, so profile inside those stages before
claiming a specific kernel bottleneck or choosing the next optimization. No
attention/FFN chunk sweep, prepack change, warm-model residency or extra paid
benchmark was performed. Default cache remains 128 MiB; CLI, administrative SDK
and HTTP now accept explicit budgets. See
[full conditions, end-to-end timing and evidence](validation.md#modal-t4-1024-geometry-cache-comparison).

### Chunk profile trials on T4

The server declarations apply self-attention queries 1024, independent
cross-attention queries 0 (all queries), FFN rows 8192 and sparse rows 16384.
CLI defaults remain 256/256/1024/4096. FP32, full-key attention, checkpoints,
12-step schedules, geometry and cache budget are not reduced by this change.

One-shot `main` tuning Apps completed 36 warm operator/positive-block cases,
each with two warmups and five timed repeats. Source workloads used 10,305 tokens,
1536 channels, 12 heads, hidden width 8192 and five global context tokens.
Sparse tests used random 64-channel features on a 262,144-coordinate prefix of
the frozen decoded geometry, not the entire decoder.

| Warm synthetic case | Conservative | Candidate |
| --- | ---: | ---: |
| Cross attention, 256 versus all queries | 3.05 ms | 2.61 ms |
| Self attention, 256 versus 1024 queries, strided layout | 242.04 ms | 200.53 ms |
| FFN, 1024 versus 8192 rows | 117.94 ms | 117.68 ms |
| Positive block, all chunk changes except sparse | 454.39 ms | 417.99 ms |
| Sparse convolution, 4096 versus 16384 rows | 111.60 ms | 58.21 ms |

Cross-only and FFN-only block differences were small; do not multiply these
operator ratios into a full-generation forecast. CUDA profiler activity captured
the FP32 SM75 CUTLASS memory-efficient attention kernel. In a separate same-shape
probe, automatic SDPA at query chunk 1024 took 201.44 ms with 72.38 MiB incremental
allocation; forced math took 356.40 ms with 1159.25 MiB. The higher-memory math
backend was rejected. Copying Q/K/V into contiguous layout did not win in these
tested candidates and no new layout policy was enabled.

A further paired study loaded the actual pinned HR model once and used identical
frozen conditioning, initial noise and prepared 1024 MiB projection cache.
After one warmup/profile, three timed rounds alternated forward/reverse/forward
profile order. Median positive-forward times at timestep 1000 were:

| Profile | Median |
| --- | ---: |
| Conservative baseline | 14,157.56 ms |
| Cross-query separation only | 14,141.40 ms |
| Plus self queries 1024 | 12,990.76 ms |
| Plus FFN rows 8192 | 12,957.29 ms |

The final profile is 8.48% lower in this paired forward test; FFN's incremental
difference is small. This is one positive forward, not all timesteps, negative
CFG passes, model loads or a full asset. Sparse rows have no role in this HR-flow
study and were tested separately. Larger/smaller query candidates and the complete
block matrix remain in the recorded operator summary.

**End-to-end evidence does not establish a speedup.** A fresh full 1024/PBR run
at `9a1a9ca` completed without OOM at 7.01 GiB peak allocation, but GPU stages
took 668.17 s versus the previous different-worker 644.02 s. Model callbacks were
615.49 versus 604.84 s. Worker/cold-start/IO differences are not controlled here;
do not claim that the paired 8.48% forward result applies to generation. The
larger-chunk profile is a configurable candidate, not a universally faster or
all-input-safe default. Detailed conditions, numerical changes and evidence are
in [validation](validation.md#modal-t4-chunk-profile-trials).

Tiled/fused sparse kernels, packed neighbor maps, selective FP16, CUDA graphs,
default attention-layout changes and CPU rasterizer rewrites remain deferred
until real GPU profiles identify the next bottleneck.

References: [PyTorch 2.6 profiler](https://docs.pytorch.org/docs/2.6/profiler.html),
[SDPA backend/numerical behavior](https://docs.pytorch.org/docs/2.6/generated/torch.nn.functional.scaled_dot_product_attention.html).
