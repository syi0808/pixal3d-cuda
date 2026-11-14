# pixal3d-cuda

[![CPU contracts](https://github.com/syi0808/pixal3d-cuda/actions/workflows/tests.yml/badge.svg)](https://github.com/syi0808/pixal3d-cuda/actions/workflows/tests.yml)

## v0.2: official Pixal3D first

The default runtime now delegates to **unmodified, commit-pinned
[TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D/tree/f7cf38429b0bd264f1995f0f8743a88b1c728b94)**:
official DiTs/samplers, DINOv3 + NAF conditioning, sparse decoders, CuMesh hole
filling and `o_voxel` remesh/PBR export. No MLX-derived neural operators or CPU
mesh exporter are used by this path. Job-scoped DINO reuse, active projection and
bounded static conditioning reuse are now enabled; `--reference` disables these
hooks in the same official pipeline. The MLX checkout is not modified.

**Status: actual Modal L4 single-image 1024/4K PBR generation passed for both
reference and optimized profiles. The one-pair performance pilot did not pass
strict native tensor parity; it does not establish equivalent-quality speedup.
RMBG-2.0 is disabled/optional,
not a default download or initialization dependency.** See
[measurements and remaining checks](docs/performance.md).
Follow-up fixed-kernel profiling passes **149 full neural checkpoints exactly**
for reference repeat/optimized on this shoe, including HR/texture and raw mesh/PBR
voxels. Native export remains non-bit-identical. See [scope and bottlenecks](docs/full-profiling.md).
Covered checkpoint initialization skipping and exact parallel inpaint are now
enabled. L4 CPU 2/4/8 trials select an 8-core GPU worker/4 inpaint workers for
latency; 4/8-core costs are near break-even in cold-asset sensitivity.
[Measured optimization and rollout](docs/loading-and-cpu-tuning.md).
Historical T4 GLBs/timings belong to the
archived implementation and are not acceptance evidence for this rewrite.

| Baseline choice | Value |
| --- | --- |
| Geometry | 1024 default; official 1536 cascade also selectable |
| Texture / face target | 4096 / 1,000,000, matching official CLI defaults |
| Model residency | Official low-VRAM mode; `--standard` opts into all-model residency |
| Attention / sparse conv | PyTorch SDPA / native FlexGEMM |
| Precision / steps | Official BF16 flow + FP16 decoder torso; original 12-step schedules |
| Single / mapped images | Original SV / dedicated MV pipelines and weights |
| Export | Original CUDA remesh + embedded base-color and metallic/roughness GLB |
| Initial deployment target | **L4 / SM89**, declared explicitly; no automatic GPU fallback |
| Reuse budgets | DINO 128 MiB; conditioning 256 MiB per flow, configurable |
| Loading / CPU export | Complete-coverage no-init; GPU worker 8 CPU cores / inpaint 4; API/prepare CPU unchanged |

**T4 is not accepted by the unmodified BF16 baseline.** Re-enabling T4 requires
a separately measured precision/kernel compatibility project, not silently
changing precision silently. Free Colab T4 is therefore not a supported v0.2 target.
L4 fit is verified for one prepared shoe, not all inputs or mapped views. See
[migration, constraints and optimization investigation](docs/upstream-runtime.md).

## Native setup

On Linux with CUDA Toolkit 12.4, an SM80/86/89 GPU, sufficient host RAM/disk and a
Python 3.12 environment:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[upstream,camera]'
pixal3d-cuda setup-upstream --build --cuda-arch 8.9
hf auth login
pixal3d-cuda download
pixal3d-cuda doctor
pixal3d-cuda generate --image object.png --output-dir outputs/native-reference --reference --save-reference
```

Use `8.0`/`8.6` for corresponding Ampere GPUs. Setup clones locked dependency
sources outside this repository, then explicitly compiles native extensions.
It does not launch cloud compute. The CUDA 12.4/Python 3.12/SM89 recipe built on Modal;
this is not validation of every local/Colab environment or pretrained execution.
The upstream example wheels target another GPU/Python ABI and are deliberately
not reused. Model access and each model's license must be reviewed separately.
Default inputs must already be prepared/framed; RGBA alpha is composited on black
without automatic crop/matting. See [optional background removal and license opt-in](docs/background-removal.md).
The wrapper pins local model paths and the NAF source;
it does not change their network computations.

Mapped inputs retain the image/camera mapping API:

```bash
pixal3d-cuda download --multiview --skip-camera
pixal3d-cuda generate --images front.png right.png back.png left.png \
  --transforms cameras.json --output-dir outputs/native-mv
```

There is no four-view limit. Frame ordering, FOV, camera matrices and image
framing must agree. Input validation/normalization is reused, but upstream
`load_views`/`run_mv` performs actual conditioning and inference.

## Validate And Measure

See [implemented optimizations, controls and validation limits](docs/optimizations.md).
The L4 pilot measured 606.28 s reference versus 530.44 s optimized stage sums,
excluding diagnostic tensor saving, queue/startup and transport. Denoise/decode
was effectively unchanged at 160 s, and native tensor shapes differed.
See [the scope and parity failure](docs/performance.md) before interpreting these
numbers as an optimization benefit.

```bash
python scripts/benchmark_upstream.py --image object.png
python scripts/benchmark_upstream.py --image object.png --execute --output-dir outputs/native-bench
```

Without `--execute` this prints a plan and never initializes CUDA, downloads
models, submits jobs or contacts Modal. Execution alternates the reference and
optimized official low-VRAM/SDPA profiles by default; `standard` is selectable.
There is no precision/resolution fallback. It stops on
failure/OOM instead of excluding failed cases. The one-pair L4 pilot is not a
repeated, equivalent-output speedup result.

Each native generation produces `result.glb`, `report.json`, `upstream.log` and
`upstream-request.json`. Optional `--save-reference` writes **private** native
latents and pre-remesh geometry/voxel tensors to `reference.pt`; these can expose
input-derived data. Reports identify requested/actual geometry, conditioning
sizes, source/model pins, stage timings and PyTorch allocated/reserved peaks.
Device-used VRAM (including allocations outside PyTorch) and process RSS are
also sampled every 100 ms; these sampled peaks can miss shorter spikes.
Native reports are a new schema, not compatible with old stage-resume reports.

## Modal and compatibility

[Modal main declarations](docs/modal.md) preserve authenticated asynchronous
upload/status/GLB/report endpoints. GPU inference now **includes native export**;
the old CPU matting/export phases are not used. The declaration targets L4 and
stays scale-to-zero. Deploy/prefetch/generation are explicit manual operations.

Breaking changes: geometry 512 and previous projection-cache settings are
rejected; defaults are 1024 geometry / 4096 texture / 1M face target; partial
stage resume and deferred CPU export are unsupported. Completed matching results
can be reused with `--resume`. No silent downgrade or alternative runtime exists.
The former neural operators, CPU mesh/PBR exporter, T4 notebooks, tuning adapter
and associated implementation-specific tests have been removed, not hidden behind
a backend switch.

See [historical acceptance evidence](docs/validation.md) and the
[pre-rebase architecture audit](docs/architecture-comparison.md).
Prior code and notebooks are available only through Git history; there is one
runtime and one native baseline notebook in the current tree.

## CPU contracts

```bash
python -m pip install -e '.[dev,modal,server]'
python -m pytest -q
```

Controlled upstream doubles verify wiring and policy, **not native kernel,
pretrained quality, full GLB parity, GPU capacity or deployment correctness**.
