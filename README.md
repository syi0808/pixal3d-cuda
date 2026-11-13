# pixal3d-cuda

[![CPU contracts](https://github.com/syi0808/pixal3d-cuda/actions/workflows/tests.yml/badge.svg)](https://github.com/syi0808/pixal3d-cuda/actions/workflows/tests.yml)

## v0.2: official Pixal3D first

The default runtime now delegates to **unmodified, commit-pinned
[TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D/tree/f7cf38429b0bd264f1995f0f8743a88b1c728b94)**:
official DiTs/samplers, DINOv3 + NAF conditioning, sparse decoders, CuMesh hole
filling and `o_voxel` remesh/PBR export. No MLX-derived neural operators or CPU
mesh exporter are used by this path. The MLX checkout is not modified.

**Status: CPU orchestration contracts tested; native image build and real GPU
generation are NOT yet validated.** Historical T4 GLBs/timings belong to the
archived implementation and are not acceptance evidence for this rewrite.
No cloud GPU or Modal deployment was started for the rebase.

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

**T4 is not accepted by the unmodified BF16 baseline.** Re-enabling T4 requires
a separately measured precision/kernel compatibility project, not silently
changing precision silently. Free Colab T4 is therefore not a supported v0.2 target.
L4 is an unvalidated build target, not a fit/speed guarantee. See
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
pixal3d-cuda generate --image object.png --output-dir outputs/native-reference --save-reference
```

Use `8.0`/`8.6` for corresponding Ampere GPUs. Setup clones locked dependency
sources outside this repository, then explicitly compiles native extensions.
It does not launch cloud compute. This source-build recipe is **not yet build-tested**.
The upstream example wheels target another GPU/Python ABI and are deliberately
not reused. Model access and each model's license must be reviewed separately,
including BiRefNet/RMBG-2.0. The wrapper pins local model paths and the NAF source;
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

## Measure before optimizing

```bash
python scripts/benchmark_upstream.py --image object.png
python scripts/benchmark_upstream.py --image object.png --execute --output-dir outputs/native-bench
```

Without `--execute` this prints a plan and never initializes CUDA, downloads
models, submits jobs or contacts Modal. Execution alternates official low-VRAM
and standard SDPA profiles, with no precision/resolution fallback. It stops on
failure/OOM instead of excluding failed cases. No speedup is claimed yet.

Each native generation produces `result.glb`, `report.json`, `upstream.log` and
`upstream-request.json`. Optional `--save-reference` writes **private** native
latents and pre-remesh geometry/voxel tensors to `reference.pt`; these can expose
input-derived data. Reports identify requested/actual geometry, conditioning
sizes, source/model pins, stage timings and PyTorch allocated/reserved peaks.
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
