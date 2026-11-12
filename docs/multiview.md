# Posed four-view CUDA generation

Implemented in this separate CUDA checkout; the MLX checkout is not modified.
**Real pretrained four-view CUDA/T4 generation has not yet been validated.** CPU
contract tests are not an OOM, quality, or CUDA/MLX parity guarantee.

## Input and execution

Supply one to four posed images and `transforms.json` in one directory. Four
views normally cover the object at 0/90/180/270 degrees. Frame order matters;
view zero defines the output frame. Use the actual camera calibration, not just
the filenames. Paths must remain inside the directory, including symlinks.

```bash
pixal3d-cuda download --multiview
python scripts/verify_checkpoints.py --multiview
pixal3d-cuda generate --views-dir inputs/object --output-dir outputs/object-mv
```

Use `--no-rembg` for already masked RGBA images. Unlike single-image input, views
are **not cropped, recentered, or EXIF-rotated**: their framing must match their
poses. Background removal preserves dimensions. Images are premultiplied onto
black for conditioning. Camera estimation and single-image FOV overrides are
not used in this mode.

`assets/multiview/transforms.example.json` is an **assumed** level orbit, 20-degree
horizontal FOV, distance 3.1192, world +Z up, Blender/NeRF camera-to-world poses
(camera looks along -Z, +Y up). Copy it only for images matching that rig. It does
not calibrate arbitrary photos or generated front/side/back images. Inconsistent
perspective, framing, scale, or object identity can still break reconstruction.

`camera_angle_x` is in radians, optionally overridden per frame. Matrices must
be finite, right-handed rigid 4x4 c2w transforms. `mesh_scale` must be positive.
Changing any image, camera, order, or scale invalidates resume checkpoints.

To compare one versus four inputs **within the same MV model**, use
`--num-views 1` and `--num-views 4` in separate output directories. One posed view
still uses MV weights, not the separate `--image` single-view weights.

## Model and memory semantics

- Dedicated four MV flow checkpoints, pinned to
  `TencentARC/Pixal3D@b0cb2e1b794cab9aa0ac38a95d794a4d9337437f`;
  `pipeline_mv.json` provides the official schedules and normalization.
- Relative poses are `canonical_front @ inverse(main_c2w) @ view_c2w`.
- DINO/NAF features are projected per view, then averaged in FP32. Context
  dimensions do not grow with view count; the 3D flow cascade runs **once**.
- One view is on the GPU at a time during backbone/conditioning extraction.
  DINO features are stored on CPU and identical processed images reuse backbone
  output without reusing camera-dependent projections.
- MV shape conditioning uses NAF-512; texture conditioning separately uses
  NAF-1024. Single-image conditioning retains its previous NAF-512 behavior.
- Geometry stays 512, all four schedules retain 12 steps. DINO, flows, sparse
  decoders and NAF remain FP32 following the actual T4 FP16 failures. No silent
  token pruning, quantization, or lower-resolution fallback.

This avoids four full generations or four concurrently resident models. It
does not imply four views are free or guarantee the same token count/VRAM. Do
not extrapolate the MLX benchmark's memory or timing to T4.

## Colab validation

On 2026-11-12, all **3,450 tensor names/shapes** across the seven pinned MV
flow/decoder checkpoints matched the CUDA meta models, using only remote
safetensors headers (no full weight downloads). Reproduce the metadata check:

```bash
python scripts/verify_checkpoints.py --multiview --headers-only
```

This checks architecture compatibility, not tensor values, DINO/NAF weights,
numerical parity or GPU inference.

Use [the experimental MV notebook](../notebooks/colab_t4_multiview_512.ipynb).
The original single-image notebook remains pinned to its historical release.
The real-GPU acceptance runner supports the same posed directory:

```bash
python -u scripts/validate_colab_t4.py --views-dir inputs/object \
  --num-views 4 --no-rembg --output-dir outputs/t4-mv-four
```

It refuses CPU/non-T4 devices and records every image hash, poses, source
revision, GLB hash, hardware and elapsed time. Review `acceptance.json`,
`run/report.json`, and actual GLB renders before calling the port validated.

## References

The posed-average path follows the read-only MLX fork and
[official MV inference](https://github.com/TencentARC/Pixal3D/blob/f7cf38429b0bd264f1995f0f8743a88b1c728b94/inference_mv.py),
the [projected conditioning implementation](https://github.com/TencentARC/Pixal3D/blob/f7cf38429b0bd264f1995f0f8743a88b1c728b94/pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py),
and the [pinned MV configuration](https://huggingface.co/TencentARC/Pixal3D/blob/b0cb2e1b794cab9aa0ac38a95d794a4d9337437f/pipeline_mv.json).
