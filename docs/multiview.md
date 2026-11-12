# Mapped image/camera CUDA generation

Implemented in this separate CUDA checkout; the MLX checkout is not modified.
**Real pretrained mapped multi-view CUDA/T4 generation has not yet been validated.** CPU
contract tests are not an OOM, quality, or CUDA/MLX parity guarantee.

## Input and execution

Supply an ordered list of **N images and N explicitly mapped cameras**. The
model input is no longer a fixed four-view rig; one, three, five or more mapped
images use the same average-fusion path. The Modal adapter separately defaults
to an operational maximum of 16 views; this is not a model limit or fit promise.
This matches the new read-only MLX camera-mapping contract at revision
`56b4ac8`, without depending on that checkout.

```bash
pixal3d-cuda download --multiview
python scripts/verify_checkpoints.py --multiview
pixal3d-cuda generate --images photo-a.png photo-b.png photo-c.png \
  --transforms cameras.json --output-dir outputs/object-mv
```

Each frame uses `image_index` to identify an entry in the **CLI/Python image
list**, not the position of that frame in the JSON array. Indices must form a
complete one-to-one mapping of `0..N-1`. Either all frames have indices, or all
omit them and use image-list order. Duplicate, missing, out-of-range, boolean
and mixed indexed/unindexed mappings fail before inference. Image zero is the
main view even if its frame appears last in the JSON.

Every image needs one of these explicit camera forms:

- `azimuth_degrees`, `elevation_degrees`, positive `distance`, `fov_degrees`:
  zero-roll orbit around the origin, world +Z up. Elevation is -90 to +90;
  horizontal FOV is strictly between 0 and 180 degrees.
- `transform_matrix`: rigid, right-handed 4x4 Blender/NeRF c2w pose, plus
  `camera_angle_x` in radians (shared or per frame), or per-frame `fov_degrees`.
  Matrices and orbit fields cannot be mixed on the same frame. Do not specify
  both per-frame FOV units.

See `assets/multiview/cameras.mapped.example.json` for three indexed orbit
cameras deliberately listed out of order. **These are example assumptions,
not automatic calibration.** Use actual camera values matching your images.
No missing camera is filled with a default four-view rig.

The Python API accepts the same mapping object directly:

```python
from pixalcuda.pipeline import Pipeline

Pipeline("cuda").generate(
    None, "outputs/object-mv",
    images=["photo-a.png", "photo-b.png", "photo-c.png"],
    camera_metadata={"frames": [
        {"image_index": index, "azimuth_degrees": index * 120,
         "elevation_degrees": 0, "distance": 3.119205, "fov_degrees": 20}
        for index in range(3)
    ]},
)
```

For backward compatibility, `--views-dir inputs/object` still reads
`transforms.json` and each frame's relative `file_path`. It now accepts any
positive count and both camera forms. Its paths must remain inside the directory,
including symlinks. With explicit `--images`, camera JSON filesystem paths are
ignored; only the supplied image list is read. Duplicate source filenames in
different directories are therefore unambiguous.

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
Changing any selected image, camera, order, or scale invalidates resume checkpoints.

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

CPU-only normalization was compared directly against read-only MLX revision
`56b4ac8`: 1, 2, 3, 5, 16 and 17 indexed views and five orbit poses produced
identical metadata/poses. Unit tests also exercise a five-view controlled
pipeline with real PBR export. These checks are not pretrained GPU generation
or model numerical-parity evidence.

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
