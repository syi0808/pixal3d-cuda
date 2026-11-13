# Optional background removal

RMBG-2.0 is **not a default dependency**. The normal CLI, Python and Modal HTTP
paths use `remove_background=false`. Default download/prefetch excludes its
snapshot, the local pipeline config clears its model path, and a job-local lazy
adapter prevents the upstream eager constructor from loading matting weights.
Moving the pipeline to CUDA/CPU does not construct the background model either.
No upstream checkout, 3D weights, sampler, decoder or PBR exporter is changed.

## Default inputs

- Supply an already framed/prepared image. A transparent RGBA image is recommended.
- Single-image alpha is composited on black **without automatic crop or matting**.
  An opaque RGB image keeps its supplied pixels/framing; its background is not
  magically removed. Do not treat raw-photo results as original auto-matting parity.
- Mapped views retain their supplied masks and calibrated framing, without crop.
- `--no-rembg` remains an explicit spelling of the new default.

There is no silent replacement with another background-removal model. Compare
reference/optimized runs using the same prepared image, framing and camera.

## Explicit opt-in only

The [BRIA model card](https://huggingface.co/briaai/RMBG-2.0) specifies a separate
non-commercial license and commercial-agreement requirement. Review the model's
terms and access conditions before opting in; downloading approval is not itself
commercial permission. This change is not a license clearance for the other
models or the entire service.

For local use after that review:

```bash
pixal3d-cuda download --include-rembg
pixal3d-cuda generate --image object.png --remove-background
```

For Modal, **both** operator and client opt-in are required:

1. Set `allow_rmbg=true` in `deploy/modal.toml` only after license review, then redeploy.
2. Prefetch with `python -m deploy.modal_app --action prefetch --remove-background`.
3. Submit HTTP `options.remove_background=true`, or use the administrative
   `--remove-background` flag. An omitted request option still defaults to false.

With the checked-in `allow_rmbg=false`, explicit requests return 422 before job
admission and prefetch refuses the opt-in before downloads. Clients cannot change
the server policy through HTTP. Health exposes `background_removal_default=false`
and `rmbg_enabled=false`. A ready marker's `rembg` bit reflects what was actually
prefetched, rather than always claiming matting weights exist.

When enabled, the original BiRefNet wrapper is constructed only on its first
actual call and honors upstream device/offload requests. Existing meaningful
RGBA alpha can bypass that inference in upstream preprocessing. Explicit opt-in
prefetch still downloads the model; it is not a claim of zero download for such
requests. Runtime reports distinguish `requested`, `model` and `loaded`.
