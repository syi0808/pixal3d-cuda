# Mesh artifact investigation

## Confirmed defects in the first T4 output

The original acceptance GLB is identified by SHA-256 `35710dcdd27f00f3b7fd8e1ad7d10af4d83aea6bd96a0c4b9341a319aed307a8`. Its successful finite-geometry/PBR checks did **not** establish surface quality.

- Of 150,306 triangles, **21,985 are exact coincident duplicates**, identified by their float32 triangle positions regardless of vertex/UV indices. Overlapping faces can z-fight and carry conflicting normals/UVs.
- **38 exported vertex normals have zero length**. Finite values alone are not a sufficient normal-validation contract.
- Reconstructing geometric adjacency by exact position gives **67,758 nonmanifold edges**. This is a diagnostic weld of the exported mesh, not a claim about its UV-split index topology.

The previous CUDA cleanup used `Trimesh(process=True)`, one decimation pass, and `fix_normals`. It omitted the topology repair/component filtering present in the reference cleanup, merged distinct coincident vertex fans, and did not repeat sanitization after simplification and float32 conversion. Averaging normals across invalid fans and accepting duplicate triangles can therefore survive into the GLB. These are postprocessing/export defects; the decoder's sigmoid/edge/split-weight extraction equations match the [official FDG decoder](https://github.com/microsoft/TRELLIS.2/blob/main/trellis2/models/sc_vaes/fdg_vae.py).

A controlled **normal-only** local ablation preserved triangle-expanded positions, UVs, and material while replacing the normals. It improved some dark shading but retained the broken surface. It is not a new neural generation and demonstrates why changing lighting or normals alone is insufficient.

## CUDA runtime changes

- Remove degenerate and exact geometrically duplicate triangles without globally welding vertex identities.
- Filter genuinely detached tiny components before splitting topology, retaining attached fine parts.
- Split nonmanifold edges and bow-tie vertex fans before decimation, orient manifold neighbors consistently, and repeat repair after each decimation pass.
- Limit hole filling to small genuine geometric boundary loops; never cap artificial fan-cut seams. Remaining nonorientable conflict faces are isolated without moving/deleting their triangles.
- Compute angle-weighted shading normals within consistently oriented manifold fans, split at 75-degree creases, and reject zero/non-unit normals in exported GLBs. Preserve normals through UV splits.
- Record before/after topology metrics in the run report. Keep 512 geometry, original weights, all four 12-step schedules, FP32 inference, PBR channels, and resource guards unchanged.

Focused regressions cover coincident triangles, disconnected coincident vertex fans, bow-tie vertices, small attached details, winding conflicts, nonorientable cycles, artificial boundary seams, hole perimeter limits, and opposing normal cancellation. The current development-host suite passes **39 tests with 2 CUDA-only skips**.

The repair also passes on the already-damaged original mesh as a local stress case: coincident triangles, nonmanifold edges, inconsistent manifold edges, and zero shading normals become zero. This cannot recover detail damaged by the original decimation and is **not** the requested fresh T4 acceptance run.

## Fresh T4 revalidation

**Pending:** regenerate the original public shoe from scratch in a fresh output directory on Colab T4, retain acceptance/report/test records, and compare PBR renders under the same camera/lighting. Do not reuse the old source-fingerprinted stage checkpoints or interpret the local ablations as full GPU validation.

```bash
python -u scripts/validate_colab_t4.py --output-dir outputs/colab-t4-shoe-topology-v1
```

The separate MLX repository is read-only reference material and is not modified or imported by the CUDA runtime. `scripts/render_glb_pbr.py` is now included here for independent Blender PBR witnesses.
