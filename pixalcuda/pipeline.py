"""Fixed 512 geometry cascade with full PBR texture generation."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as functional

from .conditioning import Dino, NAF, build_conditioning, build_multiview_conditioning, image_tensor
from .decoder import SparseDecoder, StructureDecoder
from .export import clean_mesh, export_pbr
from .flow import Flow, sample_flow
from .mesh import decoder_output_to_mesh
from .multiview import PreparedViews, encode_views, prepare_views, read_mapped_views, read_views
from .ops import Limits
from .preprocess import camera_from_fov, estimate_camera, prepare_image
from .runtime import Runtime, atomic_json, release
from .sparse import requantize
from .weights import CHECKPOINTS, DINO_REVISION, MOGE_REVISION, NAF_SHA256, PIXAL_REVISION, ModelStore, sha256_file


def dense_coordinates(resolution, device):
    axis = torch.arange(resolution, device=device)
    spatial = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    return functional.pad(spatial, (1, 0)).int()


def finish_export(output_dir, runtime=None):
    runtime = runtime or Runtime.restore(output_dir, "cpu")
    runtime.device = torch.device("cpu")
    runtime.report["postprocess_device"] = str(runtime.device)
    try:
        with runtime.stage("pbr_export"):
            mesh = runtime.load("mesh")
            if mesh is None:
                decoded = runtime.load("shape_decoded")
                if decoded is None:
                    raise ValueError("GPU shape decoder checkpoint is missing")
                vertices, faces = decoder_output_to_mesh(decoded["attributes"].numpy(), decoded["coordinates"].numpy(), resolution=512)
                vertices, faces, cleanup = clean_mesh(vertices, faces, runtime.report["identity"]["target_faces"], return_report=True)
                mesh = {"vertices": torch.from_numpy(vertices), "faces": torch.from_numpy(faces), "cleanup": cleanup}
                runtime.save("mesh", mesh)
                del decoded
            runtime.report["mesh_cleanup"] = mesh["cleanup"]
            texture = runtime.load("texture_decoded")
            if texture is None:
                raise ValueError("GPU texture decoder checkpoint is missing")
            destination = runtime.directory / "result.glb"
            validation = export_pbr(mesh["vertices"].numpy(), mesh["faces"].numpy(), texture["coordinates"][:, 1:].numpy(),
                                    texture["attributes"].numpy(), destination, runtime.report["identity"]["texture_size"])
            runtime.report.update(status="complete", output=str(destination), validation=validation)
        runtime.write_report()
        return destination
    except BaseException as error:
        runtime.report.update(status="failed", error=f"{type(error).__name__}: {error}")
        runtime.write_report()
        raise


class Pipeline:
    def __init__(self, device="cuda", limits=Limits(), store=None):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Select a GPU runtime in Colab.")
        self.limits = limits
        self.default_store = store is None
        self.store = store or ModelStore()

    @torch.inference_mode()
    def generate(self, image_path, output_dir, seed=42, texture_size=1024, target_faces=100000,
                 camera="auto", fov_degrees=49.13, remove_background=True, resume=False,
                 *, views_dir=None, num_views=None, defer_export=False, images=None, camera_metadata=None):
        if sum((bool(image_path), bool(views_dir), images is not None)) != 1:
            raise ValueError("Supply exactly one of image_path, views_dir or mapped images")
        if images is None and camera_metadata is not None:
            raise ValueError("camera_metadata requires mapped images")
        multiview = bool(views_dir) or images is not None
        if num_views is not None and not multiview:
            raise ValueError("num_views requires views_dir or mapped images")
        if multiview and (camera != "auto" or fov_degrees != 49.13):
            raise ValueError("Mapped cameras supply the FOV; camera/FOV overrides are not allowed")
        manifest = None
        if multiview:
            manifest = read_mapped_views(images, camera_metadata, num_views) if images is not None else read_views(views_dir, num_views)
        if self.default_store:
            self.store = ModelStore(multiview=multiview)
        elif bool(getattr(self.store, "multiview", False)) != multiview:
            raise ValueError("ModelStore conditioning mode differs; dedicated MV flows are required")
        config = self.store.config()
        identity = {"input_sha256": None if multiview else sha256_file(image_path),
                    "conditioning_mode": "multiview_average" if multiview else "single",
                    "views": manifest.identity if multiview else None,
                    "seed": seed, "geometry_resolution": 512,
                    "texture_size": texture_size, "target_faces": target_faces,
                    "camera": "posed" if multiview else camera,
                    "fov_degrees": None if multiview else fov_degrees, "remove_background": remove_background,
                    "dino_precision": "float32",
                    "flow_precision": "float32", "sparse_decoder_precision": "float32",
                    "limits": asdict(self.limits), "pixal_revision": getattr(self.store, "revision", PIXAL_REVISION),
                    "dino_revision": DINO_REVISION, "moge_revision": MOGE_REVISION, "naf_sha256": NAF_SHA256,
                    "runtime_schema": 1, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                    "source_files": {path.name: sha256_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))}}
        if self.store.directory is not None:
            identity["local_checkpoints"] = {
                name: {"bytes": self.store.checkpoint(name).stat().st_size,
                       "mtime_ns": self.store.checkpoint(name).stat().st_mtime_ns}
                for name in CHECKPOINTS
            }
        runtime = Runtime(output_dir, self.device, identity, resume)
        self.runtime = runtime
        directory = runtime.directory
        try:
            with runtime.stage("preprocess_camera"):
                if multiview:
                    image = prepare_views(manifest, remove_background)
                    for index, view in enumerate(image.images):
                        view.save(directory / f"input-{index:02d}.png")
                else:
                    image = prepare_image(image_path, remove_background)
                    image.save(directory / "input.png")
                camera_params = runtime.load("camera")
                if camera_params is None:
                    if multiview:
                        camera_params = image.main_camera
                    else:
                        camera_params = estimate_camera(image, self.device) if camera == "auto" else camera_from_fov(fov_degrees)
                    runtime.save("camera", camera_params)
            with runtime.stage("dino_features"):
                encoded = runtime.load("dino")
                if encoded is None:
                    def encode(model):
                        if multiview:
                            return {str(size): encode_views(model, image, size, self.device) for size in (512, 1024)}
                        return {str(size): model(image_tensor(image, size, self.device)).cpu() for size in (512, 1024)}
                    encoded = runtime.run_model(lambda: Dino(limits=self.limits), self.store.dino(), encode,
                                                dtype=torch.float32,
                                                ignored=("embeddings.mask_token", "norm.weight", "norm.bias", "rope_embeddings.inv_freq"))
                    runtime.save("dino", encoded)
            with runtime.stage("sparse_structure"):
                structure = runtime.load("structure")
                if structure is None:
                    coordinates = dense_coordinates(16, self.device)
                    if multiview:
                        context = build_multiview_conditioning(encoded["512"], image, coordinates, 16, 512, self.device)
                    else:
                        context = build_conditioning(encoded["512"].to(self.device), image_tensor(image, 512, self.device), coordinates, 16, camera_params)
                    noise = self.noise((1, 8, 16, 16, 16), seed)
                    latent = runtime.run_model(lambda: Flow(8, 8, 1024, dense=True, limits=self.limits), self.store.checkpoint("structure"),
                        lambda model: sample_flow(model, noise, context, coordinates[:, 1:], config["sparse_structure_sampler"]["params"]),
                        dtype=torch.float32, ignored=("rope_phases",))
                    del context, noise
                    logits = runtime.run_model(StructureDecoder, self.store.checkpoint("structure_decoder"), lambda model: model(latent.to(torch.float16)))
                    occupancy = functional.max_pool3d((logits > 0).float(), 2, stride=2)[0, 0]
                    structure = functional.pad(occupancy.nonzero(), (1, 0)).int().cpu()
                    self.check_tokens(structure)
                    runtime.save("structure", structure)
                    del latent, logits, occupancy
            with runtime.stage("shape_lr"):
                lr_latent = runtime.load("shape_lr")
                if lr_latent is None:
                    coordinates = structure.to(self.device)
                    context = self.condition(encoded["512"], image, 512, coordinates, camera_params)
                    noise = self.noise((len(coordinates), 32), seed + 1)
                    normalized = runtime.run_model(lambda: Flow(limits=self.limits), self.store.checkpoint("shape_lr"),
                        lambda model: sample_flow(model, noise, context, coordinates[:, 1:], config["shape_slat_sampler"]["params"]), dtype=torch.float32)
                    lr_latent = self.denormalize(normalized, config["shape_slat_normalization"]).cpu()
                    runtime.save("shape_lr", lr_latent)
                    del context, noise, normalized
            with runtime.stage("coordinate_subdivision"):
                hr_coordinates = runtime.load("coordinates")
                if hr_coordinates is None:
                    decoded = runtime.run_model(lambda: SparseDecoder(limits=self.limits), self.store.checkpoint("shape_decoder"),
                        lambda model: model(lr_latent.to(self.device), structure.to(self.device), coordinates_only=True), dtype=torch.float32)
                    hr_coordinates = requantize(decoded, 512).cpu()
                    self.check_tokens(hr_coordinates)
                    runtime.save("coordinates", hr_coordinates)
                    del decoded
            del lr_latent
            with runtime.stage("hr_conditioning"):
                hr_context = runtime.load("hr_conditioning")
                if hr_context is None:
                    hr_context = {name: value.cpu() for name, value in self.condition(encoded["1024"], image, 1024, hr_coordinates.to(self.device), camera_params).items()}
                    runtime.save("hr_conditioning", hr_context)
            texture_context = hr_context
            if multiview:
                with runtime.stage("texture_conditioning"):
                    texture_context = runtime.load("texture_conditioning")
                    if texture_context is None:
                        texture_context = {name: value.cpu() for name, value in self.condition(
                            encoded["1024"], image, 1024, hr_coordinates.to(self.device), camera_params, naf_target=1024).items()}
                        runtime.save("texture_conditioning", texture_context)
            del encoded
            with runtime.stage("shape_hr"):
                shape = runtime.load("shape_hr")
                if shape is None:
                    coordinates = hr_coordinates.to(self.device)
                    context = {name: value.to(self.device) for name, value in hr_context.items()}
                    noise = self.noise((len(coordinates), 32), seed + 2)
                    shape = runtime.run_model(lambda: Flow(limits=self.limits), self.store.checkpoint("shape_hr"),
                        lambda model: sample_flow(model, noise, context, coordinates[:, 1:], config["shape_slat_sampler"]["params"]), dtype=torch.float32).cpu()
                    runtime.save("shape_hr", shape)
                    del context, noise
            with runtime.stage("shape_decode"):
                shape_decoded = runtime.load("shape_decoded")
                if shape_decoded is None:
                    denormalized = self.denormalize(shape.to(self.device), config["shape_slat_normalization"])
                    output, coordinates, guides = runtime.run_model(lambda: SparseDecoder(limits=self.limits), self.store.checkpoint("shape_decoder"),
                        lambda model: model(denormalized, hr_coordinates.to(self.device)), dtype=torch.float32)
                    shape_decoded = {"attributes": output.float().cpu(), "coordinates": coordinates.cpu(), "guides": guides}
                    runtime.save("shape_decoded", shape_decoded)
                    del output, coordinates, denormalized
            with runtime.stage("texture_flow"):
                texture = runtime.load("texture")
                if texture is None:
                    coordinates = hr_coordinates.to(self.device)
                    context = {name: value.to(self.device) for name, value in texture_context.items()}
                    noise = self.noise((len(coordinates), 32), seed + 3)
                    texture = runtime.run_model(lambda: Flow(64, limits=self.limits), self.store.checkpoint("texture"),
                        lambda model: sample_flow(model, noise, context, coordinates[:, 1:], config["tex_slat_sampler"]["params"], concat=shape.to(self.device)), dtype=torch.float32).cpu()
                    runtime.save("texture", texture)
                    del context, noise
            del hr_context, texture_context, shape
            with runtime.stage("texture_decode"):
                texture_data = runtime.load("texture_decoded")
                if texture_data is None:
                    latent = self.denormalize(texture.to(self.device), config["tex_slat_normalization"])
                    output, coordinates, _ = runtime.run_model(lambda: SparseDecoder(texture=True, limits=self.limits), self.store.checkpoint("texture_decoder"),
                        lambda model: model(latent, hr_coordinates.to(self.device), guides=shape_decoded["guides"]), dtype=torch.float32)
                    texture_data = {"attributes": (output.float() * 0.5 + 0.5).cpu(), "coordinates": coordinates.cpu()}
                    runtime.save("texture_decoded", texture_data)
                    del output, coordinates, latent
            release(self.device)
            if defer_export:
                runtime.report["status"] = "awaiting_export"
                runtime.write_report()
                marker = directory / "gpu-complete.json"
                atomic_json(marker, {"status": "awaiting_export", "fingerprint": runtime.fingerprint})
                return marker
            runtime.resume = True
            return finish_export(directory, runtime)
        except BaseException as error:
            runtime.report.update(status="failed", error=f"{type(error).__name__}: {error}")
            runtime.write_report()
            raise

    def condition(self, encoded, image, size, coordinates, camera, naf_target=512):
        if isinstance(image, PreparedViews):
            return self.runtime.run_model(lambda: NAF(self.limits), self.store.naf(),
                lambda model: build_multiview_conditioning(encoded, image, coordinates, 32, size,
                                                           self.device, model, naf_target), dtype=torch.float32)
        return self.runtime.run_model(lambda: NAF(self.limits), self.store.naf(),
            lambda model: build_conditioning(encoded.to(self.device), image_tensor(image, size, self.device), coordinates, 32, camera, naf=model), dtype=torch.float32)

    def noise(self, shape, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return torch.randn(shape, generator=generator, device=self.device, dtype=torch.float32)

    def check_tokens(self, coordinates):
        if not 0 < len(coordinates) <= self.limits.max_tokens:
            raise MemoryError(f"Active tokens={len(coordinates):,}, limit={self.limits.max_tokens:,}. No geometry tokens were pruned and resolution remains 512.")

    @staticmethod
    def denormalize(latent, normalization):
        return latent * latent.new_tensor(normalization["std"]) + latent.new_tensor(normalization["mean"])
