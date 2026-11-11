"""Explicit foreground preprocessing and optional pinned MoGe-2 camera."""

import math

import numpy as np
from PIL import Image, ImageOps
import torch

from .conditioning import image_tensor
from .weights import MOGE_REPO, MOGE_REVISION


def prepare_image(path, remove_background=True):
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    alpha = np.asarray(image.getchannel("A"))
    if remove_background and (alpha == 255).all():
        try:
            from rembg import new_session, remove
        except ImportError as error:
            raise RuntimeError("Install pixal3d-cuda[preprocess], or supply an RGBA image / --no-rembg") from error
        image = remove(image, session=new_session("u2net", providers=["CPUExecutionProvider"]))
    array = np.asarray(image)
    foreground = np.argwhere(array[..., 3] > 127)
    if not len(foreground):
        raise ValueError("Input has no foreground pixels")
    top, left = foreground.min(0)
    bottom, right = foreground.max(0) + 1
    side = max(2, int(math.ceil(max(bottom - top, right - left) * 1.1)))
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    box = (round(center_x - side / 2), round(center_y - side / 2), round(center_x - side / 2) + side, round(center_y - side / 2) + side)
    crop = image.crop(box)
    canvas = Image.new("RGB", crop.size, "black")
    canvas.paste(crop, mask=crop.getchannel("A"))
    return canvas


def camera_from_fov(fov_degrees):
    if not 1 <= fov_degrees <= 150:
        raise ValueError("Horizontal FOV must be between 1 and 150 degrees")
    radians = math.radians(fov_degrees)
    return {"camera_angle_x": radians, "distance": 0.5 / math.tan(radians / 2), "mesh_scale": 1.0}


@torch.inference_mode()
def estimate_camera(image, device):
    try:
        from moge.model.v2 import MoGeModel
    except ImportError as error:
        raise RuntimeError("Auto camera requires pixal3d-cuda[camera]. Use --camera fixed --fov-degrees 49.13 explicitly to skip it.") from error
    model = MoGeModel.from_pretrained(MOGE_REPO, revision=MOGE_REVISION).to(device).eval()
    try:
        result = model.infer(image_tensor(image, 512, device)[0], use_fp16=False)
        focal = float(result["intrinsics"][0, 0])
        return camera_from_fov(math.degrees(2 * math.atan(0.5 / focal)))
    finally:
        del model
