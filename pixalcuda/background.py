"""Job-local, opt-in loading for the upstream background-removal model."""

from contextlib import contextmanager
import importlib


class OptionalBackgroundRemover:
    def __init__(self, factory, model_name=None):
        self.factory = factory
        self.model_name = model_name
        self.model = None
        self.device = "cpu"

    def to(self, device):
        self.device = device
        if self.model is not None:
            self.model.to(device)
        return self

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")

    def __call__(self, image):
        if self.model_name is None:
            raise RuntimeError("Background removal is disabled; supply prepared RGB/RGBA or explicitly enable RMBG-2.0 after license review")
        if self.model is None:
            self.model = self.factory(model_name=self.model_name)
            self.model.to(self.device)
        return self.model(image)


@contextmanager
def deferred_background_removal():
    module = importlib.import_module("pixal3d.pipelines.rembg")
    original = module.BiRefNet
    removers = []
    def create(model_name=None):
        remover = OptionalBackgroundRemover(original, model_name)
        removers.append(remover)
        return remover
    module.BiRefNet = create
    try:
        yield removers
    finally:
        module.BiRefNet = original
