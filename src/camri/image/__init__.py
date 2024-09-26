from .deprecated import OrthoViewer, singlerow_orthoplot
from .handler import Handler


class ImageLoader(Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def load(path):
    return ImageLoader(path)