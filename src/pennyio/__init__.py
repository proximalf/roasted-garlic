from . import convert, operations
from .io import load_image, save_image
from .types import Image, is_image

try:
    # if Qt isn't installed this will fail
    from .qt import numpy_to_pixmap  # type: ignore
except ImportError or ModuleNotFoundError:
    def numpy_to_pixmap(*args, **kwargs):
        raise RuntimeError("numpy_to_pixmap requires Qt - pip install pennyio[qt]")
