from . import convert, operations
from .format import Channels, ImageFormat, determine_image_format
from .io import load_image, save_image
from .types import Image, is_image

try:
    # if Qt isn't installed this will fail
    from .qt import image_to_pixmap 
except ImportError or ModuleNotFoundError:

    def image_to_pixmap(*args, **kwargs):
        raise RuntimeError("numpy_to_pixmap requires Qt - pip install pennyio[qt]")
