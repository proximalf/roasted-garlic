import logging
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple

import numpy as np

from ..convert import convert_image, convert_uint_to_normalised_float
from ..io import load_image
from ..types import Image

logger = logging.getLogger()

DEBUG_IMAGE = Path(__file__).parent / "image.png"
# RAW_IMAGE = Path(__file__).parent / "raw-image.CR2"

BIT_8 = 2**8 - 1
BIT_16 = 2**16 - 1


def draw_square(image: Image, value: Tuple[float | int, ...] | float | int, size_ratio: float = 0.5) -> Image:
    """
    Draw a square on an image with a given value.
    """
    if image.ndim < 2:
        raise Exception("Invalid size array.")

    if isinstance(value, tuple | list):
        if image.ndim == 3:
            if len(value) != image.shape[2]:
                raise Exception(f"Value must be an array equal to number of channels in image.")

    h, w = image.shape[:2]
    size = int(min(h, w) * size_ratio)

    x0, y0 = (w - size) // 2, (h - size) // 2
    x1, y1 = x0 + size, y0 + size

    image[y0:y1, x0:x1] = value
    return image


def generate_image(
    dtype: np.typing.DTypeLike,
    value: float | int | Tuple[float | int, ...],
    shape: Tuple[int, ...] = (256, 256),
    alpha: Optional[float | int] = None,
) -> np.ndarray:
    """
    Generate an image of a given dtype, with a square drawn in the centre of the image of a given value.
    """

    image = np.zeros(shape=shape, dtype=dtype)

    if alpha and image.ndim == 3 and image.shape[-1] == 4:
        image[..., 3] = alpha

    return draw_square(image, value=value)


class TestImages:
    """
    Collection of test images, as methods so they are only called when required.
    - Grey
    - Colour
    - Alpha
    """

    # Grey Images
    @staticmethod
    def grey8() -> Image:
        """mono 8-bit"""
        logger.debug("Using image - grey8")
        return generate_image(dtype=np.uint8, value=200)

    @staticmethod
    def grey16() -> Image:
        """mono 16-bit"""
        logger.debug("Using image - grey16")
        return generate_image(dtype=np.uint16, value=50000)

    # Colour Images
    @staticmethod
    def rgb8() -> Image:
        """RGB 8-bit"""
        logger.debug("Using image - rgb8")
        return generate_image(dtype=np.uint8, value=(200, 0, 0), shape=(256, 256, 3))

    @staticmethod
    def rgb16() -> Image:
        """RGB 16-bit"""
        logger.debug("Using image - rgb16")
        return generate_image(dtype=np.uint16, value=(32000, 32000, 0), shape=(256, 256, 3))

    @staticmethod
    def rgbf16() -> Image:
        """RGB float"""
        logger.debug("Using image - rgb16")
        return generate_image(dtype=np.float16, value=(255, 155, 100), shape=(256, 256, 3))

    # Alpha Images
    @staticmethod
    def rgba16() -> Image:
        """RGBA 16-bit"""
        logger.debug("Using image - rgba16")
        return generate_image(dtype=np.uint16, value=(0, 65535, 0, 65535), shape=(256, 256, 4), alpha=BIT_16)

    @staticmethod
    def rgbaf16() -> Image:
        """RGBA float"""
        logger.debug("Using image - frgba16")
        return generate_image(dtype=np.float16, value=(0.3, 1.0, 0.3, 1.0), shape=(256, 256, 4), alpha=1.0)

    @staticmethod
    def rgbaf32() -> Image:
        """RGBA float32"""
        logger.debug("Using image - frgba16")
        return generate_image(dtype=np.float32, value=(0.5, 1.0, 0.1, 1.0), shape=(256, 256, 4), alpha=1.0)

    @staticmethod
    def colour() -> Image:
        """JB colour image"""
        logger.debug("Using image - colour")
        DEBUG_IMAGE = Path(__file__).parent / "image.png"
        return load_image(DEBUG_IMAGE)

    @staticmethod
    def float_jb() -> Image:
        """JB float image"""
        return convert_uint_to_normalised_float(TestImages.colour())

    @staticmethod
    def mono_jb() -> Image:
        """JB mono image"""
        return convert_image(TestImages.colour(), "mono")

    @staticmethod
    def raw_cr2() -> Image:
        """Its fucking raw"""
        return load_image(RAW_IMAGE)
