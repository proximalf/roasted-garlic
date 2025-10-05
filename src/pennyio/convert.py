from typing import Literal, Optional

import cv2 as cv
import numpy as np

from .types import Image

# Source: https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/POYNTON1/ColorFAQ.html#RTFToC11
# out of date for correct conversions for modern monitor colour space.
# https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
RED_FACTOR = 299 / 1000
GREEN_FACTOR = 587 / 1000
BLUE_FACTOR = 114 / 1000


def convert_array_to_mono(image: Image) -> np.ndarray:
    """
    Convert a 3 channel array into grayscale.
    """
    dtype = image.dtype  # Set to input dtype
    depth = len(image.shape)

    if depth == 2:
        R = image[:, 0]
        G = image[:, 1]
        B = image[:, 2]
    elif depth == 3:
        R = image[:, :, 0]
        G = image[:, :, 1]
        B = image[:, :, 2]
    else:
        raise TypeError(f"Cannot convert, invalid array {image.shape}")

    mono = RED_FACTOR * R + GREEN_FACTOR * G + BLUE_FACTOR * B
    return mono.astype(dtype)


def convert_image(image: Image, type: Literal["mono", "colour", "color", "invert"]) -> Optional[Image]:
    """
    Converts image into either `"mono"` or `"colour"`
    Will return `None` if Image cannot be converted.
    """
    match type:
        case "mono":
            if image.shape[-1] != 3:
                raise ValueError(f"Invalid shape of image: {image.shape} != 3")
            return convert_array_to_mono(image)
        case "colour":
            if len(image.shape) != 2:
                raise ValueError(f"Invalid shape of image: {image.shape} != 2")
            return cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        case "color":  # Americans -.-
            if len(image.shape) != 2:
                raise ValueError(f"Invalid shape of image: {image.shape} != 2")
            return cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        case "invert":
            # Can only invert a mono image
            if len(image.shape) != 2:
                raise ValueError(f"Invalid shape of image: {image.shape} != 2")
            return cv.bitwise_not(image)
        case _:
            raise KeyError(f"Invalid type - {type}")
