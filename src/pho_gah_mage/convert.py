from typing import Literal, Optional

import cv2 as cv
import numpy as np

from .types import Image

# Source: https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/POYNTON1/ColorFAQ.html#RTFToC11
# out of date for correct conversions for modern monitor colour space.
WEIGHTS = {
    "R": 0.2989,
    "G": 0.5870,
    "B": 0.1140,
}


def convert_3_channel_array_to_gray(image: Image) -> np.ndarray:
    """
    Convert a 3 channel array into grayscale.
    """
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

    gray_image = WEIGHTS["R"] * R + WEIGHTS["G"] * G + WEIGHTS["B"] * B
    return gray_image


def convert_image(image: Image, type: Literal["mono", "colour", "color", "invert"]) -> Optional[Image]:
    """
    Converts image into either `"mono"` or `"colour"`
    Will return `None` if Image cannot be converted.
    """
    match type:
        case "mono":
            if image.shape[-1] != 3:
                raise ValueError(f"Invalid shape of image: {image.shape} != 3") 
            return cv.cvtColor(image, cv.COLOR_RGB2GRAY)
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
