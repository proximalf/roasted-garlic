import logging
from typing import Literal, Optional

import cv2 as cv
import numpy as np

from .format import ImageFormat, determine_image_format
from .types import BIT8, BIT16, Image

logger = logging.getLogger()

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


def convert_image(image: Image, type: Literal["mono", "colour", "color", "invert"], silent: bool = False) -> Image:
    """
    Converts image into either `"mono"` or `"colour"`
    Will return `None` if Image cannot be converted.
    Set silent to True to ignore and return invalid image.
    """
    match type:
        case "mono":
            if image.shape[-1] != 3:
                if silent:
                    return image  # Return if silent
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


def convert_uint_to_normalised_float(image: Image) -> Image:
    """
    Crudely convert image of dtype uint8 | uint16 to float32.
    """
    if not np.issubdtype(image.dtype, np.integer):
        raise Exception("Can only cast uint to float")

    max_value = np.iinfo(image.dtype).max
    return image.astype(np.float16) / max_value


def convert_float_to_uint(image: Image, image_format: ImageFormat | None = None) -> Image:
    """
    Converts any provided float format image, into its corrasponding uint format.
    Providing an `image_format` is optional, which skips the check.

    Raises
    ----------
    Exception if the image format if not suitable for converting.
    """
    if image_format is None:
        image_format = determine_image_format(image)

    if image_format not in (ImageFormat.MonoFloat, ImageFormat.ColourFloat, ImageFormat.AlphaFloat):
        raise Exception(f"Image format is not suitable for converting into uint - {image_format}")

    # scale if a normal image
    is_normal_image = image.max() <= 1.0

    if is_normal_image:
        # Clip off values below zero, and if it is normalised, it should be below 1 regardless.
        image = np.clip(image, 0.0, 1.0)

        if image.dtype == np.float32:
            image *= BIT16
        else:
            image *= BIT8

        logger.debug(f"Image has been scaled - this is probably why the image looks weird.")

    # Clip and convert to suitable dtype
    if image.dtype == np.float32:
        image = np.clip(image, 0, BIT16)
        image = image.astype(np.uint16)
    else:
        if image.max() > BIT8:
            logger.warning(f"Image has been clipped - if the image is all white, check the dtype of the image")
        image = np.clip(image, 0, BIT8)
        image = image.astype(np.uint8)

    return image
