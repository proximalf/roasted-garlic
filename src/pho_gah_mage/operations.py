from typing import Literal, Tuple, Union

import numpy as np

from .types import Image


def add(image: Image, value: Union[float, int, Image]) -> Image:
    """Add value to image."""
    return image + value


def subtract(image: Image, value: Union[float, int, Image]) -> Image:
    """Subtract value from image."""
    return image - value


def multiply(image: Image, value: Union[float, int, Image]) -> Image:
    """Multiply image by value."""
    return image * value


def divide(image: Image, value: Union[float, int, Image]) -> Image:
    """Divide image by value."""
    return image // value


def crop(image: Image, rect: Tuple[int, int, int, int], pad: int = 0) -> Image:
    """
    Simple crop image from rect, returning the cropped image and the bounding rectangle adjusted by the padding.

    Parameters
    ----------
    image: Image
        Image to crop.
    rect: Tuple[int, int, int, int]
        Region to chop.
    pad: int
        Padding around cropped region. `realigned_rect = (0+pad, 0+pad, w, h)`.

    Returns
    ----------
    Image
        Cropped to size, with any padding applied.
    """
    x, y, w, h = rect
    original_h, original_w = image.shape

    if x + w + pad > original_w or y + h + pad > original_h:
        raise ValueError("Crop error - Pad too big")

    cropped_image = image[(y - pad) : (y + h + pad), (x - pad) : (x + w + pad)]

    return cropped_image


def flip_image(image: Image, direction: Literal["UpDown", "LeftRight"]) -> Image:
    """
    Flips Image in a given direction is either "UpDown" or "LeftRight".
    """
    axis = 0 if direction == "UpDown" else 1
    return np.flip(image, axis)
