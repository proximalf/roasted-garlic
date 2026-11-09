from matplotlib.colors import Normalize
from typing import Literal, NamedTuple, Tuple
from matplotlib.pyplot import get_cmap

import cv2 as cv
import numpy as np

from .types import Image


def add(image: Image, value: float | int | Image) -> Image:
    """Typed. Add value to image."""
    return image + value


def subtract(image: Image, value: float | int | Image) -> Image:
    """Typed. Subtract value from image."""
    return image - value


def multiply(image: Image, value: float | int | Image) -> Image:
    """Typed. Multiply image by value."""
    return image * value


def divide(image: Image, value: float | int | Image) -> Image:
    """Typed. Divide image by value."""
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
    original_h, original_w = image.shape[:2]

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


def threshold(image: Image, lower: int, upper: int = 255, type: int = cv.THRESH_BINARY) -> Image:
    """
    Binary threshold.

    Convience wrapper around cv.threshold.
    cv.threshold(image, lower, upper, type=cv.THRESH_BINARY)
    """
    _, thresh = cv.threshold(image, lower, upper, type)
    return thresh


def resize(image: Image, max_size: int, interpolation: int = cv.INTER_LINEAR) -> Image:
    """
    Resize image to set max_size. Scale is rounded, so may return exactly max_size.
    Set interpolation - cv.INTER_LINEAR, cv.INTER_AREA, cv.INTER_NEAREST, cv.INTER_MAX
    """
    h, w = image.shape[:2]

    scale = min(max_size / w, max_size / h)

    if scale >= 1:
        return image

    new_w = round(w * scale)
    new_h = round(h * scale)

    return cv.resize(
        src=image,
        dsize=(new_w, new_h),
        interpolation=interpolation,
    )


def bit_shift(image: Image, direction: Literal["left", "right"], shift: int = 1) -> Image:
    """
    Shift an images values left or right, by magnitude of shift.
    Will overflow.
    """
    match direction:    
        case "left":
            return np.left_shift(image, shift)
        case "right":
            return np.right_shift(image, shift)


def apply_colourmap(image: Image, colourmap_name: str, limits: tuple[float, float], resample: int | None = None) -> Image:
    """
    Apply a matplotlib colourmap to image.
    Returns image as a 8bit rgb
    https://matplotlib.org/stable/users/explain/colors/colormaps.html
    """
    lower_limit, upper_limit = limits
    norm = Normalize(vmin=lower_limit, vmax=upper_limit)
    normalised = norm(image)
    
    cmap = get_cmap(colourmap_name)
    
    if resample:
        cmap = cmap.resampled(resample)
    
    # convert from float to 8 bit
    image = (cmap(normalised)[:, :, :3] * 255).astype(np.uint8)

    return image