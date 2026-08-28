import logging
from enum import Enum
from typing import Callable, List, NamedTuple, Tuple

import numpy as np

from .types import Image

logger = logging.getLogger()


class Channels(NamedTuple):
    """
    Represent a given image as individual RGB channels.
    """

    R: Image
    G: Image
    B: Image

    @staticmethod
    def from_image(image: Image) -> "Channels":
        try:
            R = image[:, :, 0]
            G = image[:, :, 1]
            B = image[:, :, 2]
        except:
            raise ValueError(f"Cannot convert image of shape: {image.shape} into Channels.")

        return Channels(R, G, B)


class ImageFormat(Enum):
    Mono8 = "mono-8"
    Mono16 = "mono-16"
    MonoFloat = "mono-float"
    Colour8 = "colour-8"
    Colour16 = "colour-16"
    ColourFloat = "colour-float"
    Alpha8 = "alpha-8"
    Alpha16 = "alpha-16"
    AlphaFloat = "alpha-float"

    def __repr__(self) -> str:
        return f"ImageFormat.{self.name}"

    def is_mono(self) -> bool:
        return self in (ImageFormat.Mono8, ImageFormat.Mono16, ImageFormat.MonoFloat)

    def is_float(self) -> bool:
        return self in (ImageFormat.MonoFloat, ImageFormat.ColourFloat, ImageFormat.AlphaFloat)

    def is_colour(self) -> bool:
        """If image is alpha it must be colour."""
        return self in (ImageFormat.Colour8, ImageFormat.Colour16, ImageFormat.ColourFloat) or self.is_alpha()

    def is_alpha(self) -> bool:
        return self in (ImageFormat.Alpha8, ImageFormat.Alpha16, ImageFormat.AlphaFloat)

    def get_bins(self) -> int:
        """
        Returns the number of bins and the max_value to use when calculating the histogram of an image.
        image_format is optional.

        Returns
        8bit - 2^8
        16bit - 2^16
        float - 1000
        """
        logger.warning("DEPRECIATING") # This isn't relevent to format, as you can determine bins from bits

        match self:
            case ImageFormat.AlphaFloat | ImageFormat.ColourFloat | ImageFormat.MonoFloat:
                bins = 1000
            case ImageFormat.Mono8 | ImageFormat.Colour8 | ImageFormat.Alpha8:
                bins = 2**8 - 1

            case ImageFormat.Mono16 | ImageFormat.Colour16 | ImageFormat.Alpha16:
                bins = 2**16 - 1

        return bins

    @property
    def bits(self) -> int | None:
        """
        Return number of bit rate in image
        """
        match self:
            case ImageFormat.Mono8 | ImageFormat.Colour8 | ImageFormat.Alpha8:
                return 8

            case ImageFormat.Mono16 | ImageFormat.Colour16 | ImageFormat.Alpha16:
                return 16
            
            case _:
                return None

def determine_image_format(image: Image) -> ImageFormat:
    """
    Determine the format of image. This will raise Exceptions if image is not 2D or has more than 4 channels.
    An image with only 2 dimensions is typically a mono / greyscale image.
    Colour images are 3 dimensional.
    Alpha images have a 4th channel.

    Raises
    ----------
    Exception if image is not 2D
    TypeError if image has more than 4 channels.
    """
    if image.ndim < 2:
        raise Exception(f"Array is not a valid image: {image.shape}")

    mono = image.ndim == 2

    is_8bit = image.dtype == np.uint8
    is_16bit = image.dtype == np.uint16
    is_float = np.issubdtype(image.dtype, np.floating)

    logger.debug(f"Determining image format: {mono =} - {is_float =} - {image.dtype =}")

    if mono:
        if is_8bit:
            return ImageFormat.Mono8
        elif is_16bit:
            return ImageFormat.Mono16
        else:
            return ImageFormat.MonoFloat
    else:
        channels = image.shape[2]
        if channels > 4:
            raise TypeError(f"Unsupported number of channels: {channels}")

        has_alpha = channels > 3

        if has_alpha:
            if is_8bit:
                return ImageFormat.Alpha8
            elif is_16bit:
                return ImageFormat.Alpha16
            else:
                return ImageFormat.AlphaFloat

        # At this point should be a 3 channel colour image.
        if is_8bit:
            return ImageFormat.Colour8

        elif is_16bit:
            return ImageFormat.Colour16
        else:
            # If cannot be determined then it must be a colour float type.
            return ImageFormat.ColourFloat


def is_image(array: np.ndarray) -> bool:
    """
    Check if object is a valid image.
    """
    try:
        # as long as it passes this function the array must be an image.
        _ = determine_image_format(array)
        return True
    except:
        return False
