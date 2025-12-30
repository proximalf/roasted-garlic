
from enum import Enum
from typing import List, Tuple, Callable, NamedTuple

import logging
import numpy as np

from .types import Image

logger = logging.getLogger()

class ImageFormat(Enum):
    Mono8 = "mono-8"
    Mono16 ="mono-16"
    MonoFloat ="mono-float"
    Colour8 = "colour-8"
    Colour16 ="colour-16"
    ColourFloat ="colour-float"
    Alpha8 = "alpha-8"
    Alpha16 = "alpha-16"
    AlphaFloat = "alpha-float"

    def __repr__(self) -> str:
        return f"ImageFormat.{self.name}"

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