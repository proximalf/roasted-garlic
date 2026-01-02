from pennyio.test_data import TestImages
from pennyio.format import determine_image_format, ImageFormat

import numpy as np
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def test_image_formats() -> None:
    """
    Test `determine_image_format` function returns the correct ImageFormat Enum.
    """

    image_format = determine_image_format(TestImages.grey8())
    assert image_format  == ImageFormat.Mono8
    assert image_format.is_mono() == True

    image_format = determine_image_format(TestImages.grey16())
    assert image_format  == ImageFormat.Mono16
    assert image_format.is_mono() == True
    
    image_format = determine_image_format(TestImages.rgb8())
    assert image_format  == ImageFormat.Colour8
    assert image_format.is_colour() == True

    image_format = determine_image_format(TestImages.rgb16())
    assert image_format  == ImageFormat.Colour16
    assert image_format.is_colour() == True

    image_format = determine_image_format(TestImages.float_jb())
    assert image_format  == ImageFormat.ColourFloat
    assert image_format.is_colour() == True
    assert image_format.is_float() == True
    
    image_format = determine_image_format(TestImages.rgba16())
    assert image_format  == ImageFormat.Alpha16
    assert image_format.is_alpha() == True

    image_format = determine_image_format(TestImages.rgbaf16())
    assert image_format  == ImageFormat.AlphaFloat
    assert image_format.is_alpha() == True
    assert image_format.is_float() == True

    image_format = determine_image_format(TestImages.rgbaf32())
    assert image_format  == ImageFormat.AlphaFloat
    assert image_format.is_alpha() == True
    assert image_format.is_float() == True


    

if __name__ == "__main__":
    test_image_formats()