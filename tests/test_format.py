from pennyio.test_data import TestImages
from pennyio.format import determine_image_format, ImageFormat

import numpy as np
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def test_image_formats() -> None:
    """
    Test `determine_image_format` function returns the correct ImageFormat Enum.
    """

    assert determine_image_format(TestImages.grey8()) == ImageFormat.Mono8
    assert determine_image_format(TestImages.grey16()) == ImageFormat.Mono16
    
    assert determine_image_format(TestImages.rgb8()) == ImageFormat.Colour8
    assert determine_image_format(TestImages.rgb16()) == ImageFormat.Colour16
    assert determine_image_format(TestImages.float_jb()) == ImageFormat.ColourFloat
    
    assert determine_image_format(TestImages.rgba16()) == ImageFormat.Alpha16
    assert determine_image_format(TestImages.rgbaf16()) == ImageFormat.AlphaFloat
    assert determine_image_format(TestImages.rgbaf32()) == ImageFormat.AlphaFloat
    

if __name__ == "__main__":
    test_image_formats()