from typing import Any, Dict, List, Literal, Union, Collection

import cv2 as cv
from numpy import ndarray

BIT8 = 2**8 - 1
BIT16 = 2**16 - 1

Image = ndarray
"""
Type alias for images, cv and numpy.
"""

SUPPORTED_IMAGE_TYPES = (".BMP", ".CR2", ".JPG", ".PNG", ".TIF", ".TIFF")
"""
Supported Image Types. Expects to be checked against Path.suffix, which returns leading . and filetype.
".BMP", ".CR2", ".JPG", ".PNG", ".TIF", ".TIFF"
"""

SAVE_IMAGE_TYPES = Literal["JPG", "PNG", "TIFF", "BMP"]
"""
Supported formats that images can be saved as.
In order of preference.
"JPG", "PNG", "TIFF", "BMP"
"""

RAW_FILES = (".NEF", ".CR2")
"""
Raw files that can be loaded into application.
"""