from typing import Any, Collection, Dict, List, Literal, Union
from pathlib import Path
import cv2 as cv
from numpy import ndarray

BIT8 = 2**8 - 1
BIT16 = 2**16 - 1

Image = ndarray
"""
Type alias for images, cv and numpy.
"""

SUPPORTED_IMAGE_TYPES = (".BMP", ".CR2", ".JPG", ".PNG", ".TIF", ".TIFF")

class SupportedImageType:
    types = ("BMP", "CR2", "JPG", "PNG", "TIF", "TIFF")

    def check_equality(self, other: str | Path) -> bool:
        if isinstance(other, str):
            suffix = other.upper().lstrip(".")
        elif isinstance(other, Path):
            suffix = other.suffix.upper().lstrip(".")
            
        return suffix in self.types

    def __eq__(self, other) -> bool:
        return self.check_equality(other)

    def ___contains__(self, other: str | Path) -> bool:
        return self.check_equality(other)

SUPPORTED_IMAGE_TYPE = SupportedImageType()
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
