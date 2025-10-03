from typing import Any, Dict, List, Literal, Union

import cv2 as cv
from numpy import ndarray


Image = Union[ndarray, cv.typing.MatLike]
"""
Type alias for images, cv and numpy.
"""


SUPPORTED_IMAGE_TYPES: List[str] = [".BMP", ".CR2", ".JPG", ".PNG", ".TIF", ".TIFF"]
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

RAW_FILES = [".NEF", ".CR2"]
"""
Raw files that can be loaded into application.
"""

IMAGE_FILE_FILTER: Dict[str, str] = {
    "Bitmap (*.bmp)": ".bmp",
    "JPEG (*.jpg)": ".jpg",
    "PNG (*.png)": ".png",
    "Tagged Image File Format (*.tiff)": ".tiff",
}
"""
A filter to be used when saving images, could also cover loading.
"""

def is_image(object: Any) -> bool:
    """
    Check if object is a valid image.
    """
    if isinstance(object, ndarray):
        return True
    return False
