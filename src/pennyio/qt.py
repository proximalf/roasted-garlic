import logging

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

from .convert import convert_float_to_uint
from .format import ImageFormat, determine_image_format
from .types import BIT16, Image

logger = logging.getLogger()


def numpy_to_pixmap(array: np.ndarray) -> QPixmap:
    """
    This function is being renamed, to image_to_pixmap.
    """
    print("This function is being renamed.")
    return image_to_pixmap(array)


def image_to_pixmap(image: Image) -> QPixmap:
    """
    Converts a numpy image array to a QImage type and return as a QPixmap.
    For grayscale, it should be (height, width).
    For RGBA (height, width, channels).

    For 16-bit images, Qt expects an alpha channel, one will be appended if an image is provided with out one.

    QImage doesn't support natively float, the image will be converted to int8 if float16 else int16.
    For float images, if the max value is 1.0, it will be assumed to be a normalised image, and will be
    scaled and converted to an int type.

    There may be unintended effects when using float images.

    Parameters
    ----------
    array: ndarray
        Must be an image array.

    Returns
    ----------
    image: QPixmap
        Image as a QPixmap to display within a qt app.

    Raises
    ----------
    Exception if image is not 2D
    TypeError if image has more than 4 channels.
    """
    if not image.flags.c_contiguous:  # Make array contiguous
        image = np.ascontiguousarray(image)

    height, width = image.shape[:2]

    image_format = determine_image_format(image)

    logger.debug(f"Converting image np array to qimage {image.dtype =} - {image_format =}")

    if image_format in (ImageFormat.MonoFloat, ImageFormat.ColourFloat, ImageFormat.AlphaFloat):
        image = convert_float_to_uint(image, image_format)
        image_format = determine_image_format(image)

    match image_format:
        case ImageFormat.Mono8:
            qimage_format = QImage.Format.Format_Grayscale8

        case ImageFormat.Mono16:
            qimage_format = QImage.Format.Format_Grayscale16

        case ImageFormat.Colour8:
            qimage_format = QImage.Format.Format_RGB888

        case ImageFormat.Alpha8:
            qimage_format = QImage.Format.Format_RGBA8888

        case ImageFormat.Colour16:
            # This is the same as the Format_RGBA64 except alpha must always be 65535.
            qimage_format = QImage.Format.Format_RGBX64
            # Needs an alpha channel, so make it fully opaque
            alpha = np.full((height, width, 1), BIT16, dtype=np.uint16)
            image = image.astype(np.uint16)
            image = np.concatenate((image, alpha), axis=2)

        case _:
            # Default as RGBA64.
            qimage_format = QImage.Format.Format_RGBA64

    bytes_per_line = image.strides[0]
    qimage = QImage(image.data, width, height, bytes_per_line, qimage_format)

    pixmap = QPixmap.fromImage(qimage, Qt.ImageConversionFlag.ColorOnly)
    return pixmap
