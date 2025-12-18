import logging

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

logger = logging.getLogger()


def numpy_to_pixmap(array: np.ndarray) -> QPixmap:
    """
    Converts a numpy array to a QPixmap.
    The array should be in the format (height, width, channels).
    For grayscale, it should be (height, width).

    QImage unfortunately only accepts int types, values between 0-255 as RGB, this
    function will check the arrays type if float or >int8 it will normalise to these bounds.
    However if you encounter a scrambled image, check it's values.

    Parameters
    ----------
    array: ndarray
        Must be an image array.

    Returns
    ----------
    image: QPixmap
        Image as a QPixmap to display within a qt app.
    """
    # Is this really needed.
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)

    dtype = array.dtype
    # Normalise the array to within the bounds of QImage
    if dtype is np.dtype("float") or dtype is not np.dtype("uint8"):
        normalised_bit_value: int = 8  # as max channel is 255
        array *= (2**normalised_bit_value - 1) / array.max()
        array = array.astype(np.uint8)

    if dtype is not np.dtype("uint8"):
        logger.debug(f"Array did not need normalising - dtype: {dtype}")

    # Check if the array is grayscale (2D) or color (3D)
    depth = len(array.shape)

    logger.debug(f"Converting nparray to qimage - shape = {array.shape} dtype = {dtype}")

    if depth == 2:  # Grayscale
        height, width = array.shape
        image = QImage(array.data, width, height, width, QImage.Format.Format_Grayscale8)
    elif depth > 2:
        height, width, channels = array.shape

        if channels == 3:  # RGB
            image = QImage(array.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        elif channels == 4:  # RGBA
            image = QImage(array.data, width, height, 4 * width, QImage.Format.Format_RGBA8888)
        else:
            raise TypeError(f"Unsupported number of channels: {channels}")

    else:
        raise TypeError(f"Unsupported array shape: {array.shape}")

    pixmap = QPixmap.fromImage(image, Qt.ImageConversionFlag.ColorOnly)
    return pixmap
