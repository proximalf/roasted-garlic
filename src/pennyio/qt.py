import logging

import numpy as np
from PySide6.QtCore import Qt 
from PySide6.QtGui import QImage, QPixmap

logger = logging.getLogger()
 

def numpy_to_pixmap(array: np.ndarray) -> QPixmap:
    """
    Converts a numpy array to a QImage type and return as a QPixmap.
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
    """ 
    if not array.flags.c_contiguous: # Make array contiguous
        array = np.ascontiguousarray(array)

    height, width = array.shape[:2]

    logger.debug(f"Converting ndarray to qimage - shape = {array.shape} dtype = {array.dtype}")

    if np.issubdtype(array.dtype, np.floating):
        is_normal_array = array.max() <= 1.0
        
        if is_normal_array:
            array = np.clip(array, 0.0, 1.0)

        if array.dtype == np.float16:
            if is_normal_array:
                array *= 2**8-1
            array= array.astype(np.uint8)
        else:
            if is_normal_array:
                array *= 2**16-1
            array= array.astype(np.uint16)

    # Check if the array is grayscale (2D) or color (3D)
    depth = array.ndim
    is_grayscale = depth == 2
    has_channels = depth > 2
    is_8bit = array.dtype == np.uint8

    channels = 1 if not has_channels else array.shape[2]
    is_rgb = channels == 3
    has_alpha = channels == 4

    if depth < 2:
        raise TypeError(f"Unsupported array shape: {array.shape}")

    if channels > 4:
            raise TypeError(f"Unsupported number of channels: {channels}")

    # Default as RGBA64.
    qimage_format = QImage.Format.Format_RGBA64

    if is_8bit: 
        if is_rgb: 
            qimage_format = QImage.Format.Format_RGB888
        if has_alpha:
            qimage_format = QImage.Format.Format_RGBA8888 
        if is_grayscale:
            qimage_format = QImage.Format.Format_Grayscale8

    elif is_grayscale:  # Grayscale    
        qimage_format = QImage.Format.Format_Grayscale16
        
    elif is_rgb:
        # This is the same as the Format_RGBA64 except alpha must always be 65535.
        qimage_format = QImage.Format.Format_RGBX64
        # Needs an alpha channel, so make it fully opaque
        alpha = np.full((height, width, 1), 65535, dtype=np.uint16)  
        array = np.concatenate((array, alpha), axis=2)

    bytes_per_line = array.strides[0]
    qimage = QImage(array.data, width, height, bytes_per_line, qimage_format) 

    pixmap = QPixmap.fromImage(qimage, Qt.ImageConversionFlag.ColorOnly) 
    return pixmap
