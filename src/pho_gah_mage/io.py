from pathlib import Path
from typing import Optional, Union

import cv2 as cv
import numpy as np
import rawpy

from .types import RAW_FILES, SAVE_IMAGE_TYPES, SUPPORTED_IMAGE_TYPES, Image


def load_raw_image(image_file: Path, output_bits: int = 16) -> Image:
    """
    Helper function loading raw image. Use the libraw wrapper Rawpy.
    `output_bits` defaults to 16-bit.
    """
    raw_processor = rawpy.imread(str(image_file))
    
    # Build Params object
    p = rawpy._rawpy.Params()
    p.no_auto_bright=True
    p.use_auto_wb=False
    p.user_flip=0
    p.use_camera_wb=1
    p.output_bps=output_bits
    p.four_color_rgb=True
    # These do not exist as a usable params within the postprocess function,
    # however are used.
    p.exp_correc = 1 
    p.user_qual = 8

    raw_image = raw_processor.postprocess(params=p)

    return raw_image

def convert_colour_image_to_mono(colour_image: Image) -> Image:
    """
    Converts a colour image with 3 channels in the format of RGB, into a greyscale / mono image.
    """
    # This is just a simple factor conversion from 
    # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
    RED_FACTOR = 299/1000
    GREEN_FACTOR = 587/1000
    BLUE_FACTOR = 114/1000
    R = colour_image[:, :, 0]
    G = colour_image[:, :, 1]
    B = colour_image[:, :, 2]
    
    mono = RED_FACTOR * R + GREEN_FACTOR * G + BLUE_FACTOR * B
    
    return mono

def load_image(filepath: Path, convert_to_mono: bool = False, raw_bits: int = 16) -> Image:
    """
    Loads image from filepath. 
    Uses rawpy for RAW files and opencv for everything else.

    Parameters
    ----------
    filepath: Path
        Path of image.
    convert_to_mono: bool = False
        Will return a greyscale / mono image if True, and loaded image is colour.
    raw_bits: int = 16
        Set the bit rate of a loaded raw image.

    Returns
    ---------
    image: Image
        Loaded image as an ndarray.
    """
    if not filepath.exists():
        raise FileExistsError(f"File cannot be found - {filepath}")

    if filepath.suffix.upper() not in SUPPORTED_IMAGE_TYPES:
        raise TypeError(f"File not supported: {filepath}")

    if filepath.suffix in RAW_FILES:
        image = load_raw_image(filepath, raw_bits)
    else:
        image = cv.imread(str(filepath), cv.IMREAD_ANYCOLOR)

        # Convert a colour image as cv reads colours in BGR format which isn't that intuitive.
        if image.shape[-1] == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    if convert_to_mono and image.shape[-1] == 3:
        image = convert_colour_image_to_mono(image)

    return image


def save_image(
    filename: Union[Path, str],
    image: Image,
    cmap: Optional[int] = None,
    filetype: Union[str, SAVE_IMAGE_TYPES] = "TIFF",
) -> None:
    """
    Save image helper function. Filename is appended with filetype.
    Creates parent folder if it doesn't exist.
    'cmap' no longer default.
    Suggested Colourmaps:
        cv.COLORMAP_VIRIDIS

    Parameters
    ----------
    filename: Union[Path, str]
        Filename to save file under.
    image: Image
        An image array.
    cmap: Optional[int]
        Colormap to apply to image, default is cv.COLORMAP_VIRIDIS.
    filetype: Union[str, Literal["BMP", "PNG", "TIFF", "JPG"]]
        Filetype to save image as, this is BMP as default.

    Raises
    ----------
    ValueError
        Incorrect filetype used, default is BMP.
    """
    filetype = filetype.strip(".")
    if filetype.upper() not in ["BMP", "PNG", "TIFF", "JPG"]:
        raise ValueError(f"Error during saving, invalid filetype! {filetype =}")

    if isinstance(filename, str):
        filename = Path(filename)

    if not filename.parent.exists():
        filename.parent.mkdir(parents=True)

    if cmap is not None:
        image = cv.applyColorMap(image.astype(np.uint8), cmap)

    if cmap is not None or cmap:
        image = cv.applyColorMap(image.astype(np.uint8), cmap)

    cv.imwrite(str(filename.with_suffix("." + filetype.lower())), image)
