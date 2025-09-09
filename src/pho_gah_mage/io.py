from pathlib import Path
from typing import Optional, Union

import cv2 as cv
import numpy as np
import rawpy

from .types import RAW_FILES, SAVE_IMAGE_TYPES, SUPPORTED_IMAGE_TYPES, Image


def load_raw_image(filepath: Path, convert_to_mono: bool = False, output_bits: int = 8) -> Image:
    """
    Helper function loading raw image. Uses rawpy.

    """
    img = rawpy.imread(str(filepath))
    raw_image = img.postprocess(no_auto_bright=True, use_auto_wb=False, output_bps=output_bits)
    if not convert_to_mono:
        return raw_image

    R = raw_image[:, :, 0]
    G = raw_image[:, :, 1]
    B = raw_image[:, :, 2]

    # This is just a simple conversion from https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
    grey_image = 0.2989 * R + 0.5870 * G + 0.1140 * B

    return grey_image


def load_image(filepath: Path, convert_to_mono: bool = False, bits: int = 8) -> Image:
    """
    Loads image for processing. Uses rawpy and opencv. Converts colour images to monochromatic.

    Parameters
    ----------
    filepath: Path
        Path of image.

    Returns
    ---------
    image: Image
        Loaded monochromatic image as an ndarray.
    """
    if not filepath.exists():
        raise FileExistsError(f"File cannot be found - {filepath}")

    if filepath.suffix.upper() not in SUPPORTED_IMAGE_TYPES:
        raise TypeError(f"File not supported: {filepath}")

    if filepath.suffix in RAW_FILES:
        image = load_raw_image(filepath, convert_to_mono, bits)
    else:
        image = cv.imread(str(filepath), cv.IMREAD_ANYCOLOR)

        # Convert a colour image as cv reads colours in BGR format which isn't that common.
        if image.shape[-1] == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        if convert_to_mono:
            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

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
