from pathlib import Path

import cv2 as cv
import numpy as np
import rawpy

from .format import determine_image_format
from .convert import convert_array_to_mono
from .types import RAW_FILES, SAVE_IMAGE_TYPES, SUPPORTED_IMAGE_TYPES, Image, SUPPORTED_IMAGE_TYPE


def load_raw_image(image_file: Path, output_bits: int = 16) -> Image:
    """
    Helper function loading raw image. Use the libraw wrapper Rawpy.
    `output_bits` defaults to 16-bit.
    """
    raw_processor = rawpy.imread(str(image_file))

    # Build Params object
    p = rawpy._rawpy.Params()  # type: ignore # This is valid.
    p.no_auto_bright = True
    p.use_auto_wb = False
    p.user_flip = 0
    p.use_camera_wb = 1
    p.output_bps = output_bits
    p.four_color_rgb = True
    # These do not exist as a usable params within the postprocess function,
    # however are used.
    p.exp_correc = 1
    p.user_qual = 8

    raw_image = raw_processor.postprocess(params=p)

    return raw_image


def load_image(path: Path, convert_to_mono: bool = False, raw_bits: int = 16) -> Image:
    """
    Loads image from path.
    Uses rawpy for RAW files and opencv for everything else.
    If convert_to_mono is set, it will use the an internal function.
    Refer to `convert_to_mono` in `pennyio.convert`

    Parameters
    ----------
    path: Path
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
    if not path.exists():
        raise FileNotFoundError(f"File cannot be found - {path}")

    if path.suffix.upper() not in SUPPORTED_IMAGE_TYPES:
        raise TypeError(f"File not supported: {path}")

    if path.suffix in RAW_FILES:
        image = load_raw_image(path, raw_bits)
    else:
        image = cv.imread(str(path), cv.IMREAD_ANYCOLOR)

        if image is None:
            raise Exception(f"File cannot be loaded! {path}")

        # Convert a colour image as cv reads colours in BGR format which isn't that intuitive.
        if image.shape[-1] == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    if convert_to_mono and image.shape[-1] == 3:
        image = convert_array_to_mono(image)

    return image


def save_image(
    path: Path | str,
    image: Image,
    cmap: int | None = None,
    filetype: str | SAVE_IMAGE_TYPES = "PNG",
    default_on_error: bool = True,
) -> None:
    """
    Save image helper function. Path is appended with filetype.
    Creates parent folder if it doesn't exist.
    Suffix of the path takes precedence.

    Suggested Colourmaps:
        cv.COLORMAP_VIRIDIS

    Parameters
    ----------
    path: Path | str
        File name to save file under.
    image: Image
        An image array.
    cmap: int | None
        Colormap to apply to image, default is cv.COLORMAP_VIRIDIS.
    filetype: str | Literal["BMP", "PNG", "TIFF", "JPG"]
        Filetype to save image as, this is TIFF as default.z
    default_on_error: bool
        Setting this to True forces the default filetype PNG to be used.
        No error will be raised.

    Raises
    ----------
    TypeError
        Incorrect filetype used, default is PNG.
    """
    if isinstance(path, str):
        path = Path(path)

    if path.suffix != "":
        filetype = path.suffix

    filetype = filetype.lower().strip(".")

    valid_filetype = filetype == SUPPORTED_IMAGE_TYPE

    if not valid_filetype:

        if not default_on_error:
            raise TypeError(f"Error during saving, invalid filetype! {filetype =}")
        
        filetype = SUPPORTED_IMAGE_TYPE.default


    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    format = determine_image_format(image)

    # Convert image to BGR as cv will write in this format.
    if format.is_colour:
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    if cmap is not None: # TODO: remove to own func
        image = cv.applyColorMap(image.astype(np.uint8), cmap)

    cv.imwrite(str(path.with_suffix("." + filetype)), image)
