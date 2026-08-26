from typing import Union
import numpy as np

from .selection import SelectionArea

INTEGER_TYPES = [
    np.uint,
    np.uint8,
    np.uint16,
]


def crop_image_from_mask(
    image_to_crop: np.ndarray, mask: np.ndarray, fill_value: Union[int, float] = np.nan
) -> np.ndarray:
    """
    Crop an image based on a boolean mask by applying a bounding box and backfilling the non-masked regions.

    Parameters
    ----------
    image_to_crop np.ndarray
        The image array to be cropped.
    mask: np.ndarray[bool]
        A boolean mask array of the same shape as the image, indicating which pixels to keep.
    fill_value: Union[int, float]
        A value to fill the regions outside the mask. Default is np.nan.

    Returns
    ----------
    np.ndarray
        A cropped image array containing only the masked region, with the rest filled with `fill_value`.

    Notes
    ----------
    The function first applies the fill_value to all pixels outside the mask,
    then extracts the region within the bounding box defined by the mask's truth values.
    """
    if image_to_crop.dtype in INTEGER_TYPES and fill_value is np.nan:
        fill_value = 0 # int can't use np.nan

    # Doing this consumes and edits the image, so best to copy.
    image_to_crop = image_to_crop.copy()
    image_to_crop[~mask] = fill_value

    rows, cols = np.where(mask)
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()

    cropped_image = image_to_crop[min_row : max_row + 1, min_col : max_col + 1]
    return cropped_image


def crop_image_from_selection(image: np.ndarray, selection: SelectionArea) -> np.ndarray:
    """
    Crop an image based on a given SelectionArea.

    Parameters
    ----------
    image: np.ndarray
        The image array to be cropped.
    selection: SelectionArea
        The SelectionArea object defining the region to crop.

    Returns
    ----------
    np.ndarray
        A cropped image array corresponding to the region defined by the selection area.

    Notes
    ----------
    This function generates a mask from the selection area
    and then calls `crop_image_from_mask` to perform the cropping.
    """
    mask = selection.as_numpy_mask(image.shape)
    return crop_image_from_mask(image, mask)
