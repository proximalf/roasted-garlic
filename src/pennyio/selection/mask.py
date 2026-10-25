from enum import Enum
from typing import List, NamedTuple, Tuple, Sequence

import numpy as np
from shapely.geometry import LineString, Polygon, Point
from shapely.geometry.base import BaseGeometry


class PixelIndices(NamedTuple):
    """
    Convience wrapper for rows and cols.

    Example: `array[pixel_indices] = 255`
    Sets all pixels with matching index as value 255.

    Attributes
    ----------
    rows, cols
        Index positions.
    """

    rows: np.ndarray
    cols: np.ndarray


def shape_to_numpy_mask(shape: BaseGeometry, mask_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a Shapely shape into a binary 2D numpy mask. The result is unordered.

    Parameters:
    -----------
    shape : Geometry
        A Shapely geometry object (e.g., Polygon) that defines the area to be filled in the mask.

    mask_shape : Tuple[int, int]
        A tuple representing the desired output shape of the mask in the format (height, width).

    Returns:
    --------
    mask
        Mask of bool values.

    Notes:
    ------
    The function uses rasterio's rasterization to create the mask. Ensure
    that the coordinates of the input shape are aligned with the mask's grid, as the
    origin and pixel size are set to 1x1 in this implementation.
    """
    try:
        from rasterio.features import rasterize
    except:
        ImportError("Missing rasterio package.")

    # Create a list of shapes and their corresponding values
    shapes = [(shape, 1)]  # 1 for filled pixels

    if len(mask_shape) < 2:
        raise ValueError("Mask must have at least 2 axes.")

    # Create an empty mask with the specified image shape
    mask = rasterize(
        shapes,
        out_shape=mask_shape[0:2],  # must be a 2D mask i think.
        fill=0,  # Background value
    )

    # as a boolean
    return mask.astype(np.bool_)


def shape_to_numpy_indices(shape: BaseGeometry, mask_shape: Tuple[int, int]) -> PixelIndices:
    """
    Convert a Shapely shape into pixel indices, retaining order based on distance from

    Parameters:
    -----------
    shape : Geometry
        A Shapely geometry object (e.g., Polygon) that defines the area to be filled in the mask.

    mask_shape : Tuple[int, int]
        A tuple representing the desired output shape of the mask in the format (height, width).

    Returns:
    --------
    PixelIndices
        Index values for the mask, this is required to preserve ordering.

    Notes:
    ------
    The function uses rasterio's rasterization to create the mask. Ensure
    that the coordinates of the input shape are aligned with the mask's grid, as the
    origin and pixel size are set to 1x1 in this implementation.
    """
    mask = shape_to_numpy_mask(shape, mask_shape)

    # Convert to points to preserve order, rasterio doesn't do this
    rows, cols = np.where(mask)
    pixel_points = [
        # pixel coordinates x=column y=row
        Point(col, row)
        for row, col in zip(rows, cols)
    ]

    # https://shapely.readthedocs.io/en/stable/manual.html#linear-referencing-methods
    distances = np.array([shape.project(point) for point in pixel_points])
    # sort by distance
    order = np.argsort(distances)

    # reorder
    rows = rows[order]
    cols = cols[order]

    return PixelIndices(rows, cols)