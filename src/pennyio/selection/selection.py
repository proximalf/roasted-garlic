import numpy as np

from enum import Enum
from typing import Tuple, NamedTuple, List

from rasterio.features import rasterize
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point, Polygon, LineString

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
    # Create a list of shapes and their corresponding values
    shapes = [(shape, 1)]  # 1 for filled pixels

    # Create an empty mask with the specified image shape
    mask = rasterize(
        shapes,
        out_shape=mask_shape[0:2],  # Ensure its just a 2D mask, not sure I should do this.
        fill=0,  # Background value
    )

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
    distances = np.array([
        shape.project(point) 
        for point in pixel_points
    ])
    # sort by distance
    order = np.argsort(distances)

    # reorder
    rows = rows[order]
    cols = cols[order]

    return PixelIndices(rows, cols)


class ShapeType(Enum):
    """
    Shape type to define a shape within a SelectionArea.
    """

    Line = "Line"
    Rectangle = "Rectangle"
    Circle = "Circle"
    Polygon = "Polygon"

    def __str__(self):
        return f"ShapeType::{self.name}"


class SelectionArea:
    """
    Wrapper class, all the geometry is handled by shapely.
    Call method `as_numpy_indices` to return a mask of rows and columns based on the shape.
    Call method `as_numpy_mask` to return a boolean mask based on the shape.

    Attributes
    ----------
    shape: BaseGeometry
        Any given shape that inherits BaseGeometry.
    type: ShapeType
        Typing for the shape, to easily define the shape of Area for rendering.
    """

    def __init__(self, shape: BaseGeometry, type: ShapeType, id: str = "_") -> None:
        self.shape = shape
        self.type = type
        self.id = id

    def __repr__(self) -> str:
        return f"SelectionArea - {self.id} <{self.type} c:{self.shape.centroid} bb:{self.shape.boundary.bounds}>"

    def as_numpy_indices(self, mask_shape: Tuple[int, int]) -> PixelIndices:
        """
        Return pixel indices of shape with order preserced.
        Requires the dimensions of the mask, typcially the same size as the array to be masked.
        """
        return shape_to_numpy_indices(self.shape, mask_shape)

    def as_numpy_mask(self, mask_shape: Tuple[int, int]) -> np.ndarray:
        """
        Return a boolean mask of shape.
        Requires the dimensions of the mask, typcially the same size as the array to be masked.
        """
        return shape_to_numpy_mask(self.shape, mask_shape)

    @staticmethod
    def line(start: Tuple[int, int], end: Tuple[int, int], pad = 0) -> "SelectionArea":
        return create_line_selection_area(start, end, pad)

    @staticmethod
    def rectangle(top_left: Tuple[int, int], size: Tuple[int, int], centre: bool = True) -> "SelectionArea":
        return create_rectangle_selection_area(top_left, size, centre)

    @staticmethod
    def circle(centre: Tuple[int, int], radius: int) -> "SelectionArea":
        return create_circle_selection_area(centre, radius)
        
    @staticmethod
    def polygon(points: List[Tuple[int, int]]) -> "SelectionArea":
        return create_polygon_selection_area(points)

def create_line_selection_area(start: Tuple[int, int], end: Tuple[int, int], pad = 0) -> SelectionArea:
    """
    Creates a selection area representing a line segment.

    Parameters
    ----------
    start: Tuple[int, int]
        A tuple (x, y) representing the starting point of the line.
    end: Tuple[int, int]
        A tuple (x, y) representing the ending point of the line.

    Returns
    ----------
    SelectionArea
        A selection area with the shape of a line from start to end.
    """
    shape = LineString([Point(start), Point(end)])
    if pad > 0:
        shape = shape.buffer(pad)
    type = ShapeType.Line
    return SelectionArea(shape, type)


def create_rectangle_selection_area(
    top_left: Tuple[int, int], size: Tuple[int, int], centre: bool = True
) -> SelectionArea:
    """
    Creates a selection area representing a rectangle.

    Parameters
    ----------
    top_left: Tuple[int, int]
        A tuple (x, y) representing the top_left point of the rectangle.
    size: Tuple[int, int]
        A tuple (width, height) representing the dimensions of the rectangle.
    centre: bool
        A flag to state if the point is in the centre, default is True to not upset existing behaviour.

    Returns:
        SelectionArea: A selection area with the shape of a rectangle centered at `centre` with the given `size`.
    """
    width, height = size
    if centre:
        x_centre, y_centre = top_left
        top_left = Point(x_centre - width / 2, y_centre - height / 2)
        top_right = Point(x_centre + width / 2, y_centre - height / 2)
        bottom_right = Point(x_centre + width / 2, y_centre + height / 2)
        bottom_left = Point(x_centre - width / 2, y_centre + height / 2)
    else:
        x_top_left, y_top_left = top_left
        top_left = Point(x_top_left, y_top_left)
        top_right = Point(width, y_top_left)
        bottom_right = Point(width, height)
        bottom_left = Point(x_top_left, height)

    shape = Polygon([top_left, top_right, bottom_right, bottom_left])
    type = ShapeType.Rectangle
    return SelectionArea(shape, type)


def create_circle_selection_area(centre: Tuple[int, int], radius: int) -> SelectionArea:
    """
    Creates a selection area representing a circle.

    Parameters
    ----------
    centre: Tuple[int, int]
        A tuple (x, y) representing the center of the circle.
    radius: int
        The radius of the circle.

    Returns
    ----------
    SelectionArea
        A selection area with the shape of a circle centered at `centre` with the given `radius`.
    """
    shape = Point(centre).buffer(radius)
    type = ShapeType.Circle
    return SelectionArea(shape, type)


def create_polygon_selection_area(points: List[Tuple[int, int]]) -> SelectionArea:
    """
    Creates a selection area representing a polygon.

    Notes
    -----
    The shape is defined in a clockwise motion.
    eg as a square:
        1--2
        |  | top_left, top_right, bottom_right, bottom_left
        4--3

    Parameters
    ----------
    points: List[Tuple[int, int]]
        A list of tuples, where each tuple (x, y) represents a vertex of the polygon.

    Returns
    ----------
    SelectionArea
        A selection area with the shape of a polygon defined by the given `points`.
    """
    shape = Polygon([Point(xy) for xy in points])
    type = ShapeType.Polygon
    return SelectionArea(shape, type)
