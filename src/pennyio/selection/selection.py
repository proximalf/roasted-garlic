from enum import Enum
from typing import NamedTuple, Sequence

import numpy as np

from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from .point import Point, POINT_LIKE
from .mask import shape_to_numpy_indices, shape_to_numpy_mask, PixelIndices


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
    Call method `as_numpy_indices` to return a mask of rows and columns based on the shape.
    Call method `as_numpy_mask` to return a boolean mask based on the shape.

    All the geometry math is handled by shapely, this is just a simple class for creating the shapes.

    Attributes
    ----------
    coords: list[Point]
        The coords of the shape.
        Circles, point1 being the center, point 2 being the circumference.
    type: ShapeType
        Typing for the shape, to easily define the shape of Area for rendering.
    id: str
        Identity of SelectionArea.
    pad: float
        Set a value to pad / buffer the geometry.
    """

    DEFAULT_ID = "_"

    def __init__(self, coords: Sequence[Point], type: ShapeType, id: str = DEFAULT_ID, pad: float = 0) -> None:
        self.coords = coords
        self.type = type
        self.id = id
        self.pad = pad

    def __repr__(self) -> str:
        return f"SelectionArea - {self.id} <{self.type} c:{self.coords}>"


    def as_numpy_indices(self, mask_shape: tuple[int, int]) -> PixelIndices:
        """
        Return pixel indices of shape with order preserced.
        Requires the dimensions of the mask, typcially the same size as the array to be masked.
        """
        return shape_to_numpy_indices(self.convert_to_shape(), mask_shape)

    def as_numpy_mask(self, mask_shape: tuple[int, int]) -> np.ndarray:
        """
        Return a boolean mask of shape.
        Requires the dimensions of the mask, typcially the same size as the array to be masked.
        """
        return shape_to_numpy_mask(self.convert_to_shape(), mask_shape)

    def convert_to_shape(self) -> BaseGeometry:
        """
        Convert SelectionArea to Shapely Geometry.
        """
        match self.type:
            case ShapeType.Line:
                start, end = self.coords[:2]
                return create_line_geometry(start, end)
            case ShapeType.Circle:
                centre, circumference = self.coords[:2]
                hyp = Point(centre.x - circumference.x, centre.y - circumference.y)
                radius = (hyp.x**2 + hyp.y**2)**0.5
                return create_circle_geometry(centre, radius)

            case ShapeType.Rectangle:
                start, end = self.coords
                return create_rectangle_geometry(start, end)

            case ShapeType.Polygon:
                return create_polygon_geometry(self.coords)

    @staticmethod
    def line(start: POINT_LIKE, end: POINT_LIKE, id: str = DEFAULT_ID, pad=0) -> "SelectionArea":
        
        start = Point(*start)
        end = Point(*end)
        return SelectionArea([start, end], ShapeType.Line, id, pad=pad)

    @staticmethod
    def rectangle(top_left: POINT_LIKE, size: POINT_LIKE, centre: bool = True, id: str = DEFAULT_ID, pad=0) -> "SelectionArea":

        width, height = size

        if centre:
            x_centre, y_centre = top_left
            top_left = Point(x_centre - width / 2, y_centre - height / 2)
            bottom_right = Point(x_centre + width / 2, y_centre + height / 2)
        else:
            x_top_left, y_top_left = top_left
            top_left = Point(x_top_left, y_top_left)
            bottom_right = Point(width, height)
            
        return SelectionArea([top_left, bottom_right], ShapeType.Rectangle, id, pad)

    @staticmethod
    def circle(centre: POINT_LIKE, radius: int, id: str = DEFAULT_ID, pad=0) -> "SelectionArea":

        x, y = centre
        circumference = Point(x + radius, y)
        centre = Point(x, y)
        return SelectionArea([centre, circumference], ShapeType.Circle, id, pad)

    @staticmethod
    def polygon(points: Sequence[POINT_LIKE], id: str = "", pad=0) -> "SelectionArea":

        coords = [Point(*p) for p in points]
        return SelectionArea(coords, ShapeType.Polygon, id, pad)




# Convert to shapely functions
from shapely.geometry import Point as SPoint # Shapely point

def create_line_geometry(start: POINT_LIKE, end: POINT_LIKE, pad=0) -> BaseGeometry:
    """
    Creates a selection area representing a line segment.

    Parameters
    ----------
    start: POINT_LIKE
        A tuple (x, y) representing the starting point of the line.
    end: POINT_LIKE
        A tuple (x, y) representing the ending point of the line.

    Returns
    ----------
    SelectionArea
        A selection area with the shape of a line from start to end.
    """
    shape = LineString([SPoint(start), SPoint(end)])
    if pad > 0:
        shape = shape.buffer(pad)
    return shape


def create_rectangle_geometry(
    top_left: POINT_LIKE, size: POINT_LIKE, centre: bool = True
) -> BaseGeometry:
    """
    Creates a selection area representing a rectangle.

    Parameters
    ----------
    top_left: POINT_LIKE
        A tuple (x, y) representing the top_left point of the rectangle.
    size: POINT_LIKE
        A tuple (width, height) representing the dimensions of the rectangle.
    centre: bool
        A flag to state if the point is in the centre, default is True to not upset existing behaviour.

    Returns:
        SelectionArea: A selection area with the shape of a rectangle centered at `centre` with the given `size`.
    """
    width, height = size
    if centre:
        x_centre, y_centre = top_left
        top_left = SPoint(x_centre - width / 2, y_centre - height / 2)
        top_right = SPoint(x_centre + width / 2, y_centre - height / 2)
        bottom_right = SPoint(x_centre + width / 2, y_centre + height / 2)
        bottom_left = SPoint(x_centre - width / 2, y_centre + height / 2)
    else:
        x_top_left, y_top_left = top_left
        top_left = SPoint(x_top_left, y_top_left)
        top_right = SPoint(width, y_top_left)
        bottom_right = SPoint(width, height)
        bottom_left = SPoint(x_top_left, height)

    shape = Polygon([top_left, top_right, bottom_right, bottom_left])
    return shape


def create_circle_geometry(centre: POINT_LIKE, radius: int) -> BaseGeometry:
    """
    Creates a selection area representing a circle.

    Parameters
    ----------
    centre: POINT_LIKE
        A tuple (x, y) representing the center of the circle.
    radius: int
        The radius of the circle.

    Returns
    ----------
    SelectionArea
        A selection area with the shape of a circle centered at `centre` with the given `radius`.
    """
    shape = SPoint(centre).buffer(radius)
    return shape


def create_polygon_geometry(points: Sequence[POINT_LIKE]) -> BaseGeometry:
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
    points: List[POINT_LIKE]
        A list of tuples, where each tuple (x, y) represents a vertex of the polygon.

    Returns
    ----------
    SelectionArea
        A selection area with the shape of a polygon defined by the given `points`.
    """
    shape = Polygon([SPoint(xy) for xy in points])
    return shape
