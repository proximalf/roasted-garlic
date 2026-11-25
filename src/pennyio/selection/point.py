from __future__ import annotations

from collections import namedtuple

import numpy as np

# Base point that is used to construct the true point class.
# overriding the __new__ method to check types.
_BasePoint = namedtuple("_BasePoint", ["x", "y"])


class Point[T: int | float](_BasePoint):
    """
    Two dimensional Point(x, y), with a generic type parameter.
    """

    x: int | float
    y: int | float

    def __new__(cls, x: T | np.number, y: T | np.number) -> Point:
        """Validate itself."""

        if not np.isscalar(x):
            raise TypeError(f"Invalid x type: {x}")

        if not np.isscalar(y):
            raise TypeError(f"Invalid y type: {y}")

        return super().__new__(cls, x, y)

    def into_int(self) -> Point[int]:
        return Point(int(self.x), int(self.y))

    def invert(self) -> Point[T]:
        """Invert point axis"""
        return Point(self.y, self.x)

    def pythagorean(self) -> float:
        """Hypotenuse length"""
        return (self.x**2 + self.y**2) ** 0.5

    def is_zero(self) -> bool:
        return self.x == 0 and self.y == 0  # or (self.x == 0.0 and self.y == 0.0)

    def __add__(self, other: int | float | Point) -> Point:  # ty:ignore[invalid-method-override]
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        return Point(self.x + other, self.y + other)

    def __sub__(self, other: int | float | Point) -> Point:
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        return Point(self.x - other, self.y - other)

    def __mul__(self, other: int | float | Point) -> Point:  # ty:ignore[invalid-method-override]
        if isinstance(other, Point):
            return Point(self.x * other.x, self.y * other.y)
        return Point(self.x * other, self.y * other)

    def __truediv__(self, other: int | float | Point) -> Point:
        if isinstance(other, Point):
            return Point(self.x / other.x, self.y / other.y)
        return Point(self.x / other, self.y / other)

    def __divmod__(self, other: int | float | Point) -> Point:
        if isinstance(other, Point):
            return Point(self.x // other.x, self.y // other.y)
        return Point(self.x // other, self.y // other)

    def __mod__(self, other: int | float | Point) -> Point:
        if isinstance(other, Point):
            return Point(self.x // other.x, self.y // other.y)
        return Point(self.x // other, self.y // other)

    def __floordiv__(self, other: int | float | Point) -> Point:
        if isinstance(other, Point):
            return Point(self.x // other.x, self.y // other.y)
        return Point(self.x // other, self.y // other)


POINT_LIKE = Point | tuple[int | float, int | float]
