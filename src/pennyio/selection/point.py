from __future__ import annotations

from typing import Generic, NamedTuple, TypeAlias, TypeVar


class Point[T: int | float](NamedTuple):
    """
    Two dimensional Point(x, y), with a generic type parameter.
    """

    x: int | float
    y: int | float

    def into_int(self) -> Point[int]:
        if isinstance(self.x, float) and isinstance(self.y, float):
            return Point(int(self.x), int(self.y))

        if isinstance(self.x, int) and isinstance(self.y, int):
            return Point(self.x, self.y)

        raise ValueError(f"Cannot convert point if not a suitable type: {type(self.x) =} {type(self.y) =}")

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