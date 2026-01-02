from typing import Iterable

from matplotlib.lines import Line2D

FIXED_HEIGHT = 300


class PlotChannels:
    """
    Grouping common image channels together to aid with plotting.

    Use `CHANNEL.set_data(xydata)` to set line data.
    """

    def __init__(self, mono: bool = False) -> None:
        self._mono: bool = mono  # Readonly.

        self.M = Line2D([], [], label="Mono", color="black")

        if self.mono:
            return  # Do not initialise other colours.

        self.R = Line2D([], [], label="Red", color="red")
        self.G = Line2D([], [], label="Green", color="green")
        self.B = Line2D([], [], label="Blue", color="blue")

    @property
    def mono(self) -> bool:
        return self._mono

    def __iter__(self) -> Iterable[Line2D]:
        yield self.M

        if not self.mono:
            yield self.R
            yield self.G
            yield self.B
