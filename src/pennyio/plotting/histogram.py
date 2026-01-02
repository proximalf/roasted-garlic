from enum import Enum
from typing import NamedTuple, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from pennyio import Image, convert
from pennyio.format import Channels, ImageFormat, determine_image_format

from .channels import PlotChannels


class Histogram(NamedTuple):
    """
    Histogram data and bin-edge arrays.
    """

    data: np.ndarray
    bin_edges: np.ndarray


def calculate_histogram(image: Image, bins: int = 2**8, max_value: int | float = 2**8) -> Histogram:
    """
    A wrapper around numpy histogram, with some defaults.

    Parameters
    ----------
    image: Image
    bins: int
        The number of equal width bins to use.
    max_value: int | float
        The maximum value of the range for the histogram.

    Returns
    ----------
    Histogram
        data: ...
        bin_edges: ...

    """
    hist, bin_edges = np.histogram(
        image.ravel(),  # faster than flatten
        bins=bins,  # equal width of number bins
        range=(0, max_value),  # range is set to 0 because we expect images.
    )
    return Histogram(data=hist, bin_edges=bin_edges[:-1])  # Realign bin_edges.


def plot_image_histogram(axes: Axes, image: Image, plot_mono: bool = True, flip: bool = False) -> PlotChannels:
    """
    Plot histogram to a given axes.
    Image format is determined, if passed an RGB image, it will plot 4 lines R,G,B,Mono,
    set `plot_mono` to False to only plot the colours.

    Returns
    ----------
    PlotChannels
        The mpl line objects are stored in this class.
    """
    image_format = determine_image_format(image)
    plot_channels = PlotChannels(mono=image_format.is_mono())

    axes.add_line(plot_channels.M)
    if not plot_channels.mono:
        axes.add_line(plot_channels.R)
        axes.add_line(plot_channels.G)
        axes.add_line(plot_channels.B)

    bins, max = update_histogram_plot_channels(plot_channels, image, image_format, flip=flip)

    left, right = 0, bins
    axes.set_xlim(left, right)
    PAD = 1.2
    axes.set_ylim(0, max * PAD)

    return plot_channels


def update_plot_channel(
    channel: Line2D, image_channel: Image, bins: int, max_value: int | float, flip: bool = False
) -> int | float:
    """
    Conveinience function for updating a PlotChannel line. This function calculates the histogram of an image.
    """
    hist = calculate_histogram(image_channel, bins, max_value)
    if flip:
        channel.set_xdata(hist.data)
        channel.set_ydata(hist.bin_edges)
    else:
        channel.set_xdata(hist.bin_edges)
        channel.set_ydata(hist.data)

    # Return the largest value to size the plot.
    return hist.data.max()


def update_histogram_plot_channels(
    plot_channels: PlotChannels,
    image: Image,
    image_format: ImageFormat | None = None,
    plot_mono: bool = True,
    flip: bool = False,
) -> Tuple[int | float, int | float]:
    """
    This function will calulate the images histogram and sets the line data for PlotChannels.

    Set `plot_mono` to False to disable default mono plot.

    Returns
    ----------
    bins: int
        The
    y_max: int | float
        The largest value
    """
    if image_format is None:
        image_format = determine_image_format(image)

    bins = image_format.get_bins()

    if not image_format.is_float():
        x_max = bins
    else:
        if (x_max := image.max()) > 1:
            x_max = x_max
        else:
            x_max = 1.0

    # If mono only update the only channel
    if plot_channels.mono:
        y_max: int | float = update_plot_channel(plot_channels.M, image, bins, x_max, flip=flip)
        return bins, y_max

    # Plot colour channels
    y_max = []
    image_channels = Channels.from_image(image)

    # Red
    R_max = update_plot_channel(plot_channels.R, image_channels.R, bins, x_max, flip=flip)
    y_max.append(R_max)

    # Green
    G_max = update_plot_channel(plot_channels.G, image_channels.G, bins, x_max, flip=flip)
    y_max.append(G_max)

    # Blue
    B_max = update_plot_channel(plot_channels.B, image_channels.B, bins, x_max, flip=flip)
    y_max.append(B_max)

    # Mono
    if plot_mono:
        M_max = update_plot_channel(plot_channels.M, convert.convert_array_to_mono(image), bins, x_max, flip=flip)
        y_max.append(M_max)

    if flip:
        return np.max(y_max), bins

    return bins, np.max(y_max)
