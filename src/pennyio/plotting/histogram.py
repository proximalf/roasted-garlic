from typing import Tuple
from enum import Enum
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from pennyio import Image, convert
from pennyio.format import determine_image_format, ImageFormat, Channels
from pennyio.operations import calculate_histogram

from .channels import PlotChannels


def initialise_plot(axes: Axes, image: Image) -> PlotChannels:
    """
    Setup line profile plot on widget
    """
    image_format = determine_image_format(image)
    print(image_format)
    plot_channels = PlotChannels(mono=image_format.is_mono())
    print(plot_channels.mono)

    axes.add_line(plot_channels.M)
    if not plot_channels.mono:
        axes.add_line(plot_channels.R)
        axes.add_line(plot_channels.G)
        axes.add_line(plot_channels.B)

    bins, max = update_plot_channels(plot_channels, image, image_format)

    left, right = 0, bins
    axes.set_xlim(left, right)
    PAD = 1.2
    axes.set_ylim(0, max*PAD)

    return plot_channels


def update_plot_channels(plot_channels: PlotChannels, image: Image, image_format: ImageFormat | None = None) -> Tuple[int, int | float]:
    """
    This function updates the line data on a given PlotChannels object.
    """
    if image_format is None:
        image_format = determine_image_format(image)

    bins = image_format.get_bins()

    if not image_format.is_float():
        max_value = bins
    else:
        if (max_value := image.max()) > 1:
            max_value = max_value
        else:
            max_value = 1.0

    # If mono only update the only channel
    if plot_channels.mono:
        hist, bin_edges = calculate_histogram(image, bins, max_value)
        plot_channels.M.set_xdata(bin_edges)
        plot_channels.M.set_ydata(hist)
        print(bins, hist.max())
        return bins , hist.max()
    
    max = []
    image_channels = Channels.from_image(image)

    # Red
    hist, bin_edges = calculate_histogram(image_channels.R, bins, max_value)
    plot_channels.R.set_xdata(bin_edges)
    plot_channels.R.set_ydata(hist)

    max.append(hist.max())

    # Green
    hist, bin_edges = calculate_histogram(image_channels.G, bins, max_value)
    plot_channels.G.set_xdata(bin_edges)
    plot_channels.G.set_ydata(hist)

    max.append(hist.max())

    # Blue
    hist, bin_edges = calculate_histogram(image_channels.B, bins, max_value)
    plot_channels.B.set_xdata(bin_edges)
    plot_channels.B.set_ydata(hist)

    max.append(hist.max())

    # Mono
    hist, bin_edges = calculate_histogram(convert.convert_array_to_mono(image), bins, max_value)
    plot_channels.M.set_xdata(bin_edges)
    plot_channels.M.set_ydata(hist)

    max.append(hist.max())

    return bins, np.max(max)


class PlotOrientation(Enum):
    Horizontal = 0
    Vertical = 1

class HistogramPlot:
    """
    A histogram image plot.
    """

    PAD = 1.2

    def __init__(self, axes: Axes, image: Image) -> None:        
        self.axes = axes
        self.channels = initialise_plot(axes, image)
        
        self.orientation = PlotOrientation.Vertical
        
    def update_plot_data(self, image: Image) -> None:
        """
        Sets the data to display on both plot items
        """
        bins, max = update_plot_channels(self.channels, image)

        left, right = 0, bins
        self.axes.set_xlim(left, right)
        self.axes.set_ylim(0, max*self.PAD)
