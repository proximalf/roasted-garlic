from matplotlib.figure import Figure

from pennyio.plotting.histogram import (
    plot_image_histogram,
    update_histogram_plot_channels,
)
from pennyio.test_data import TestImages


def main():
    image = TestImages.colour()
    figure = Figure()
    axes = figure.add_subplot()

    plot = plot_image_histogram(axes, image)
    figure.savefig("./test-output/test.png")

    bins, max = update_histogram_plot_channels(plot, image, flip=True)
    left, right = 0, bins
    axes.set_xlim(left, right)
    axes.set_ylim(0, max)
    figure.savefig("./test-output/test-flip.png")


if __name__ == "__main__":
    main()
