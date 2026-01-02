from matplotlib.figure import Figure
from pennyio.test_data import TestImages

from pennyio.plotting.histogram import HistogramPlot

def main():
    image = TestImages.colour()
    figure = Figure()
    axes = figure.add_subplot()

    plot = HistogramPlot(axes, image)
    figure.savefig("./test.png")


if __name__ == "__main__":
    main()