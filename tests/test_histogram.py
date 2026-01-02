from pennyio.types import Image
from matplotlib.figure import Figure

from pennyio.plotting.histogram import (
    plot_image_histogram,
    update_histogram_plot_channels,
)
from pennyio.test_data import TestImages
from pennyio.convert import convert_array_to_mono

def test_histogram(filename: str, image: Image) -> None:
    print(filename, image.max(), image.dtype)
    figure = Figure()
    axes = figure.add_subplot()

    plot = plot_image_histogram(axes, image)
    figure.savefig(f"./test-output/{filename}.png")
    
    # Test if we can just update the data.
    x_max, y_max = update_histogram_plot_channels(plot, image, flip=True)

    axes.set_xlim(0, x_max)
    axes.set_ylim(0, y_max)
    
    output = f"./test-output/{filename}-flip.png"
    figure.savefig(output)
    print(f"plot saved: {output}\n")



def main():
    test_histogram("colour", TestImages.colour())
    test_histogram("colour-float", TestImages.float_jb())
    test_histogram("colour-mono", convert_array_to_mono(TestImages.colour()))

    test_histogram("raw", TestImages.raw_cr2())

    test_histogram("raw-mono", convert_array_to_mono(TestImages.raw_cr2()))
    


if __name__ == "__main__":
    main()
