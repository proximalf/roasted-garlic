import logging
from pathlib import Path
from typing import Collection, List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from pennyio.convert import convert_image
from pennyio.io import save_image
from pennyio.selection import SelectionArea
from pennyio.selection.io import write_selections, load_selections
from pennyio.test_data import TestImages
from pennyio.types import Image

output = Path(__file__).parent.parent / "test-output"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


def plot_line_profile(lines: Collection[SelectionArea]) -> None:

    image = TestImages.mono_jb()
    fig, axes = plt.subplots(1, 1)

    for i, line in enumerate(lines):
        mask = line.as_numpy_indices(image.shape)
        axes.plot(image[mask], label=f"{i+1}")

    fig.tight_layout()
    fig.legend()
    fig.savefig(output / "lines-plot.png")


def test_lines() -> None:
    line1 = SelectionArea.line((10, 10), (100, 100))
    line2 = SelectionArea.line((10, 100), (100, 10))
    line3 = SelectionArea.line((100, 100), (10, 10))
    line4 = SelectionArea.line((100, 10), (10, 100))
    line5 = SelectionArea.line((10, 100), (100, 100))
    lines = (
        line1,
        line2,
        line3,
        line4,
        line5,
    )
    plot_line_profile(lines)

    image = TestImages.mono_jb()
    marked_image = TestImages.mono_jb()
    for line in lines:
        mask = line.as_numpy_indices(image.shape)
        image[mask] = image[mask]
        marked_image[mask] = 0

    padded_line = SelectionArea.line((10, 10), (10, 100), pad=10)
    mask = padded_line.as_numpy_mask(image.shape)
    image[mask] = image[mask]
    marked_image[mask] = 255
    save_image(output / "lines-mask-unchanged.png", image)
    save_image(output / "lines-marked.png", marked_image)

    start = (10, 10)
    end = (300, 500)
    selection = SelectionArea.line(start, end)

    image = TestImages.mono_jb()
    p = selection.as_numpy_mask(image.shape)
    image[p] = image[p]

    start = (10, 300)
    end = (300, 10)
    selection = SelectionArea.line(start, end)
    p = selection.as_numpy_mask(image.shape)
    image[p] = image[p]
    # these should look jumbled
    save_image(output / "test-line-numpy-mask", image)


def test_polygon():
    points = [(10, 10), (25, 120), (150, 100), (50, 50), (100, 10), (10, 10)]

    selection = SelectionArea.polygon(points)

    image = TestImages.mono_jb()
    p = selection.as_numpy_mask(image.shape)
    image[p] = 20

    save_image(output / "test-poly", image)


def test_circle():
    circle = SelectionArea.circle(
        (100, 100),
        15,
    )

    image = TestImages.colour()
    mask = circle.as_numpy_mask(image.shape)
    image[mask] = image[mask] * 10
    save_image(output / "circle-mask.png", image)

    centre = (90, 90)
    radius = 100
    selection = SelectionArea.circle(centre=centre, radius=radius)

    image = TestImages.mono_jb()
    p = selection.as_numpy_mask(image.shape)
    image[p] = 20

    save_image(output / "test-circle", image)


def test_rectangle():
    rect = SelectionArea.rectangle((100, 100), (100, 100))

    image = TestImages.colour()
    mask = rect.as_numpy_mask(image.shape)
    image[mask] = image[mask] * 10
    save_image(output / "rect-mask.png", image)

    centre = (90, 90)
    size = (100, 100)
    selection = SelectionArea.rectangle(top_left=centre, size=size)

    image = TestImages.mono_jb()
    p = selection.as_numpy_mask(image.shape)
    image[p] = 20

    centre = (200, 200)
    size = (50, 50)
    selection = SelectionArea.rectangle(top_left=centre, size=size, centre=True)
    p = selection.as_numpy_mask(image.shape)
    image[p] = 200

    save_image(output / "test-rect", image)


def test_io() -> None:
    filename = output / "test-selection-io"
    
    start = (10, 300)
    end = (300, 10)
    
    selection1 = SelectionArea.line(start, end, id="line")
    selection2 = SelectionArea.circle((150, 150), 20, id = "cirle")

    write_selections(filename, [selection1, selection2])
    selections = load_selections(filename)

    image = TestImages.mono_jb()
    
    for selection in selections:
        p = selection.as_numpy_mask(image.shape)
        image[p] = 200

    save_image(output / "test-io", image)


def test_selections() -> None:
    test_lines()
    test_rectangle()
    test_circle()
    test_polygon()
    test_io()


if __name__ == "__main__":
    test_selections()
