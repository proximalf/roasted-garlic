import logging
from pathlib import Path
from typing import Collection, List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from pennyio.convert import convert_image
from pennyio.io import save_image
from pennyio.selection import SelectionArea
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


def test_selections() -> None:
    """
    Test `determine_image_format` function returns the correct ImageFormat Enum.
    """

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

    padded_line = SelectionArea.line((10, 10), (10, 100), 10)
    mask = padded_line.as_numpy_mask(image.shape)
    image[mask] = image[mask]
    marked_image[mask] = 255
    save_image(output / "lines-mask-unchanged.png", image)
    save_image(output / "lines-marked.png", marked_image)

    rect = SelectionArea.rectangle((100, 100), (100, 100))

    image = TestImages.colour()
    mask = rect.as_numpy_mask(image.shape)
    image[mask] = image[mask] * 10
    save_image(output / "rect-mask.png", image)

    circle = SelectionArea.circle(
        (100, 100),
        15,
    )

    image = TestImages.colour()
    mask = circle.as_numpy_mask(image.shape)
    image[mask] = image[mask] * 10
    save_image(output / "circle-mask.png", image)


if __name__ == "__main__":
    test_selections()
