"""
python -m test.test_qt
"""

import logging
from typing import Dict

try:
    # if Qt isn't installed this will fail
    from pennyio.qt import image_to_pixmap
except ImportError or ModuleNotFoundError:
    raise RuntimeError("test_qt - image_to_pixmap requires Qt - pip install pennyio[qt]")

import sys

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QGridLayout, QLabel, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

from pennyio.test_data import TestImages

TEST_IMAGES: Dict[str, np.ndarray] = {
    "grey8": TestImages.grey8(),
    "grey16": TestImages.grey16(),
    "rgb8": TestImages.rgb8(),
    "rgb16": TestImages.rgb16(),
    "rgba16": TestImages.rgba16(),
    "rgbaf16": TestImages.rgbaf16(),
    "rgba32": TestImages.rgbaf32(),
    "raw_cr2": TestImages.raw_cr2(),
    "float_jb": TestImages.float_jb(),
    "colour": TestImages.colour(),
}


class Window(QWidget):
    """
    Window widget to display the results of the conversion.
    """

    def __init__(self, images: Dict[str, np.ndarray]) -> None:
        super().__init__()

        layout = QGridLayout(self)

        for i, (name, image) in enumerate(images.items()):
            logger.debug(f"Loading Image: {name}")
            vb = QVBoxLayout()
            label = QLabel(f"Image: {name}")
            vb.addWidget(label)

            label = QLabel()
            pixmap = image_to_pixmap(image)
            label.setPixmap(pixmap)
            label.setScaledContents(True)  # scale to label size

            row = i // 3
            col = i % 3
            vb.addWidget(label)
            layout.addLayout(vb, row, col)

        self.setWindowTitle("test - image_to_pixmap")


def test_image_to_pixmap():
    """
    Test the conversion of a numpy array to a QImage and display the resulting pixmap.
    """

    app = QApplication(sys.argv)

    window = Window(TEST_IMAGES)
    window.setMaximumSize(500, 500)
    window.show()
    app.exec()


if __name__ == "__main__":
    test_image_to_pixmap()
