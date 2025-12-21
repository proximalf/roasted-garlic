"""
python -m test.test_qt
"""
from typing import Dict

try:
    # if Qt isn't installed this will fail
    from pennyio.qt import numpy_to_pixmap
except ImportError or ModuleNotFoundError:
    raise RuntimeError("test_qt - numpy_to_pixmap requires Qt - pip install pennyio[qt]")

import numpy as np

import sys
from PySide6.QtWidgets import QApplication, QLabel, QWidget, QGridLayout, QVBoxLayout
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt


from .test_lib import TEST_IMAGES


class Window(QWidget):
    """
    Window widget to display the results of the conversion.
    """

    def __init__(self, images: Dict[str, np.ndarray]) -> None:
        super().__init__()

        layout = QGridLayout(self)

        for i, (name, image) in enumerate(images.items()):
            vb = QVBoxLayout()
            label = QLabel(f"Image: {name}")     
            vb.addWidget(label)

            label = QLabel()            
            pixmap = numpy_to_pixmap(image)
            label.setPixmap(pixmap)
            label.setScaledContents(True)  # scale to label size

            row = i // 3
            col = i % 3
            vb.addWidget(label)
            layout.addLayout(vb, row, col)

        self.setWindowTitle("test - numpy_to_pixmap")

def test_numpy_to_pixmap():
    """
    Test the conversion of a numpy array to a QImage and display the resulting pixmap.
    """

    app = QApplication(sys.argv)
    
    window = Window(TEST_IMAGES)
    window.show()
    app.exec()

if __name__ == "__main__":
    test_numpy_to_pixmap()