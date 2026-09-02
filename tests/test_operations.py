from pennyio.test_data import TestImages
from pennyio.types import Image
from pennyio.operations import add, subtract, multiply, divide, crop, threshold


def basic(image: Image) -> None:
    """
    Expect the functions to operate exactly as.
    """
    pos = 20, 50
    assert add(image, image)[pos] == (image + image)[pos]
    assert subtract(image, image)[pos] == (image - image)[pos]
    assert multiply(image, image)[pos] == (image * image)[pos]
    assert divide(image, image)[pos] == (image / image)[pos]

if __name__ == "__main__":
    basic(TestImages.mono_jb())