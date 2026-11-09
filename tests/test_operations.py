from pathlib import Path
from pennyio.io import save_image
from pennyio.operations import add, crop, divide, multiply, subtract, threshold, apply_colourmap, bit_shift
from pennyio.test_data import TestImages
from pennyio.types import Image

out = Path("tests/output")
out.mkdir(exist_ok=True)

def basic(image: Image) -> None:
    """
    Expect the functions to operate exactly as.
    """
    pos = 20, 50
    assert add(image, image)[pos] == (image + image)[pos]
    assert subtract(image, image)[pos] == (image - image)[pos]
    assert multiply(image, image)[pos] == (image * image)[pos]
    assert divide(image, image)[pos] == (image / image)[pos]

def colourmap(image: Image) -> None:
    """
    Colourmap and bit shift test.
    """
    cm = apply_colourmap(image, "viridis", (image.min(), image.max()))
    save_image(out / "colourmap", cm)

    bs = bit_shift(image, "left", 1)
    save_image(out / "bit-shift", bs)


if __name__ == "__main__":
    basic(TestImages.mono_jb())
    colourmap(TestImages.mono_jb())
