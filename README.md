# Image tools

Collection of commonly used tools, assembled for my own usage at work.

Opencv is used for the IO operations.

## Installation

Add to dependencies:
```
"pennyio @ git+https://github.com/proximalf/roasted-garlic",
```

## Functions

### `determine_image_format`
Function returns an ImageFormat Enum, which should help processing images of different dtypes, such as when plotting a histogram.

### `image_to_pixmap`
Requires the package to be installed as `uv pip install pennyio[qt]`
This function returns a `QImage` from a provided numpy image array, if the displayed image is broken check the values of the array provided and the documentation of this function.