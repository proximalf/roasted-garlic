# Image tools

Collection of commonly used tools, assembled for my own usage at work.

Opencv is used for the IO operations.

## Installation

Add to dependencies:
```
"pennyio @ git+https://github.com/proximalf/roasted-garlic",
```

## Functions

### `numpy_to_qimage`
Requires the package to be installed as `pip install pennyio[qt]`
This function returns a `QImage` from a provided numpy array, if the display image is broken check the values of the array provided and the documentation of this function.