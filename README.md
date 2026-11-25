# pennyio (image operations)

Collection of frequently encountered operations and an opinionated implementation.

Opencv is used for the file operations.

## Installation

Add to dependencies:
```
"pennyio @ git+https://github.com/proximalf/roasted-garlic"
```

### CLI installation

```
uv tool install ./[cli] # inside the repo
# or
uv tool install --with "pennyio[cli] @ git+https://github.com/proximalf/roasted-garlic.git" pennyio
```

## Content

    
- load_image / save_image
Single function all formats load / save. 
When saving, the suffix will be used to determine write format.

- ImageFormat
`ImageFormat.format` - Static method to determine an images format.
`determine_image_format` - Function returns an ImageFormat Enum, which should help processing images of different dtypes, such as when plotting a histogram.

### Selection

selections are saved to file as a list of json objects.
```json
[
    {
        "name": "line",
        "type": "Shape::Line",
        "points": [[153.0,247.0], [518.0,413.0]],
    },
]
```

### Operations

- apply_colourmap - set the colourmap of an image, uses MPL colourmaps 
- bit-shift
