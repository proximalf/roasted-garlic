# pennyio (image operations)

Collection of frequently encountered operations at work.

Opencv is used for the file operations.

## Installation

Add to dependencies:
```
"pennyio @ git+https://github.com/proximalf/roasted-garlic",
```

## Content

### Format
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
