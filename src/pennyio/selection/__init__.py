try:
    from .selection import (
        SelectionArea, 
        ShapeType
        )
    from .crop import crop_image_from_selection
except ImportError or ModuleNotFoundError:
    raise RuntimeError("requires addtional packages - pip install pennyio[selection]")

