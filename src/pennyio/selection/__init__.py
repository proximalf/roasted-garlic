try:
    from .crop import crop_image_from_selection
    from .selection import SelectionArea, ShapeType
except ImportError or ModuleNotFoundError:
    raise RuntimeError("requires addtional packages - pip install pennyio[selection]")
