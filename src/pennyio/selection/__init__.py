try:
    from .crop import crop_image_from_selection
    from .selection import SelectionArea, ShapeType
    from .point import Point, POINT_LIKE
    from .mask import PixelIndices
    from .io import write_selections, load_selections
except ImportError or ModuleNotFoundError:
    raise RuntimeError("requires addtional packages - pip install pennyio[selection]")
