from pathlib import Path
from typing import Generator

from shapely import from_geojson, to_geojson

from .selection import SelectionArea, ShapeType

_SELECTION_EXTENSION = ".aoi"


def save_selection(path: Path, selection: SelectionArea, mode="w") -> None:
    """
    Write a selection to file. Sets the suffix to `.aoi`.

    Parameters
    ----------
    path: Path
        Path to save selection to.
    selection: SelectionArea
        SelectionArea to write to file.
    """
    path = path.with_suffix(_SELECTION_EXTENSION)
    with path.open(mode) as file:
        file.write(f"{selection.id}, {selection.type}, {to_geojson(selection.shape)}\n")


def load_selections(path: Path) -> Generator[SelectionArea, ..., ...]:
    """
    Load a selection from file.

    Returns
    -------
    SelectionArea
    """
    path = path.with_suffix(_SELECTION_EXTENSION)
    text = path.read_text()

    for line in text.split("\n"):
        try:
            id, type, shape = line.split(", ")
            shape = from_geojson(shape)
            type = ShapeType(type.split("::")[-1])
        except:
            raise

        yield SelectionArea(shape, type, id)


def load_selection(path: Path) -> SelectionArea:
    """
    Load a selection from file.

    Returns
    -------
    SelectionArea
    """
    path = path.with_suffix(_SELECTION_EXTENSION)
    text = path.read_text()

    try:
        id, type, shape = text.split(", ")
        shape = from_geojson(shape)
        type = ShapeType(type.split("::")[-1])
    except:
        raise

    return SelectionArea(shape, type, id)
