import json
from pathlib import Path
from typing import Iterator, Literal, Sequence, TypedDict

from .point import Point
from .selection import SelectionArea, ShapeType

_SELECTION_EXTENSION = ".json"


class SelectionsJSON(TypedDict):
    id: str
    type: str
    coords: Sequence[Point]
    pad: float


def write_selections(path: Path, selections: list[SelectionArea], mode: Literal["a", "w"] = "w") -> None:
    """
    Write a selection to file. Sets the suffix to `.json`.

    Parameters
    ----------
    path: Path
        Path to save selection to.
    selection: SelectionArea
        SelectionArea to write to file.
    """
    path = path.with_suffix(_SELECTION_EXTENSION)
    with path.open(mode) as file:
        l = len(selections)

        file.write("[")  # for list of JSON
        for i, selection in enumerate(selections, 1):

            data: SelectionsJSON = {
                "id": selection.id,
                "type": str(selection.type),
                "coords": selection.coords,
                "pad": selection.pad,
            }

            file.write(json.dumps(data))
            if i != l:
                file.write(",\n")

        file.write("]")  # close list


def load_selections(path: Path) -> list[SelectionArea]:
    """
    Write a selection to file. Sets the suffix to `.json`.

    Parameters
    ----------
    path: Path
        Path to read selections from.

    Returns
    -------
    list[SelectionAreas]
    """
    path = path.with_suffix(_SELECTION_EXTENSION)
    text = path.read_text()

    try:
        data: list[SelectionsJSON] = json.loads(text)

        return [
            SelectionArea(
                [Point(*p) for p in row["coords"]], ShapeType(row["type"].split("::")[-1]), row["id"], row["pad"]
            )
            for row in data
        ]
    except:
        raise Exception(f"Failed to load selections file: {path}")
