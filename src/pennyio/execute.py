from pathlib import Path

try:
    import click
    from click import echo
except ImportError:
    raise ImportError("This feature requires click. Install with pennyio[cli] or pennyio[all]")

from .io import load_image, save_image


@click.command()
@click.argument("path", type=Path)
@click.option("-o", "--overwrite", is_flag=True)
def main(path: Path, overwrite: bool = False) -> None:
    """
    A cli interface for converting an image to mono.

    Set flag to overwrite.
    """

    if not path.exists():
        click.echo("Path does not exist!")
        return

    image = load_image(path, True)

    output = path.with_stem(path.stem + "-mono") if not overwrite else path
    save_image(output, image)

    click.echo("Converted!")
