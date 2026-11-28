"""The pictures of one character: which files, in which order, and loading them."""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

FRONT_VIEW = "front"
VIEW_NAMES = (FRONT_VIEW, "left", "back", "right")  # order the pictures are loaded and cut out in
TEXTURE_ORDER = (FRONT_VIEW, "back", "left", "right")  # order the colour projection sums the pictures in
PICTURE_SUFFIX = ".png"


@dataclass(frozen=True)
class ViewPictures:
    """Picture file per view name (front required; left, back, right optional) and the folder whose SOURCE.json
    records who owns the pictures."""

    files: dict[str, Path]
    source_folder: Path

    @property
    def character_name(self) -> str:
        """Name for the output files: the pictures folder, or the front picture's name when files were given."""
        front = self.files[FRONT_VIEW]
        return self.source_folder.name if front.stem == FRONT_VIEW else front.stem


def find_views_in_folder(folder: Path) -> ViewPictures:
    """front.png (required) and any of left.png, back.png, right.png in folder. Raises FileNotFoundError without a
    front picture."""
    folder = Path(folder)
    front = folder / f"{FRONT_VIEW}{PICTURE_SUFFIX}"
    if not front.is_file():
        raise FileNotFoundError(f"{front} is missing")
    files = {name: folder / f"{name}{PICTURE_SUFFIX}" for name in VIEW_NAMES}
    return ViewPictures({name: path for name, path in files.items() if path.is_file()}, folder)


def collect_view_files(files_by_view: dict[str, Path | None]) -> ViewPictures:
    """Pictures given one by one (any file names); the folder of the front picture holds SOURCE.json. Raises
    FileNotFoundError when the front picture or a given picture does not exist, ValueError for unknown view names."""
    unknown = sorted(set(files_by_view) - set(VIEW_NAMES))
    if unknown:
        raise ValueError(f"unknown view names {unknown}, allowed: {', '.join(VIEW_NAMES)}")
    given = {name: Path(path) for name, path in files_by_view.items() if path is not None}
    if FRONT_VIEW not in given:
        raise FileNotFoundError("a front picture is required")
    missing = [str(path) for path in given.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"picture(s) not found: {', '.join(missing)}")
    ordered = {name: given[name] for name in VIEW_NAMES if name in given}
    return ViewPictures(ordered, given[FRONT_VIEW].parent)


def load_view_images(pictures: ViewPictures) -> dict[str, Image.Image]:
    """{view name: image} in VIEW_NAMES order, every file fully read and closed."""
    images = {}
    for name, path in pictures.files.items():
        with Image.open(path) as image:
            images[name] = image.copy()
    return images
