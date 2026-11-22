"""Colours set by hand: --paint / --paint-deep boxes and --front-oval fur bands."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from image_to_bricks.colours import fur_ovals
from image_to_bricks.consts import COLOR_NAMES
from image_to_bricks.options import Paint

PAINTED_CELLS_LISTED = 6  # painted cells named in the console line (the rest as "...")


def apply_paints(
    lego: NDArray[Any],
    surface: NDArray[np.bool_],
    grid: NDArray[Any] | None = None,
    paints: Sequence[Paint] = (),
) -> NDArray[Any]:
    """--paint: every visible (surface) cell inside each box (figure grid, ends exclusive, in the real figure size:
    NOT scaled by --boxes-for-studs) gets the colour. --paint-deep: every FILLED cell in the box, also the hidden
    wall cells behind the surface (so a piece that sits in the wall, e.g. a 1x2 brick as a nose with one stud hidden,
    is one colour all through). Applied after all colour clean-up, so small details such as eyes and a nose are not
    smoothed away. Later paints cover earlier ones. Returns lego itself without paints, else a painted copy."""
    if not paints:
        return lego
    lego = lego.copy()
    for paint in paints:
        painted = np.zeros(surface.shape, bool)
        painted[paint.box] = True
        painted &= grid.astype(bool) if paint.is_deep and grid is not None else surface
        lego[painted] = paint.colour_id
        _print_paint(paint, [tuple(int(v) for v in cell) for cell in np.argwhere(painted)])
    return lego


def _print_paint(paint: Paint, cells: list[tuple[int, ...]]) -> None:
    """Console line of one paint: colour, cell count, the first cells (z, y, x) or a warning if none."""
    listed = f" (z, y, x) {cells[:PAINTED_CELLS_LISTED]}{' ...' if len(cells) > PAINTED_CELLS_LISTED else ''}"
    print(
        f"Paint{' (deep)' if paint.is_deep else ''} {COLOR_NAMES.get(paint.colour_id, paint.colour_id)}: "
        f"{len(cells)} {'cell' if paint.is_deep else 'surface cell'}(s)"
        + (listed if cells else " - WARNING: no cell in the box, nothing painted")
    )


def paint_front_ovals(
    lego: NDArray[Any],
    grid: NDArray[Any],
    ovals: Sequence[fur_ovals.FrontOval],
    front: str | None,
    shell_thickness_studs: int,
) -> NDArray[Any]:
    """--front-oval (fur_ovals.py): oval colour bands on the front fur (front: the side the face looks to; the band
    also paints shell_thickness_studs of wall behind), then one console line per oval."""
    for oval in ovals:
        lego, painted = fur_ovals.apply_front_ovals(lego, grid, [oval], front, max(1, shell_thickness_studs))
        bands = ", ".join(f"{COLOR_NAMES.get(cid, cid)} {painted.get(cid, 0)}" for cid in oval.colour_ids)
        total = sum(painted.values())
        print(
            f"Front oval {oval.z0}:{oval.z1},{oval.h0}:{oval.h1}: {total} front cell(s) (+ the wall behind) "
            f"painted, middle outwards: {bands}" + ("" if total else " - WARNING: no fur cell in the box")
        )
    return lego
