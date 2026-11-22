"""--front-oval: regular fur pattern on the front of the figure: concentric oval bands of colours (light in the middle,
darker outwards), painted on the cells seen from the front and on the wall cells right behind them (so a piece that
reaches into the wall keeps one colour). Only cells that already have one of the band colours (the fur) change, so a
belt, a cape or the eyes painted later stay as they are."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

DEFAULT_FRONT = "-y"  # side the face looks to when --front is not given (same as view_sides.DEFAULT_FRONT)


@dataclass(frozen=True)
class FrontOval:
    """Box on the FRONT VIEW (figure grid, ends exclusive, real figure size): z0..z1 vertical, h0..h1 horizontal (x
    when the figure faces -y / +y, y when it faces -x / +x), and the band colour ids from the middle outwards."""

    z0: int
    z1: int
    h0: int
    h1: int
    colour_ids: tuple[int, ...]


def front_axes(front: str | None) -> tuple[int, bool, int]:
    """(depth axis, scan from the high end?, horizontal axis) of grid[z, y, x] for a figure facing `front`."""
    front = front or DEFAULT_FRONT
    depth_axis = 1 if front[1] == "y" else 2
    horizontal_axis = 2 if depth_axis == 1 else 1
    return depth_axis, front[0] == "+", horizontal_axis


def band_index(z: int, h: int, oval: FrontOval) -> int | None:
    """Index of the colour band of front-view cell (z, h): 0 in the middle, len(colours) - 1 at the rim; None outside
    the oval inscribed in the box."""
    centre_z, centre_h = (oval.z0 + oval.z1 - 1) / 2, (oval.h0 + oval.h1 - 1) / 2
    half_z, half_h = max((oval.z1 - oval.z0) / 2, 0.5), max((oval.h1 - oval.h0) / 2, 0.5)
    radius = float(np.hypot((z - centre_z) / half_z, (h - centre_h) / half_h))
    if radius > 1.0:
        return None
    return min(int(radius * len(oval.colour_ids)), len(oval.colour_ids) - 1)


def _wall_cells_from_front(filled_column: NDArray[Any], wall_cells: int, *, from_high_end: bool) -> list[int]:
    """Depth indices of the first `wall_cells` filled cells seen from the front (a run of consecutive cells)."""
    order = range(len(filled_column) - 1, -1, -1) if from_high_end else range(len(filled_column))
    run: list[int] = []
    for depth in order:
        if filled_column[depth]:
            run.append(depth)
            if len(run) == wall_cells:
                break
        elif run:
            break
    return run


def apply_front_ovals(
    lego: NDArray[Any], grid: NDArray[Any], ovals: list[FrontOval], front: str | None, wall_cells: int
) -> tuple[NDArray[Any], dict[int, int]]:
    """Paint the oval bands. lego: colour ids [z, y, x]; grid: bool filled (solid, before hollowing); wall_cells: how
    many cells deep each front column is painted (the shell thickness). Later ovals cover earlier ones. Returns
    (new colours, {colour id: front cells painted})."""
    depth_axis, from_high_end, horizontal_axis = front_axes(front)
    lego = lego.copy()
    painted: dict[int, int] = {}
    for oval in ovals:
        fur = np.isin(lego, list(oval.colour_ids))
        for z in range(max(oval.z0, 0), min(oval.z1, grid.shape[0])):
            for h in range(max(oval.h0, 0), min(oval.h1, grid.shape[horizontal_axis])):
                band = band_index(z, h, oval)
                if band is None:
                    continue
                index: list[int | slice] = [z, slice(None), slice(None)]
                index[horizontal_axis] = h
                run = _wall_cells_from_front(grid[tuple(index)], wall_cells, from_high_end=from_high_end)
                if not run or not fur[tuple(_cell(z, h, run[0], depth_axis, horizontal_axis))]:
                    continue
                colour = oval.colour_ids[band]
                for depth in run:
                    lego[tuple(_cell(z, h, depth, depth_axis, horizontal_axis))] = colour
                painted[colour] = painted.get(colour, 0) + 1
    return lego, painted


def _cell(z: int, h: int, depth: int, depth_axis: int, horizontal_axis: int) -> tuple[int, int, int]:
    """[z, y, x] of the cell at front-view (z, h) and the given depth."""
    cell = [z, 0, 0]
    cell[depth_axis], cell[horizontal_axis] = depth, h
    return cell[0], cell[1], cell[2]
