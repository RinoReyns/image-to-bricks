"""The built model: placed pieces (PlacedBrick) on the cell grid (LegoModel), and small helpers on pieces.

Coordinates: x, y in studs and z in plate layers of the model grid (bool [z, y, x]); the base, if any, is the lowest
LegoModel.base_layers layers.
"""

import csv
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from image_to_bricks.consts import BRICK_HEIGHT_PLATES, colour_name

L_SHAPE = "corner 2x2 (L, 3 studs)"


@dataclass
class PlacedBrick:
    """One placed piece: a rectangular plate / brick, or an L piece (2x2 corner plate / brick, see notch)."""

    x: int  # stud position (grid column, 0-indexed)
    y: int  # stud position (grid row, 0-indexed)
    z: int  # plate layer index (0-indexed, bottom to top)
    w: int  # width in studs (x extent)
    d: int  # depth in studs (y extent)
    h_plates: int  # height in plates (1 or 3)
    color: int  # palette colour id
    # -1 = rectangle; 0..3 = L piece (2x2 box minus one cell), the empty cell:
    # 0 = (x, y), 1 = (x+1, y), 2 = (x, y+1), 3 = (x+1, y+1)
    notch: int = -1


def piece_cells(b: PlacedBrick) -> list[tuple[int, int]]:
    """[(x, y)] studs covered by the piece (rectangle, or the 3 cells of an L piece)."""
    out = [(b.x + i, b.y + j) for j in range(b.d) for i in range(b.w)]
    if b.notch >= 0:
        out.remove((b.x + (b.notch & 1), b.y + (b.notch >> 1)))
    return out


def owner_grid(bricks: Sequence[PlacedBrick | None], grid_shape: tuple[int, int, int]) -> NDArray[np.int32]:
    """int32 [z, y, x]: index in `bricks` of the piece on each cell (-1 = empty); None entries (pieces a pass has
    removed, kept in place so the other indices stay valid) are skipped."""
    owner = np.full(grid_shape, -1, np.int32)
    for index, brick in enumerate(bricks):
        if brick is None:
            continue
        for x, y in piece_cells(brick):
            owner[brick.z : brick.z + brick.h_plates, y, x] = index
    return owner


def piece_mask(b: PlacedBrick) -> NDArray[np.bool_]:
    """bool [d, w] footprint of the piece (all True for a rectangle)."""
    m = np.ones((b.d, b.w), bool)
    if b.notch >= 0:
        m[b.notch >> 1, b.notch & 1] = False
    return m


@dataclass
class LegoModel:
    """The finished build: the cell grid and the pieces that cover it exactly (saved with --save-model)."""

    grid: NDArray[np.bool_]  # [z, y, x], True = filled
    bricks: list[PlacedBrick] = field(default_factory=list)
    base_layers: int = 0  # the lowest N plate layers are the stand (0 = no base)
    base_color: int | None = None
    base_margin: int = 0  # studs of base around the figure's own grid (0 without a base)
    # [(z, y, x, text)]: pieces swapped for a special part (--accent)
    accents: list[tuple[int, int, int, str]] = field(default_factory=list)
    front: str | None = None  # side of the grid the figure faces: "-y", "+y", "-x", "+x" or None (--front)


def plate_origin(grid: NDArray[np.bool_], base_layers: int = 0) -> tuple[int, int]:
    """(x, y) of the base plate's corner in grid coordinates: the axis numbers of the layer maps count from
    here (0, 0). Footprint of the bottom layer (= the base); without a base, of the whole figure."""
    foot = grid[0] if base_layers else grid.any(axis=0)
    ys, xs = np.nonzero(foot)
    return (int(xs.min()), int(ys.min())) if len(xs) else (0, 0)


def front_stud(b: PlacedBrick, front: str | None) -> tuple[tuple[int, int], bool]:
    """Stud of piece b nearest the front side, and True if the piece lies across the front direction
    (both ends equally near)."""
    if front in ("-y", "+y"):
        return (b.x, b.y if front == "-y" else b.y + b.d - 1), b.w > 1 and b.d == 1
    if front in ("-x", "+x"):
        return (b.x if front == "-x" else b.x + b.w - 1, b.y), b.d > 1 and b.w == 1
    return (b.x, b.y), False


def export_inventory_csv(model: LegoModel, path: str) -> None:
    """Write the piece list (section, type, size, colour, quantity), sorted, with a total line."""
    counts: dict[tuple[str, int, int, int, int, bool], int] = {}
    print("Exporting inventory CSV...")
    for b in model.bricks:
        w, d = sorted((b.w, b.d), reverse=True)  # normalize orientation
        section = "base" if b.z < model.base_layers else "figure"
        key = (section, w, d, b.h_plates, b.color, b.notch >= 0)
        counts[key] = counts.get(key, 0) + 1

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["section", "piece_type", "width_studs", "depth_studs", "height_plates", "color_name", "quantity", "shape"]
        )
        for (section, w, d, h_plates, color, is_l), qty in sorted(counts.items()):
            piece_type = ("Brick" if h_plates == BRICK_HEIGHT_PLATES else "Plate") + (" (special)" if is_l else "")
            writer.writerow(
                [
                    section,
                    piece_type,
                    w,
                    d,
                    h_plates,
                    colour_name(color),
                    qty,
                    L_SHAPE if is_l else "",
                ]
            )
        writer.writerow([])
        writer.writerow(["TOTAL PIECES", "", "", "", "", "", len(model.bricks), ""])


def grid_to_xyz_bool(grid: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """grid is [z, y, x]; matplotlib's ax.voxels wants filled[x, y, z]."""
    return np.transpose(grid, (2, 1, 0))
