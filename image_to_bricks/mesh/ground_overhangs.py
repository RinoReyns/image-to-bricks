"""Columns that start in the air are extended down (--ground-overhangs G).

The voxel outline of a boot or leg often starts one or two plates higher than the cells beside it: the lowest
cell of a wall column has nothing under it, so its plate floats while building ("!" in the PDF) although the
column above it is tall. Here such a column is extended down to the next filled cell (or to the figure's first
layer, which stands on the base) when the gap is at most G plates and the column above is at least
`min_column` plates tall. Short specks (a 1-2 plate bit of fur) are left alone, and so are long overhangs (a hat
brim, a tail tip, a boot fold), whose gap is larger than G. New cells take the colour of the column above them.

Grid: bool [z, y, x] (figure grid, z = 0 is the first figure layer).
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

GROUND_MIN_COLUMN_PLATES = 3  # column above the gap must be at least this tall (--ground-min-column)

Fill = tuple[int, int, int, int]  # (z, y, x, z of the column cell whose colour it takes)


def _column_height(grid: NDArray[Any], z: int, y: int, x: int) -> int:
    height = 0
    while z + height < grid.shape[0] and grid[z + height, y, x]:
        height += 1
    return height


def _gap_below(grid: NDArray[Any], z: int, y: int, x: int) -> int:
    """Empty cells directly under (z, y, x) down to the next filled cell or the first layer (inclusive)."""
    gap = 0
    while z - gap - 1 >= 0 and not grid[z - gap - 1, y, x]:
        gap += 1
    return gap


def ground_overhangs(
    grid: NDArray[Any], max_gap: int, min_column: int = GROUND_MIN_COLUMN_PLATES
) -> tuple[NDArray[Any], list[Fill]]:
    """Copy of grid with the gaps under starting-in-the-air columns filled (see the module docstring), and the
    filled cells with the column cell their colour comes from."""
    original = grid.astype(bool)
    out = original.copy()
    fills: list[Fill] = []
    for z in range(1, original.shape[0]):
        starts = original[z] & ~original[z - 1]
        for y, x in zip(*np.nonzero(starts), strict=True):
            gap = _gap_below(original, z, y, x)
            if 0 < gap <= max_gap and _column_height(original, z, y, x) >= min_column:
                for filled_z in range(z - gap, z):
                    out[filled_z, y, x] = True
                    fills.append((filled_z, int(y), int(x), z))
    return out, fills
