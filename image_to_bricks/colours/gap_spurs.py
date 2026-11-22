"""Single-cell spurs that close a gap between two parts (--open-gaps).

Voxelising can leave one stray cell sticking out of a part into a narrow gap (a heel cell between two boots). It
touches the other part only at a corner, so the parts stay separate in the build, but from the outside the gap looks
closed. Such a cell is removed when it (a) has exactly one side neighbour in its layer, (b) has nothing directly
above or below, and (c) is the only link between two parts of its layer that touch nowhere else (8-connected).
Thin details made of several cells (a tail tip, a sword) are kept: their cells have neighbours above or below, or
more than one side neighbour, or they link nothing.

Grid: bool [z, y, x].
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve, label

_SIDES = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.int32)
_EIGHT = np.ones((3, 3), bool)


def _parts(layer: NDArray[Any]) -> int:
    return int(label(layer, structure=_EIGHT)[1])


def gap_spurs(grid: NDArray[Any]) -> list[tuple[int, int, int]]:
    """Cells (z, y, x) that are single-cell spurs closing a gap (see the module docstring)."""
    g = grid.astype(bool)
    out = []
    for z in range(g.shape[0]):
        layer = g[z]
        if not layer.any():
            continue
        sides = convolve(layer.astype(np.int32), _SIDES, mode="constant")
        above = g[z + 1] if z + 1 < g.shape[0] else np.zeros_like(layer)
        below = g[z - 1] if z > 0 else np.zeros_like(layer)
        candidates = layer & (sides == 1) & ~above & ~below
        if not candidates.any():
            continue
        before = _parts(layer)
        for y, x in zip(*np.nonzero(candidates), strict=True):
            trial = layer.copy()
            trial[y, x] = False
            if _parts(trial) > before:
                out.append((z, int(y), int(x)))
    return out


def open_gaps(grid: NDArray[Any]) -> tuple[NDArray[Any], list[tuple[int, int, int]]]:
    """Copy of grid without the gap spurs, and the removed cells."""
    spurs = gap_spurs(grid)
    out = grid.astype(bool).copy()
    for c in spurs:
        out[c] = False
    return out, spurs
