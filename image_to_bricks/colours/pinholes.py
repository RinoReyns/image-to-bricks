"""See-through pinholes in thin parts (--fill-pinholes).

A part thinner than a cell (a turned-up hat brim, a cape edge) that runs diagonally through the grid can lose a
cell here and there when it is voxelised: from the side you can look through the build. Such a pinhole is a small
hole in the silhouette of the model seen along one axis. For each one, the cell along the line of sight that has
the most filled neighbours is filled, so the sheet closes where it was broken. Real gaps (between the legs, under a
raised arm) are bigger than the limit and stay open.

Grid: bool [z, y, x]; views along x and along y (front / back / left / right, as in the flat views).
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_fill_holes, convolve, label

FILL_PINHOLES_MAX = 3  # largest hole (cells of the silhouette) that is filled (--fill-pinholes N)
Y_AXIS, X_AXIS = 1, 2  # grid axes [z, y, x]: lines of sight along y (front / back view) and x (side views)


def _neighbour_count(grid: NDArray[Any]) -> NDArray[Any]:
    """int [z, y, x]: filled cells among the 26 neighbours of every cell."""
    kernel = np.ones((3, 3, 3), np.int32)
    kernel[1, 1, 1] = 0
    neighbour_counts: NDArray[Any] = convolve(grid.astype(np.int32), kernel, mode="constant", cval=0)
    return neighbour_counts


def _silhouette_holes(grid: NDArray[Any], axis: int, max_cells: int) -> NDArray[Any]:
    """bool [projection]: pixels of holes of at most max_cells in the silhouette of `grid` seen along `axis`."""
    seen = grid.any(axis=axis)
    holes = binary_fill_holes(seen) & ~seen
    labels, n = label(holes)
    if not n:
        return np.zeros_like(holes)
    sizes = np.bincount(labels.ravel())
    small = np.nonzero(sizes <= max_cells)[0]
    return np.isin(labels, small[small > 0])


def fill_pinholes(grid: NDArray[Any], max_cells: int = FILL_PINHOLES_MAX) -> tuple[NDArray[Any], int]:
    """Copy of grid with the pinholes (silhouette holes of <= max_cells, seen along x or y) closed; one cell per
    hole pixel, at the depth with the most filled neighbours. Returns (grid, cells added)."""
    out = grid.astype(bool).copy()
    added = 0
    for axis in (X_AXIS, Y_AXIS):
        holes = _silhouette_holes(out, axis, max_cells)
        if not holes.any():
            continue
        count = _neighbour_count(out)
        for a, b in zip(*np.nonzero(holes), strict=True):
            depth = _best_depth(out, count, axis, a, b)
            cell = (a, depth, b) if axis == Y_AXIS else (a, b, depth)
            if depth >= 0 and not out[cell]:
                out[cell] = True
                added += 1
    return out, added


def _line(arr: NDArray[Any], axis: int, a: int, b: int) -> NDArray[Any]:
    """The cells of arr [z, y, x] along `axis` (1 = y, 2 = x) at projection pixel (a, b) = (z, x) or (z, y)."""
    return arr[a, :, b] if axis == Y_AXIS else arr[a, b, :]


def _best_depth(grid: NDArray[Any], count: NDArray[Any], axis: int, a: int, b: int) -> int:
    """Depth along the line of sight to fill: where the neighbouring lines of sight (one pixel up / down / left /
    right in the silhouette) are filled, so a broken sheet continues; ties go to the most filled 26-neighbours.
    -1 if no cell along the line touches the model."""
    rows, cols = grid.shape[0], grid.shape[Y_AXIS if axis == X_AXIS else X_AXIS]
    sheet = np.zeros(grid.shape[axis], np.int32)
    for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        if 0 <= a + da < rows and 0 <= b + db < cols:
            sheet += _line(grid, axis, a + da, b + db)
    touch = _line(count, axis, a, b)
    if touch.max() == 0:
        return -1
    return int(np.lexsort((-touch, -sheet))[0])
