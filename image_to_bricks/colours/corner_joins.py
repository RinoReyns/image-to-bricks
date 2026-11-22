"""Close corner-only steps in thin parts (--join-corners N).

A thin sheet that runs diagonally through the grid (a hat's turned-up brim, a thin crown wall) is voxelised as a
staircase of cells that touch only at their corners. In LEGO those cells become separate 1x1 stacks with gaps
between them: the part looks spotty and hangs on single studs. Here, inside one layer, two filled cells that touch
only at a corner (both cells between them empty) get one of the two between cells filled, when
- they belong to different side-connected parts of the layer, the smaller of which has <= N cells (a big part
  touching another big part at a corner is a real gap, e.g. two boots), and
- the two parts are already one body within JOIN_SLAB_LAYERS plates above and below (so a corner contact between
  two separate pieces of the figure is not closed).
The between cell with more filled neighbours (6 faces, 3D) is used. Runs on the coloured grid (see
shape_edits.fix_gaps_and_pinholes): new cells take the colour of the nearest cell.

Grid: bool [z, y, x].
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve, label

JOIN_SLAB_LAYERS = 2  # plates above and below in which the two parts must already be connected
_SIDE_CONNECTED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
_FACE_NEIGHBOURS = np.zeros((3, 3, 3), np.int32)
_FACE_NEIGHBOURS[0, 1, 1] = _FACE_NEIGHBOURS[2, 1, 1] = 1
_FACE_NEIGHBOURS[1, 0, 1] = _FACE_NEIGHBOURS[1, 2, 1] = 1
_FACE_NEIGHBOURS[1, 1, 0] = _FACE_NEIGHBOURS[1, 1, 2] = 1
_DIAGONALS = ((1, 1), (1, -1))


def _is_one_body_nearby(grid: NDArray[Any], z: int, cell_a: tuple[int, int], cell_b: tuple[int, int]) -> bool:
    """True if the two cells of layer z are face-connected through filled cells within JOIN_SLAB_LAYERS layers."""
    lowest, highest = max(z - JOIN_SLAB_LAYERS, 0), min(z + JOIN_SLAB_LAYERS, grid.shape[0] - 1)
    bodies, _ = label(grid[lowest : highest + 1])
    body_a = bodies[z - lowest][cell_a]
    return bool(body_a > 0 and body_a == bodies[z - lowest][cell_b])


def _corner_join_in_layer(
    grid: NDArray[Any], z: int, max_part_cells: int, neighbour_counts: NDArray[Any]
) -> tuple[int, int] | None:
    """The first between cell (y, x) to fill in layer z, or None."""
    layer = grid[z]
    parts, _ = label(layer, structure=_SIDE_CONNECTED)
    part_sizes = np.bincount(parts.ravel())
    height, width = layer.shape
    for y, x in zip(*np.nonzero(layer), strict=True):
        for step_y, step_x in _DIAGONALS:
            other_y, other_x = y + step_y, x + step_x
            if not (0 <= other_y < height and 0 <= other_x < width) or not layer[other_y, other_x]:
                continue
            if layer[y, other_x] or layer[other_y, x]:
                continue  # already side-connected through a between cell
            part_a, part_b = parts[y, x], parts[other_y, other_x]
            if part_a == part_b or min(part_sizes[part_a], part_sizes[part_b]) > max_part_cells:
                continue
            if not _is_one_body_nearby(grid, z, (y, x), (other_y, other_x)):
                continue
            between = [(y, other_x), (other_y, x)]
            return max(between, key=lambda cell: neighbour_counts[z][cell])
    return None


def join_corners(grid: NDArray[Any], max_part_cells: int) -> tuple[NDArray[Any], list[tuple[int, int, int]]]:
    """Copy of grid with corner-only steps of thin parts closed (see the module docstring), and the added cells
    (z, y, x)."""
    out = grid.astype(bool).copy()
    added = []
    for z in range(out.shape[0]):
        if not out[z].any():
            continue
        while True:
            neighbour_counts = convolve(out.astype(np.int32), _FACE_NEIGHBOURS, mode="constant")
            cell = _corner_join_in_layer(out, z, max_part_cells, neighbour_counts)
            if cell is None:
                break
            out[z][cell] = True
            added.append((z, int(cell[0]), int(cell[1])))
    return out, added
