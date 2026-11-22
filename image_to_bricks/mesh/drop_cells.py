"""--drop-box: cells the user chose to leave out of the figure (e.g. a one-cell bump that would hang in the air)."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

GridBox = tuple[slice, slice, slice]  # (slice z, slice y, slice x) in the figure grid, ends exclusive


def drop_cells_in_boxes(
    grid: NDArray[Any], boxes: Sequence[GridBox]
) -> tuple[NDArray[Any], list[list[tuple[int, int, int]]]]:
    """Empty every filled cell inside the boxes. grid: bool [z, y, x] (figure grid, not changed in place).
    Returns (new grid, per box the removed cells as (z, y, x), sorted)."""
    new_grid = grid.astype(bool).copy()
    removed_per_box = []
    for box in boxes:
        inside = np.zeros_like(new_grid)
        inside[box] = True
        removed = inside & new_grid
        removed_per_box.append([(int(z), int(y), int(x)) for z, y, x in np.argwhere(removed)])
        new_grid &= ~removed
    return new_grid, removed_per_box


def format_box(box: GridBox) -> str:
    """'Z0:Z1,Y0:Y1,X0:X1' as typed on the command line."""
    return ",".join(f"{part.start}:{part.stop}" for part in box)
