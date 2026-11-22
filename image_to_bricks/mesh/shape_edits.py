"""Edits of the cell grid: thickening, sheets, prop, gap / pinhole / corner / overhang fixes, drop boxes.

Grids are bool [z, y, x] (x, y in studs, z in plate layers); a box is (slice z, slice y, slice x), ends exclusive.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label

from image_to_bricks.colours import corner_joins, gap_spurs, pinholes
from image_to_bricks.colours.colour_sampling import fill_from_nearest
from image_to_bricks.mesh import drop_cells, ground_overhangs
from image_to_bricks.options import (
    Box,
    ShapeOptions,
)

SIDE_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # (step y, step x) to the four side neighbours of a stud


def _side_neighbours(y: int, x: int) -> list[tuple[int, int]]:
    """(y, x) of the four side neighbours of a stud (may lie outside the grid)."""
    return [(y + step_y, x + step_x) for step_y, step_x in SIDE_STEPS]


def thicken_thin(grid: NDArray[np.bool_], box: Box | None = None) -> tuple[NDArray[np.bool_], int]:
    """Make parts that are one stud wide in a layer sturdier. A cell is thin if it lies in no filled 2x2 block
    of its layer; every thin cell gets its empty 4-neighbours (a plus), so two consecutive layers of a thin
    part (even when it steps diagonally) overlap by at least 2 studs instead of hanging by one. Thick parts
    never change and nothing is removed. With `box` = (slice z, slice y, slice x) only thin cells inside it are
    thickened (a whole figure has many 1-wide edges; this is meant for one thin part such as the tail).
    Returns (new grid, number of cells added)."""
    filled = grid.astype(bool)
    thickened = filled.copy()
    layer_count, depth_y, depth_x = filled.shape
    for z in range(layer_count):
        layer = filled[z]
        if not layer.any():
            continue
        full_blocks = layer[:-1, :-1] & layer[1:, :-1] & layer[:-1, 1:] & layer[1:, 1:]
        in_full_block = np.zeros_like(layer)
        for step_y in (0, 1):
            for step_x in (0, 1):
                in_full_block[step_y : depth_y - 1 + step_y, step_x : depth_x - 1 + step_x] |= full_blocks
        thin = layer & ~in_full_block
        if box is not None:
            if not (box[0].start <= z < box[0].stop):
                continue
            inside_box = np.zeros_like(thin)
            inside_box[box[1], box[2]] = True
            thin &= inside_box
        for y, x in np.argwhere(thin):
            for neighbour_y, neighbour_x in _side_neighbours(y, x):
                if 0 <= neighbour_y < depth_y and 0 <= neighbour_x < depth_x:
                    thickened[z, neighbour_y, neighbour_x] = True
    return thickened, int(thickened.sum() - filled.sum())


def add_prop(grid: NDArray[np.bool_], box: Box, size: int = 2) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """A post from the lowest cell inside `box` that has nothing under it down to the ground (or the first filled
    cell), so a hanging part (tail tip) can be built bottom-up. Only empty cells are filled.
    Returns (new grid, mask of the added cells)."""
    filled = grid.astype(bool)
    _, depth_y, depth_x = filled.shape
    prop = np.zeros_like(filled)
    hanging = np.zeros_like(filled)
    hanging[1:] = filled[1:] & ~filled[:-1]
    inside_box = np.zeros_like(filled)
    inside_box[box] = True
    hanging &= inside_box
    if not hanging.any():
        return filled, prop
    lowest_z = int(np.nonzero(hanging.any(axis=(1, 2)))[0].min())
    ys, xs = np.nonzero(hanging[lowest_z])
    centre_y, centre_x = ys.mean(), xs.mean()
    nearest = int(np.argmin((ys - centre_y) ** 2 + (xs - centre_x) ** 2))
    # keep the whole post inside the grid
    y0, x0 = min(int(ys[nearest]), depth_y - size), min(int(xs[nearest]), depth_x - size)
    for step_y in range(size):
        for step_x in range(size):
            y, x = y0 + step_y, x0 + step_x
            if y >= depth_y or x >= depth_x:
                continue
            for z in range(lowest_z - 1, -1, -1):
                if filled[z, y, x]:
                    break
                prop[z, y, x] = True
    return filled | prop, prop


def _widen_single_stud_joint(
    grid: NDArray[np.bool_], part: NDArray[np.bool_], other_layer: int, inside_box: NDArray[np.bool_]
) -> None:
    """If `part` (bool [y, x]) touches layer other_layer by exactly one stud, fill one cell of other_layer beside
    that stud under / over the part (in place; the first in (y, x) order)."""
    joint = part & grid[other_layer]
    if joint.sum() != 1:
        return
    ((joint_y, joint_x),) = np.argwhere(joint)
    _, depth_y, depth_x = grid.shape
    candidates = [
        (y, x)
        for y, x in _side_neighbours(joint_y, joint_x)
        if 0 <= y < depth_y and 0 <= x < depth_x and part[y, x] and not grid[other_layer, y, x] and inside_box[y, x]
    ]
    if candidates:
        y, x = sorted(candidates)[0]
        grid[other_layer, y, x] = True


def widen_joints(grid: NDArray[np.bool_], box: Box, max_part: int = 24) -> tuple[NDArray[np.bool_], int]:
    """Where a small part (a tail segment, at most `max_part` studs in its layer) rests on the layer below by a
    single stud, or carries the layer above by a single stud, add one cell beside that stud so the joint is two
    studs (a 2x2 part stepping diagonally touches the next by one corner only). Only inside `box`; only adds.
    Returns (new grid, number of cells added)."""
    filled = grid.astype(bool)
    widened = filled.copy()
    layer_count, depth_y, depth_x = filled.shape
    box_layers = range(box[0].start, min(box[0].stop, layer_count))
    inside_box = np.zeros((depth_y, depth_x), bool)
    inside_box[box[1], box[2]] = True
    for z in range(box_layers.stop - 2, box_layers.start - 1, -1):
        if z < 0 or z + 1 >= layer_count:
            continue
        for is_part_on_top in (True, False):
            part_layer, other_layer = (z + 1, z) if is_part_on_top else (z, z + 1)
            part_labels, part_count = label(widened[part_layer] & inside_box)
            for part_label in range(1, part_count + 1):
                part = part_labels == part_label
                if part.sum() <= max_part:
                    _widen_single_stud_joint(widened, part, other_layer, inside_box)
    return widened, int(widened.sum() - filled.sum())


def thicken_sheets(grid: NDArray[np.bool_], box: Box, direction: int = 1) -> tuple[NDArray[np.bool_], int]:
    """Cells inside `box` that are a one-plate-thick sheet (filled, nothing filled directly above or below:
    a hat brim, a cape edge) get one more plate above (direction +1) or below (-1). A 1-plate sheet has no stud
    contact with anything beside its own layer, so its outer ring floats; with 2 plates the pieces of the two
    layers overlap. Only adds. Returns (new grid, cells added)."""
    filled = grid.astype(bool)
    layer_count = filled.shape[0]
    filled_above = np.zeros_like(filled)
    filled_above[:-1] = filled[1:]
    filled_below = np.zeros_like(filled)
    filled_below[1:] = filled[:-1]
    sheet = filled & ~filled_above & ~filled_below
    inside_box = np.zeros_like(filled)
    inside_box[box] = True
    sheet &= inside_box
    thickened = filled.copy()
    zs, ys, xs = np.nonzero(sheet)
    for z, y, x in zip(zs, ys, xs, strict=True):
        new_z = z + direction
        if 0 <= new_z < layer_count:
            thickened[new_z, y, x] = True
    return thickened, int(thickened.sum() - filled.sum())


def _drop_boxes(grid: NDArray[Any], boxes: Sequence[Box]) -> NDArray[Any]:
    """--drop-box: remove the filled cells in each box (figure grid) and print one line per box."""
    if not boxes:
        return grid
    grid, removed_per_box = drop_cells.drop_cells_in_boxes(grid, boxes)
    for box, removed in zip(boxes, removed_per_box, strict=True):
        if removed:
            print(f"Drop box {drop_cells.format_box(box)}: {len(removed)} cell(s) removed (z, y, x): {removed}")
        else:
            print(f"  WARNING Drop box {drop_cells.format_box(box)}: no filled cell in the box, nothing removed")
    return grid


def fix_gaps_and_pinholes(
    grid: NDArray[Any], lego: NDArray[Any], shape: ShapeOptions
) -> tuple[NDArray[Any], NDArray[Any]]:
    """--open-gaps, then --fill-pinholes, --join-corners, --ground-overhangs, --drop-box, on the coloured grid:
    they run AFTER colouring because the automatic colour groups (k-means) shift when even a few cells change,
    which can drop a whole colour. Added cells take the colour of the nearest cell of the original grid; removed
    cells are cleared. lego = colour id [z, y, x]. Returns (grid, lego)."""
    pinhole_cells, corner_cells = shape.fill_pinholes_max_cells, shape.join_corners_max_cells
    overhang_plates, min_column = shape.ground_overhangs_max_plates, shape.ground_min_column_plates
    if not (shape.open_gaps or pinhole_cells or corner_cells or overhang_plates or shape.drop_boxes):
        return grid, lego
    before = grid.astype(bool).copy()
    if shape.open_gaps:  # before the pinholes: a thin sheet cell taken here is put back where the sheet continues
        grid, spurs = gap_spurs.open_gaps(grid)
        print(f"Open gaps: {len(spurs)} single-cell spur(s) removed (z, y, x): {spurs}")
    if pinhole_cells:
        grid, added_cells = pinholes.fill_pinholes(grid, pinhole_cells)
        print(
            f"Fill pinholes: +{added_cells} cells (see-through holes of <= {pinhole_cells} cells in thin parts closed)"
        )
    if corner_cells:  # after the gap spurs: a spur removed between two boots must stay open
        grid, joins = corner_joins.join_corners(grid, corner_cells)
        print(f"Join corners: +{len(joins)} cells (corner-only steps in parts of <= {corner_cells} cells closed)")
    column_fills: list[tuple[int, int, int, int]] = []
    if overhang_plates:  # last: the gaps under columns of the final shape
        grid, column_fills = ground_overhangs.ground_overhangs(grid, overhang_plates, min_column)
        print(
            f"Ground overhangs: +{len(column_fills)} cells (columns >= {min_column} plates tall that started "
            f"<= {overhang_plates} plate(s) in the air now reach down to the layer below)"
        )
    grid = _drop_boxes(grid, shape.drop_boxes)  # very last: the user's choice wins over every automatic fix
    lego = fill_from_nearest(lego, before & grid, grid & ~before)
    for z, y, x, column_z in column_fills:  # an extended column keeps the colour of the column above
        lego[z, y, x] = lego[column_z, y, x]
    lego = np.where(grid, lego, 0).astype(lego.dtype)
    return grid, lego
