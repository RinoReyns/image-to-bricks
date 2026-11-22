"""--carve-gap: separate a layer of one colour (a cape) from the body by a gap.

For capes, coats, skirts... that voxelize fused to the body. Between two height fractions (0 = bottom, 1 = top),
walking in from the side where the carve colour (ColourOptions.carve_colour_id) dominates (auto-detected), the
first such cell is kept as a CAPE_THICKNESS-stud layer and the next CAPE_GAP cell(s) are removed if the body
continues behind. Above the range the layer stays attached. Enable with --carve-gap FROM TO [--carve-color NAME].

Arrays: grid [z, y, x] bool (filled); lego [z, y, x] colour ids.
"""

import os
from typing import Any

import numpy as np
from numpy.typing import NDArray

from image_to_bricks.colours.colour_sampling import fill_from_nearest, save_color_debug
from image_to_bricks.colours.palette_matching import DEFAULT_COLOUR
from image_to_bricks.consts import COLOR_NAMES
from image_to_bricks.mesh.voxelize import surface_mask
from image_to_bricks.options import ColourOptions

CAPE_THICKNESS = 1  # studs of cape kept
CAPE_GAP = 1  # studs removed between cape and body
CAPE_MIN_BODY = 2  # body must continue this many studs behind the gap
Y_AXIS, X_AXIS = 1, 2  # horizontal axes of grid[z, y, x]
GAP_DEBUG_COLOUR_ID = 2  # Red: the removed gap in the debug picture...
GAP_DEBUG_COLOUR_ID_ON_RED_CAPE = 4  # ...Blue when the cape itself is Red

type Side = tuple[int, int]  # (horizontal axis, sign): the side of the grid the cape faces


def _back_direction(
    grid: NDArray[np.bool_], lego: NDArray[Any], z0: int, z1: int, carve_colour_id: int
) -> tuple[Side, float]:
    """The horizontal side (axis, sign) whose exposed faces in layers z0..z1 are most carve_colour_id, and that
    share. Raises ValueError if the layers have no filled cell."""
    best: Side | None = None
    best_share = -1.0
    layers = grid[z0:z1]
    for axis in (Y_AXIS, X_AXIS):
        for sign in (-1, 1):
            shifted = np.roll(layers, -sign, axis=axis)
            grid_edge = np.zeros_like(layers)
            edge_index: list[slice | int] = [slice(None)] * 3
            edge_index[axis] = -1 if sign == 1 else 0
            grid_edge[tuple(edge_index)] = True
            exposed = layers & (~shifted | grid_edge)
            if not exposed.any():
                continue
            share = float((lego[z0:z1][exposed] == carve_colour_id).mean())
            if share > best_share:
                best, best_share = (axis, sign), share
    if best is None:
        raise ValueError(f"--carve-gap: no filled cell in layers {z0}-{z1}, nothing to carve")
    return best, best_share


def _find_cape_gap(
    grid: NDArray[np.bool_], lego: NDArray[Any], layers: range, back: Side, cape_colour: int
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """(gap cells to remove, cape cells kept): on every line from the back side inward that starts with
    CAPE_THICKNESS cape-colour cells and continues for CAPE_GAP + CAPE_MIN_BODY filled cells."""
    axis, sign = back
    carved = np.zeros_like(grid)
    cape_kept = np.zeros_like(grid)
    run_length = CAPE_THICKNESS + CAPE_GAP + CAPE_MIN_BODY
    for z in layers:
        occupied, colours, carved_layer, kept_layer = grid[z], lego[z], carved[z], cape_kept[z]
        if axis == Y_AXIS:  # lines run along y: transpose (views) so every line is a row
            occupied, colours, carved_layer, kept_layer = occupied.T, colours.T, carved_layer.T, kept_layer.T
        for line in range(occupied.shape[0]):
            line_cells = occupied[line]
            from_back = np.arange(len(line_cells))[::-1] if sign > 0 else np.arange(len(line_cells))
            hits = np.nonzero(line_cells[from_back])[0]
            if len(hits) == 0:
                continue
            run = from_back[hits[0] : hits[0] + run_length]
            if len(run) < run_length or not line_cells[run].all() or colours[line][run[0]] != cape_colour:
                continue
            kept_layer[line, run[:CAPE_THICKNESS]] = True
            carved_layer[line, run[CAPE_THICKNESS : CAPE_THICKNESS + CAPE_GAP]] = True
    return carved, cape_kept


def carve_cape_gap(
    grid: NDArray[Any],
    lego: NDArray[Any],
    debug_dir: str | None = None,
    model_name: str = "model",
    colour: ColourOptions | None = None,
) -> tuple[NDArray[np.bool_], NDArray[Any]]:
    """--carve-gap: separate the layer of colour.carve_colour_id (a cape) from the body by a gap, between
    colour.carve_from_height and carve_to_height. Returns (grid, lego); with debug_dir also writes a picture."""
    colour = colour if colour is not None else DEFAULT_COLOUR
    cape_colour = colour.carve_colour_id
    grid = grid.astype(bool)  # a copy
    lego = lego.copy()
    layer_count = grid.shape[0]
    z0, z1 = int(layer_count * colour.carve_from_height), int(layer_count * colour.carve_to_height)
    (axis, sign), share = _back_direction(grid, lego, z0, z1, cape_colour)
    print(
        f"Cape carve: layers {z0}-{z1}, back side = axis {'yx'[axis - 1]} "
        f"{'+' if sign > 0 else '-'} ({100 * share:.0f}% {COLOR_NAMES[cape_colour]} faces)"
    )
    old_surface = surface_mask(grid)
    carved, cape_kept = _find_cape_gap(grid, lego, range(z0, z1), (axis, sign), cape_colour)
    grid &= ~carved
    lego[carved] = 0
    # Newly exposed body cells take the colour of the nearest original visible, non-cape cell (the leg's own
    # front / side colour).
    newly_exposed = surface_mask(grid) & ~old_surface
    colour_sources = old_surface & ~cape_kept & grid
    lego = np.where(newly_exposed, fill_from_nearest(lego, colour_sources, grid), lego)
    print(f"  carved {int(carved.sum())} voxels, recoloured {int(newly_exposed.sum())} exposed body voxels")
    if debug_dir:
        debug_ids = lego.copy()
        debug_ids[carved] = (
            GAP_DEBUG_COLOUR_ID if cape_colour != GAP_DEBUG_COLOUR_ID else GAP_DEBUG_COLOUR_ID_ON_RED_CAPE
        )
        save_color_debug(
            grid | carved,
            debug_ids,
            os.path.join(debug_dir, f"{model_name}_dbg5_cape_carve.png"),
            "5 cape carve (red = removed)",
            is_ids=True,
        )
    return grid, lego
