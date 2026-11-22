"""The stand under the figure as cells: outline, visible cells, ponds / paths, automatic margin, add_base.

Arrays are [z, y, x] (one layer: [y, x]); x, y in studs, z in plate layers; the base is the lowest base.plates layers.
"""

import contextlib
import io
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion

from image_to_bricks.analysis import stability
from image_to_bricks.consts import COLOR_NAMES
from image_to_bricks.model import LegoModel
from image_to_bricks.options import BaseOptions, BaseZone

# automatic margin (studs): the smallest margin in MIN..MAX that reaches the minimum tip angle
BASE_MARGIN_MIN = 2
BASE_MARGIN_MAX = 12

ROUNDED_SHAPE = "rounded"  # BaseOptions.shape with rounded corners ("rect" = plain rectangle)
SIDE_DIRECTIONS = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "up": (0.0, 1.0), "down": (0.0, -1.0)}  # (dx, dy)
# a cell and its 4 side neighbours: the base rim is what an erosion with this shape removes
_SIDE_NEIGHBOURS_SHAPE = np.array([[False, True, False], [True, True, True], [False, True, False]])

POND_CENTRE_FRACTION = 0.25  # pond centre: this fraction of the base size from the middle towards its side
POND_ASPECT = 1.3  # long axis / short axis of a pond (the long axis runs along its side)
POND_EDGE_WAVES = ((2, 0.16), (3, 0.10), (5, 0.06))  # (waves around the edge, relative radius change)
PATH_SIDEWAYS_WAVES = ((3.0, 1.2), (1.7, 0.6))  # (studs per radian along the path, sideways swing in studs)
PATH_WIDTH_WAVE = (2.3, 0.15)  # (studs per radian along the path, relative width change)
PATH_START_BEHIND_STUDS = 1  # the path starts this far behind the middle of the figure
PHASES_PER_ZONE = 4  # random wave phases drawn per zone; one is unused, but fewer would change every seed's shapes


@dataclass(frozen=True)
class _ZoneCanvas:
    """Where zones are drawn: stud centres, the free studs (no figure above) and the outline's box."""

    cell_x: NDArray[np.float64]  # [y, x] x of every stud centre
    cell_y: NDArray[np.float64]  # [y, x] y of every stud centre
    free_x: NDArray[np.intp]  # x of the free studs of the outline
    free_y: NDArray[np.intp]  # y of the free studs of the outline
    centre_x: float
    centre_y: float
    width: int  # studs of the outline's bounding box along x
    depth: int  # studs along y
    outline_studs: int  # studs in the outline


def _round_corners(outline: NDArray[np.bool_], box: tuple[int, int, int, int], radius_studs: int) -> None:
    """Cuts the corners of the rectangle box (y0, y1, x0, x1) in `outline` to quarter circles (in place)."""
    y0, y1, x0, x1 = box
    radius = max(0, min(radius_studs, (y1 - y0) // 2, (x1 - x0) // 2))
    if radius > 0:
        cell_y, cell_x = np.mgrid[0 : outline.shape[0], 0 : outline.shape[1]] + 0.5
        out_x = np.maximum(np.maximum(x0 + radius - cell_x, cell_x - (x1 - radius)), 0)
        out_y = np.maximum(np.maximum(y0 + radius - cell_y, cell_y - (y1 - radius)), 0)
        outline &= (out_x * out_x + out_y * out_y) <= radius * radius


def _base_outline(footprint: NDArray[np.bool_], shape: str, radius_studs: int, margin_studs: int) -> NDArray[np.bool_]:
    """Base outline for footprint bool [y, x] (already padded by the margin): its bounding box grown by
    margin_studs, rectangular or with rounded corners, plus the footprint itself so nothing of the lowest layers
    overhangs."""
    ys, xs = np.nonzero(footprint)
    y0, x0 = max(int(ys.min()) - margin_studs, 0), max(int(xs.min()) - margin_studs, 0)
    y1 = min(int(ys.max()) + 1 + margin_studs, footprint.shape[0])
    x1 = min(int(xs.max()) + 1 + margin_studs, footprint.shape[1])
    outline = np.zeros(footprint.shape, bool)
    outline[y0:y1, x0:x1] = True
    if shape == ROUNDED_SHAPE:
        _round_corners(outline, (y0, y1, x0, x1), radius_studs)
    return outline | footprint


def base_visibility(grid: NDArray[np.bool_], base_layers: int) -> NDArray[np.bool_]:
    """bool [base_layers, y, x]: base cells (the lowest base_layers layers of `grid`) seen from outside - the rim
    (side face) of every base layer, plus top-layer cells with no figure directly above. The rest is hidden:
    the inside of the lower layers and the cells under the figure."""
    bottom = grid[0]
    rim = bottom & ~binary_erosion(bottom, structure=_SIDE_NEIGHBOURS_SHAPE, border_value=0)
    visible = np.zeros((base_layers, *bottom.shape), bool)
    visible[:] = rim
    above = grid[base_layers] if grid.shape[0] > base_layers else np.zeros_like(bottom)
    visible[base_layers - 1] |= grid[base_layers - 1] & ~above
    return visible & grid[:base_layers]


def _pond_cells(
    canvas: _ZoneCanvas, side: tuple[float, float], percent: float, phases: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """bool [y, x]: an irregular ellipse covering about `percent` % of the outline, centred on the free stud
    nearest a point POND_CENTRE_FRACTION of the way from the middle towards `side`."""
    dx, dy = side
    target = np.array(
        [
            canvas.centre_x + dx * canvas.width * POND_CENTRE_FRACTION,
            canvas.centre_y + dy * canvas.depth * POND_CENTRE_FRACTION,
        ]
    )
    if len(canvas.free_x):
        free_x, free_y = canvas.free_x, canvas.free_y
        nearest = int(np.argmin((free_x + 0.5 - target[0]) ** 2 + (free_y + 0.5 - target[1]) ** 2))
        pond_x, pond_y = float(free_x[nearest] + 0.5), float(free_y[nearest] + 0.5)
    else:
        pond_x, pond_y = float(target[0]), float(target[1])
    radius = np.sqrt(percent / 100.0 * canvas.outline_studs / (np.pi * POND_ASPECT))
    axis_x, axis_y = radius * POND_ASPECT, radius
    if dx == 0:
        axis_x, axis_y = axis_y, axis_x
    relative_x, relative_y = (canvas.cell_x - pond_x) / axis_x, (canvas.cell_y - pond_y) / axis_y
    angle = np.arctan2(relative_y, relative_x)
    edge = np.ones_like(angle)
    for phase, (waves, amplitude) in zip(phases, POND_EDGE_WAVES, strict=False):
        edge = edge + amplitude * np.sin(waves * angle + phase)
    is_inside: NDArray[np.bool_] = np.hypot(relative_x, relative_y) <= edge
    return is_inside


def _path_cells(
    canvas: _ZoneCanvas, side: tuple[float, float], width_studs: float, phases: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """bool [y, x]: a wavy band about width_studs wide from the middle of the figure out to `side`."""
    dx, dy = side
    if len(canvas.free_x):
        start_x, start_y = float(canvas.free_x.mean() + 0.5), float(canvas.free_y.mean() + 0.5)
    else:
        start_x, start_y = canvas.centre_x, canvas.centre_y
    length = max(canvas.width, canvas.depth)
    along = (canvas.cell_x - start_x) * dx + (canvas.cell_y - start_y) * dy
    sideways = (canvas.cell_x - start_x) * -dy + (canvas.cell_y - start_y) * dx
    swing = np.zeros_like(along)
    for phase, (studs_per_radian, swing_studs) in zip(phases, PATH_SIDEWAYS_WAVES, strict=False):
        swing = swing + swing_studs * np.sin(along / studs_per_radian + phase)
    width_studs_per_radian, width_change = PATH_WIDTH_WAVE
    half_width = width_studs / 2.0 * (1 + width_change * np.sin(along / width_studs_per_radian + phases[2]))
    is_inside: NDArray[np.bool_] = (
        (along >= -PATH_START_BEHIND_STUDS) & (along <= length) & (np.abs(sideways - swing) <= half_width)
    )
    return is_inside


def _zone_canvas(outline: NDArray[np.bool_], free: NDArray[np.bool_]) -> _ZoneCanvas:
    """The canvas of the base outline [y, x]; free = studs with no figure above."""
    ys, xs = np.nonzero(outline)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    cell_y, cell_x = np.mgrid[0 : outline.shape[0], 0 : outline.shape[1]] + 0.5
    free_y, free_x = np.nonzero(free & outline)
    return _ZoneCanvas(
        cell_x, cell_y, free_x, free_y, (x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0, int(outline.sum())
    )


def base_zone_map(
    outline: NDArray[np.bool_], free: NDArray[np.bool_], zones: Sequence[BaseZone], seed: int = 1
) -> NDArray[np.uint8]:
    """Colour id (0 = none) for every stud of the base outline [y, x]: zones are painted in order, so a later one
    covers an earlier one. free = studs with no figure above (where a zone is seen).
    pond (side, percent): an irregular ellipse covering about `percent` % of the base, centred on the free stud
    nearest a point a quarter of the way in from that side. path (side, width): a wavy band `width` studs wide from
    the middle of the figure out to that side."""
    random_state = np.random.RandomState(seed)
    canvas = _zone_canvas(outline, free)
    zone_map = np.zeros(outline.shape, np.uint8)
    for colour_id, kind, side_name, size in zones:
        side = SIDE_DIRECTIONS[side_name]
        phases = random_state.uniform(0, 2 * np.pi, PHASES_PER_ZONE)
        draw_zone = _pond_cells if kind == "pond" else _path_cells
        zone_map[draw_zone(canvas, side, size, phases) & outline] = colour_id
    return zone_map


def choose_base_margin(grid: NDArray[np.bool_], base: BaseOptions) -> int:
    """The smallest margin in BASE_MARGIN_MIN..MAX whose base keeps the centre of mass of the (hollow) figure + base
    at least base.min_tip_angle_degrees from tipping (the maximum if none does); prints the choice."""
    margin, is_stable = BASE_MARGIN_MAX, False
    for margin in range(BASE_MARGIN_MIN, BASE_MARGIN_MAX + 1):
        with contextlib.redirect_stdout(io.StringIO()):
            trial_grid, _, _, _ = add_base(
                grid, np.zeros(grid.shape, np.uint8), np.zeros(grid.shape, bool), replace(base, margin_studs=margin)
            )
        trial = LegoModel(grid=trial_grid, bricks=[], base_layers=base.plates)
        is_stable = bool(stability.stability_check(trial, base.min_tip_angle_degrees).get("ok"))
        if is_stable:
            break
    print(
        f"Base margin (auto): {margin} studs"
        + (
            ""
            if is_stable
            else f" - WARNING: the tip angle is still below {base.min_tip_angle_degrees:.0f} deg at the maximum "
            f"margin; use a wider --base-from-layers or a taller base"
        )
    )
    return margin


def _paint_base_zones(
    colours: NDArray[Any], outline: NDArray[np.bool_], free: NDArray[np.bool_], base: BaseOptions
) -> None:
    """Paints base.zones on every base layer of colours [z, y, x] (in place) and prints the studs per colour."""
    zone_map = base_zone_map(outline, free, base.zones, base.zone_seed)
    painted = zone_map > 0
    for layer in range(base.plates):
        colours[layer][painted] = zone_map[painted]
    names = ", ".join(
        f"{COLOR_NAMES.get(i, i)} {int((zone_map == i).sum())} studs" for i in sorted(set(zone_map[painted].tolist()))
    )
    print(
        f"Base zones (seed {base.zone_seed}): {names}; "
        f"{base.colour_id and COLOR_NAMES.get(base.colour_id, base.colour_id)} "
        f"{int(((zone_map == 0) & outline).sum())} studs"
    )


def _pad_for_base(
    grid: NDArray[np.bool_], voxel_colours: NDArray[Any], visible: NDArray[np.bool_], base: BaseOptions
) -> tuple[NDArray[np.bool_], NDArray[Any], NDArray[np.bool_]]:
    """Copies of the three [z, y, x] arrays with base.plates empty layers below and base.margin_studs empty studs on
    every side."""
    layers, margin = base.plates, base.margin_studs
    depth_z, depth_y, depth_x = grid.shape
    padded_grid = np.zeros((depth_z + layers, depth_y + 2 * margin, depth_x + 2 * margin), bool)
    padded_colours = np.zeros(padded_grid.shape, voxel_colours.dtype)
    padded_visible = np.zeros(padded_grid.shape, bool)
    figure = (slice(layers, None), slice(margin, margin + depth_y), slice(margin, margin + depth_x))
    padded_grid[figure] = grid
    padded_colours[figure] = voxel_colours
    padded_visible[figure] = visible
    return padded_grid, padded_colours, padded_visible


def add_base(
    grid: NDArray[np.bool_], voxel_colours: NDArray[Any], visible: NDArray[np.bool_], base: BaseOptions
) -> tuple[NDArray[np.bool_], NDArray[Any], NDArray[np.bool_], NDArray[np.bool_]]:
    """Put base.plates layers of base.colour_id under the figure. The grid is padded by base.margin_studs on every
    side; the figure is shifted up by the base height. Returns (grid, colours, visible, base outline [y, x])."""
    layers = base.plates
    grid, colours, visible = _pad_for_base(grid, voxel_colours, visible, base)
    lowest_figure_layers = grid[layers : layers + base.footprint_layers] if base.footprint_layers > 0 else grid[layers:]
    footprint = np.asarray(lowest_figure_layers.any(axis=0), dtype=np.bool_)
    outline = _base_outline(footprint, base.shape, base.corner_radius_studs, base.margin_studs)
    grid[:layers][:, outline] = True
    colours[:layers][:, outline] = base.colour_id
    if base.zones:
        _paint_base_zones(colours, outline, outline & ~grid[layers], base)
    visible[:layers] = base_visibility(grid, layers)
    if base.layered is not None:
        visible[: base.layered.hidden_layers] = False  # hidden layers: any colour, also on the rim
    ys, xs = np.nonzero(outline)
    print(
        f"Base: {layers} plate(s), {base.shape}, {xs.max() - xs.min() + 1} x {ys.max() - ys.min() + 1} studs "
        f"({int(outline.sum())} studs), margin {base.margin_studs}, footprint of lowest "
        f"{base.footprint_layers or 'all'} layer(s) = {int(footprint.sum())} studs"
    )
    return grid, colours, visible, outline
