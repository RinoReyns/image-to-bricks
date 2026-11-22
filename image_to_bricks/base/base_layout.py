"""A base built by hand, used exactly as given (--base-layout FILE).

The CSV lists every plate of the base: layer (1 = bottom), x, y (studs from the corner of the base, the axis
numbers of the PDF base-layer maps: x to the right, y up = towards the back with --front down), width (along x),
depth (along y), color, and an optional shape note for a special piece (e.g. "technic plate 2x8 with 7 holes",
which counts as an owned special piece of the stock file). A layer may have windows (cells no plate covers).

The pipeline places the layout under the figure where the figure tips least (largest tip angle, nearest the
centre on a tie) with the whole first figure layer standing on top-layer plates, checks every seam of a layer
against the layer below (the same rules as --base-sparse-layers: a tie every BASE_TIE_EVERY studs, holding
BASE_TIE_MIN_STUDS studs on each side, and plates under the corners of the bottom layer), and puts the layout back
exactly after tie repair / clean-up / L pieces, so the instructions show the base as built.

Coordinates: layout pieces in layout studs; placed pieces (model.PlacedBrick) in the model grid; grid [z, y, x].
"""

import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from image_to_bricks.base import base_ties
from image_to_bricks.model import PlacedBrick
from image_to_bricks.stock import PLATE_HEIGHT_PLATES, PieceKey, Stock, make_key

LAYOUT_COLUMNS = ("layer", "x", "y", "width", "depth", "color")
NUMBER_COLUMNS = LAYOUT_COLUMNS[:5]  # the columns that hold whole numbers


@dataclass(frozen=True)
class LayoutPiece:
    """One hand-placed base plate (layout coordinates, layer 1 = bottom)."""

    layer: int
    x: int
    y: int
    width: int
    depth: int
    colour: int
    shape: str = ""


@dataclass(frozen=True)
class Placement:
    """Where the layout's corner lies in the model grid and how stable the result is."""

    grid_x: int
    grid_y: int
    tip_angle_degrees: float


def load_layout(path: str, colour_id: Callable[[str], int]) -> list[LayoutPiece]:
    """Reads the CSV; raises ValueError naming the row when a value is missing or two plates overlap."""
    pieces = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                layer, x, y, width, depth = [int(row[name]) for name in NUMBER_COLUMNS]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{path} row {row_number}: need whole numbers in {', '.join(NUMBER_COLUMNS)}"
                ) from error
            if layer < 1 or width < 1 or depth < 1 or x < 0 or y < 0:
                raise ValueError(f"{path} row {row_number}: layer, width, depth must be >= 1 and x, y >= 0")
            colour = colour_id(row.get("color") or "")
            pieces.append(LayoutPiece(layer, x, y, width, depth, colour, (row.get("shape") or "").strip()))
    if not pieces:
        raise ValueError(f"{path}: no plates")
    overlap = find_overlap(pieces)
    if overlap:
        raise ValueError(f"{path}: plates overlap in layer {overlap[0]} at x {overlap[1]}, y {overlap[2]}")
    return pieces


def layer_count(pieces: Sequence[LayoutPiece]) -> int:
    """Number of plate layers of the layout (its highest layer)."""
    return max(piece.layer for piece in pieces)


def layout_size(pieces: Sequence[LayoutPiece]) -> tuple[int, int]:
    """(width, depth) in studs of the layout's bounding box."""
    return max(piece.x + piece.width for piece in pieces), max(piece.y + piece.depth for piece in pieces)


def piece_ids_by_layer(pieces: Sequence[LayoutPiece]) -> NDArray[np.int32]:
    """int [layers, depth, width]: index of the piece on each cell (-1 = window)."""
    width, depth = layout_size(pieces)
    ids = np.full((layer_count(pieces), depth, width), -1, np.int32)
    for index, piece in enumerate(pieces):
        ids[piece.layer - 1, piece.y : piece.y + piece.depth, piece.x : piece.x + piece.width] = index
    return ids


def find_overlap(pieces: Sequence[LayoutPiece]) -> tuple[int, int, int] | None:
    """(layer, x, y) of the first cell covered by two plates, or None."""
    width, depth = layout_size(pieces)
    counts = np.zeros((layer_count(pieces), depth, width), np.int32)
    for piece in pieces:
        counts[piece.layer - 1, piece.y : piece.y + piece.depth, piece.x : piece.x + piece.width] += 1
    doubled = np.argwhere(counts > 1)
    return None if not len(doubled) else (int(doubled[0][0]) + 1, int(doubled[0][2]), int(doubled[0][1]))


def untied_seams(
    pieces: Sequence[LayoutPiece],
    tie_every: int = base_ties.BASE_TIE_EVERY,
    min_studs: int = base_ties.BASE_TIE_MIN_STUDS,
) -> list[str]:
    """Readable list of the seam chunks no plate of the layer below ties, and of bottom-layer corners with no
    plate (layout coordinates). Empty = every rule met."""
    ids = piece_ids_by_layer(pieces)
    studs_needed = {index: min(min_studs, piece.width * piece.depth) for index, piece in enumerate(pieces)}
    problems = []
    for upper_layer in range(ids.shape[0] - 1, 0, -1):
        problems += _untied_in_layer(ids[upper_layer], ids[upper_layer - 1], upper_layer + 1, studs_needed, tie_every)
    for y, x in base_ties.corner_cells(np.ones(ids.shape[1:], bool)):
        if ids[0, y, x] < 0:
            problems.append(f"bottom layer: no plate under the corner x {x}, y {y}")
    return problems


def _untied_in_layer(
    upper_ids: NDArray[np.int32],
    lower_ids: NDArray[np.int32],
    upper_layer: int,
    studs_needed: dict[int, int],
    tie_every: int,
) -> list[str]:
    """Readable list of the seam chunks of the upper layer (layer number upper_layer, 1 = bottom) that no plate
    of the lower layer ties; ids [y, x] = layout piece index (-1 = window), studs_needed = {piece: studs a tie
    must hold of it}."""
    requirements = base_ties.requirements(upper_ids, upper_ids >= 0, tie_every, with_feet=False)
    problems = []
    for requirement in requirements:
        if not any(
            _lower_piece_ties(requirement, upper_ids, lower_ids == lower, studs_needed)
            for lower in np.unique(lower_ids[lower_ids >= 0])
        ):
            (y, x), (other_y, other_x) = requirement.edges[0]
            problems.append(
                f"layer {upper_layer}: seam between the plates at x {x}, y {y} and x {other_x}, "
                f"y {other_y} ({len(requirement.edges)} studs) has no plate across it in layer "
                f"{upper_layer - 1}"
            )
    return problems


def _lower_piece_ties(
    requirement: base_ties.Requirement,
    upper_ids: NDArray[np.int32],
    lower_mask: NDArray[np.bool_],
    studs_needed: dict[int, int],
) -> bool:
    """True if the lower plate covering lower_mask [y, x] holds enough studs of both seam pieces and spans one of
    the seam's edges."""
    held: dict[int, int] = {}
    for upper in upper_ids[lower_mask]:
        held[int(upper)] = held.get(int(upper), 0) + 1
    a, b = requirement.first_piece, requirement.second_piece
    if held.get(a, 0) < studs_needed[a] or held.get(b, 0) < studs_needed[b]:
        return False
    return any(lower_mask[cell] and lower_mask[other_cell] for cell, other_cell in requirement.edges)


def place_layout(pieces: Sequence[LayoutPiece], grid_x: int, grid_y: int) -> list[PlacedBrick]:
    """The layout as model pieces with its corner at (grid_x, grid_y); layer 1 -> z 0."""
    return [
        PlacedBrick(grid_x + p.x, grid_y + p.y, p.layer - 1, p.width, p.depth, PLATE_HEIGHT_PLATES, p.colour)
        for p in pieces
    ]


def base_grid(grid: NDArray[np.bool_], placed: Sequence[PlacedBrick], base_layers: int) -> NDArray[np.bool_]:
    """Copy of grid whose base layers hold exactly the cells of the placed layout."""
    out = grid.copy()
    out[:base_layers] = False
    for piece in placed:
        out[piece.z, piece.y : piece.y + piece.d, piece.x : piece.x + piece.w] = True
    return out


def choose_placement(
    grid: NDArray[np.bool_],
    base_layers: int,
    pieces: Sequence[LayoutPiece],
    tip_angle: Callable[[NDArray[np.bool_]], float],
) -> Placement | None:
    """Position of the layout in the grid (all of it inside the grid, every cell of the first figure layer on a
    top-layer plate) with the largest tip_angle(grid with that base); ties: nearest the centre. None if the
    layout fits nowhere."""
    width, depth = layout_size(pieces)
    top = piece_ids_by_layer(pieces)[base_layers - 1] >= 0
    feet = grid[base_layers]
    feet_y, feet_x = np.nonzero(feet)
    best: tuple[tuple[float, int], Placement] | None = None
    for grid_y in range(grid.shape[1] - depth + 1):
        for grid_x in range(grid.shape[2] - width + 1):
            inside = (feet_x >= grid_x) & (feet_x < grid_x + width) & (feet_y >= grid_y) & (feet_y < grid_y + depth)
            if not inside.all() or not top[feet_y - grid_y, feet_x - grid_x].all():
                continue
            angle = tip_angle(base_grid(grid, place_layout(pieces, grid_x, grid_y), base_layers))
            off_centre = int(
                abs(2 * grid_x + width - feet_x.min() - feet_x.max() - 1)
                + abs(2 * grid_y + depth - feet_y.min() - feet_y.max() - 1)
            )
            key = (round(angle, 3), -off_centre)
            if best is None or key > best[0]:
                best = (key, Placement(grid_x, grid_y, angle))
    return None if best is None else best[1]


def register_special_pieces(stock: Stock, pieces: Sequence[LayoutPiece]) -> int:
    """Layout plates with a shape note that match a special stock row (same size and colour) count as owned:
    stock.have of that size and colour is raised by the matching specials used. Returns how many."""
    used: dict[PieceKey, int] = {}
    for piece in pieces:
        if piece.shape:
            key = make_key(PLATE_HEIGHT_PLATES, piece.width, piece.depth, piece.colour)
            used[key] = used.get(key, 0) + 1
    counted = 0
    for key, count in used.items():
        height, long_side, short_side, colour = key
        owned = sum(
            special_count
            for (special_height, width, depth, special_colour, _), special_count in stock.specials.items()
            if special_height == height
            and special_colour == colour
            and sorted((width, depth)) == sorted((long_side, short_side))
        )
        extra = min(owned, count)
        stock.have[key] = stock.have.get(key, 0) + extra
        counted += extra
    return counted


def reimpose(
    bricks: list[PlacedBrick], placed: Sequence[PlacedBrick], base_layers: int, stock: Stock | None = None
) -> list[PlacedBrick]:
    """bricks with every base piece replaced by the layout (repair steps may have re-tiled the base); the stock
    ledger follows."""
    kept = []
    for piece in bricks:
        if piece.z < base_layers:
            if stock is not None:
                stock.give_brick(piece)
        else:
            kept.append(piece)
    layout = [PlacedBrick(p.x, p.y, p.z, p.w, p.d, p.h_plates, p.color) for p in placed]
    if stock is not None:
        for piece in layout:
            stock.take_brick(piece)
    return layout + kept
