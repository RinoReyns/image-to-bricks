"""Owned L pieces (--use-l-pieces): corner plate / brick 2x2 with 3 studs, placed after packing and tie repair.

The packer and the tie repair only know rectangles. This last pass looks for places where an L piece can take
over EXACTLY the cells of 2 or more existing pieces (the L's 3 cells are covered by pieces that lie entirely
inside it) and swaps them for one owned L piece. Cells never change; only the pieces covering them.

  * plate L (1 plate high): the L cells in one layer are covered by 2-3 small pieces (1x2 + 1x1, 3 x 1x1);
  * brick L (3 plates high): the same L cells in 3 layers z..z+2 are covered by pieces that lie inside the
    3-layer block (e.g. a 1x2 brick + 3 plates, 9 small plates, ...).
  * colour: pieces with a visible cell fix the colour (all must agree, and an owned L of that colour must be
    left); an L over hidden cells only takes any owned colour (the colour with most left).
  * --l-skip-box: pieces touching a box (e.g. the eyes) are never swapped.
  * the log also prints the DEMAND: how many places would fit an L per colour / height if you owned one
    (upper bound, stability not checked), to see which L pieces are worth buying.
  * a swap is kept only if the stability score (loose pieces, big single-stud joints, single-stud footings)
    does not get worse, and the stock ledger follows every swap (L pieces are never bought).

An L piece is a PlacedBrick with w = d = 2 and `notch` = the empty corner of the 2x2 box (see model.piece_mask).
"""

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from image_to_bricks.analysis import stability
from image_to_bricks.consts import COLOR_NAMES
from image_to_bricks.model import LegoModel, PlacedBrick, owner_grid, piece_mask
from image_to_bricks.packing.retile_setting import is_not_worse
from image_to_bricks.stock import Stock

PLATE_HEIGHT = 1  # plates high: a corner plate
BRICK_HEIGHT = 3  # plates high: a corner brick
L_BOX = 2  # an L piece fits a 2x2 box (3 of its 4 cells)
NOTCHES = range(4)  # which corner of the box is empty (see PlacedBrick.notch)
MAX_MEMBER_CELLS = 2  # a piece with more cells than this can never lie inside an L (3 cells) together with another
MIN_PIECES_PER_SWAP = 2  # an L replaces at least this many pieces

type DemandKey = tuple[int, int | None]  # (height in plates, colour or None for hidden-only places)


class LPlacement(NamedTuple):
    """Where an L piece would go: its 2x2 box at (x0, y0) in layers z..z+height-1, corner `notch` empty."""

    z: int
    height: int  # plates (1 or 3)
    x0: int
    y0: int
    notch: int


class Box(NamedTuple):
    """Region of the grid (z, y, x), ends exclusive."""

    z0: int
    z1: int
    y0: int
    y1: int
    x0: int
    x1: int

    def touches(self, b: PlacedBrick) -> bool:
        """True if piece b overlaps the box in z, y and x."""
        return (
            b.z < self.z1
            and b.z + b.h_plates > self.z0
            and b.y < self.y1
            and b.y + b.d > self.y0
            and b.x < self.x1
            and b.x + b.w > self.x0
        )


@dataclass(frozen=True)
class LOptions:
    """Settings of the pass. skip: boxes in grid coordinates (base and margin included)."""

    base_layers: int = 0
    hang_min: int = 5  # single-stud joints hanging at least this many pieces count as weak (like --tie-min-hang)
    skip: tuple[Box, ...] = ()
    verbose: bool = True


@dataclass(frozen=True)
class Candidate:
    """One possible swap. colour: colour of the L (None = only hidden cells, decided at swap time);
    visible_colour: the colour fixed by visible cells (None = all cells hidden)."""

    gain: int  # pieces saved
    z: int
    h: int  # height in plates (1 or 3)
    x: int
    y: int
    notch: int
    ids: tuple[int, ...]  # indices of the pieces replaced
    colour: int | None
    visible_colour: int | None

    def sort_key(self) -> tuple[int, bool, int, int, int, int]:
        """Best first: most pieces saved, visible places first, then lowest layer, y, x, notch."""
        return (-self.gain, self.visible_colour is None, self.z, self.y, self.x, self.notch)


@dataclass(frozen=True)
class _Scene:
    """What a candidate check looks at: pieces, who owns each cell (own[z, y, x] = piece index or -1),
    visibility (bool [z, y, x] or None = everything visible), stock (None = pretend every L colour is owned)."""

    bricks: Sequence[PlacedBrick]
    own: NDArray[np.int32]
    visible: NDArray[np.bool_] | None
    stock: Stock | None
    skip: tuple[Box, ...]


# --- stability score ---------------------------------------------------------------------------------------


def _count_big_weak_joints(
    adj: stability.Contacts,
    comp: list[int],
    sizes: dict[int, int],
    bridges: list[stability.Bridge],
    hang_min: int,
) -> int:
    """Single-stud joints that hang at least hang_min pieces on one side."""
    count = 0
    for p, u, hang_u in bridges:
        if adj[p][u] > stability.WEAK_STUDS:
            continue
        total = sizes[comp[p]]
        if min(hang_u, total - hang_u) >= hang_min:
            count += 1
    return count


def _count_single_stud_footings(bricks: Sequence[PlacedBrick], adj: stability.Contacts, base_layers: int) -> int:
    """Pieces of the first figure layer that touch the base with exactly one stud."""
    if not base_layers:
        return 0
    return sum(
        1
        for i, b in enumerate(bricks)
        if b.z == base_layers and sum(w for j, w in adj[i].items() if bricks[j].z < base_layers) == 1
    )


def stability_score(
    bricks: Sequence[PlacedBrick], grid: NDArray[np.bool_], base_layers: int, hang_min: int
) -> tuple[int, int, int]:
    """(loose pieces, single-stud joints hanging >= hang_min pieces, single-stud footings); lower is better."""
    adj = stability.build_contacts(LegoModel(grid=grid, bricks=list(bricks), base_layers=base_layers))
    comp, sizes, bridges = stability.analyse_graph(adj)
    main = max(sizes, key=lambda component: sizes[component])
    loose = sum(n for c, n in sizes.items() if c != main)
    return (
        loose,
        _count_big_weak_joints(adj, comp, sizes, bridges, hang_min),
        _count_single_stud_footings(bricks, adj, base_layers),
    )


# --- finding candidates ------------------------------------------------------------------------------------


def _block_starts(piece: PlacedBrick, height: int) -> list[int]:
    """z of every block of `height` plates that can contain the piece."""
    if height == PLATE_HEIGHT:
        return [piece.z]
    return list(range(max(0, piece.z + piece.h_plates - BRICK_HEIGHT), piece.z + 1))


def _candidate_placements(piece: PlacedBrick, shape: tuple[int, int, int]) -> Iterator[LPlacement]:
    """Every L placement whose 2x2 box contains the piece and fits the grid."""
    nz, ny, nx = shape
    heights = (PLATE_HEIGHT, BRICK_HEIGHT) if piece.h_plates == PLATE_HEIGHT else (BRICK_HEIGHT,)
    for h in heights:
        for z in _block_starts(piece, h):
            if z + h > nz:
                continue
            for x0 in range(piece.x - 1, piece.x + 1):
                for y0 in range(piece.y - 1, piece.y + 1):
                    if x0 >= 0 and y0 >= 0 and x0 + L_BOX <= nx and y0 + L_BOX <= ny:
                        for notch in NOTCHES:
                            yield LPlacement(z, h, x0, y0, notch)


def _l_cells(x0: int, y0: int, notch: int) -> set[tuple[int, int]]:
    """(x, y) of the 3 studs of an L in the box at (x0, y0) with the corner `notch` empty."""
    return {(x0 + (j & 1), y0 + (j >> 1)) for j in NOTCHES if j != notch}


def _covering_ids(scene: _Scene, z: int, h: int, cells: set[tuple[int, int]]) -> set[int] | None:
    """Indices of the pieces covering the cells in layers z..z+h-1; None if a cell is empty."""
    ids: set[int] = set()
    for layer in range(z, z + h):
        for x, y in cells:
            i = int(scene.own[layer, y, x])
            if i < 0:
                return None
            ids.add(i)
    return ids


def _lies_inside(b: PlacedBrick, z: int, h: int, cells: set[tuple[int, int]]) -> bool:
    """True if the (rectangular) piece is within layers z..z+h-1 and its studs are all among `cells`."""
    if b.notch >= 0 or b.z < z or b.z + b.h_plates > z + h:
        return False
    return all((b.x + dx, b.y + dy) in cells for dy in range(b.d) for dx in range(b.w))


def _has_visible_cell(scene: _Scene, b: PlacedBrick) -> bool:
    if scene.visible is None:
        return True
    return bool(scene.visible[b.z : b.z + b.h_plates, b.y : b.y + b.d, b.x : b.x + b.w].any())


def _choose_colour(stock: Stock | None, h: int, visible_colours: set[int]) -> tuple[int | None, int | None] | None:
    """(colour of the L, colour fixed by visible cells) or None if no owned L fits. stock None = every colour is owned
    (hidden-only places then get colour None: any colour will do)."""
    if visible_colours:
        colour = next(iter(visible_colours))
        if stock is not None and stock.l_free(h, colour) <= 0:
            return None
        return colour, colour
    if stock is None:
        return None, None
    any_colour = stock.l_any_colour(h)
    return None if any_colour is None else (any_colour, None)


def _candidate_at(scene: _Scene, placement: LPlacement) -> Candidate | None:
    """The swap for this L placement, or None if it is not possible."""
    z, h, x0, y0, notch = placement
    cells = _l_cells(x0, y0, notch)
    ids = _covering_ids(scene, z, h, cells)
    if ids is None or len(ids) < MIN_PIECES_PER_SWAP:
        return None  # an empty cell, or nothing to merge
    visible_colours: set[int] = set()
    for i in ids:
        b = scene.bricks[i]
        if any(box.touches(b) for box in scene.skip) or not _lies_inside(b, z, h, cells):
            return None  # skipped, an L already, or the piece sticks out of the L
        if _has_visible_cell(scene, b):
            visible_colours.add(b.color)
    if len(visible_colours) > 1:
        return None  # two visible colours: one L cannot be both
    choice = _choose_colour(scene.stock, h, visible_colours)
    if choice is None:
        return None
    colour, visible_colour = choice
    return Candidate(len(ids) - 1, z, h, x0, y0, notch, tuple(sorted(ids)), colour, visible_colour)


def find_candidates(
    bricks: Sequence[PlacedBrick],
    grid: NDArray[np.bool_],
    visible: NDArray[np.bool_] | None,
    stock: Stock | None,
    skip: Sequence[Box] = (),
) -> list[Candidate]:
    """All L swaps possible now (none applied yet), best first; they may overlap each other.
    stock None = pretend every L colour is owned (used for the demand report)."""
    depth_z, depth_y, depth_x = grid.shape
    grid_shape = (depth_z, depth_y, depth_x)
    scene = _Scene(bricks, owner_grid(bricks, grid_shape), visible, stock, tuple(skip))
    seen: set[LPlacement] = set()
    found: list[Candidate] = []
    for piece in bricks:
        if piece.notch >= 0 or piece.w * piece.d > MAX_MEMBER_CELLS:
            continue  # does not fit inside an L together with another piece
        for placement in _candidate_placements(piece, grid_shape):
            if placement in seen:
                continue
            seen.add(placement)
            candidate = _candidate_at(scene, placement)
            if candidate is not None:
                found.append(candidate)
    found.sort(key=Candidate.sort_key)
    return found


# --- demand report -----------------------------------------------------------------------------------------


def demand(
    bricks: Sequence[PlacedBrick], grid: NDArray[np.bool_], visible: NDArray[np.bool_] | None, skip: Sequence[Box] = ()
) -> tuple[Counter[DemandKey], int]:
    """Places that would take an L piece if every colour were owned, non-overlapping, best first.
    Returns (Counter {(height, colour or None for hidden-only): places}, pieces that would be saved)."""
    taken: set[int] = set()
    places: Counter[DemandKey] = Counter()
    saved = 0
    for c in find_candidates(bricks, grid, visible, None, skip):
        if taken.intersection(c.ids):
            continue
        taken.update(c.ids)
        places[(c.h, c.visible_colour)] += 1
        saved += c.gain
    return places, saved


def _owned_of_height(stock: Stock, h: int) -> int:
    return sum(n for (height, _), n in stock.l_have.items() if height == h)


def _demand_row(colour: int | None, by_height: dict[int, int], stock: Stock) -> str:
    if colour is None:
        name = "hidden (any colour)"
        own_plates, own_bricks = _owned_of_height(stock, PLATE_HEIGHT), _owned_of_height(stock, BRICK_HEIGHT)
    else:
        name = COLOR_NAMES.get(colour, f"Color{colour}")
        own_plates, own_bricks = (
            stock.l_have.get((PLATE_HEIGHT, colour), 0),
            stock.l_have.get((BRICK_HEIGHT, colour), 0),
        )
    plates, bricks = by_height[PLATE_HEIGHT], by_height[BRICK_HEIGHT]
    extra = max(0, plates - own_plates) + max(0, bricks - own_bricks)
    tail = f"any colour: {extra}" if colour is None else str(extra)
    return f"  {name:22s} {plates:7d} {own_plates:6d} {bricks:7d} {own_bricks:6d}   {tail}"


def print_demand(places: Counter[DemandKey], saved: int, stock: Stock) -> None:
    """Console table of the demand: places per colour and height against the L pieces owned."""
    if not places:
        print("L demand: no place where an L piece would fit")
        return
    by_colour: dict[int | None, dict[int, int]] = {}
    for (h, colour), n in places.items():
        by_colour.setdefault(colour, {PLATE_HEIGHT: 0, BRICK_HEIGHT: 0})[h] += n
    print(
        f"L demand: places where an L would fit (upper bound, stability not checked), {sum(places.values())} in total, "
        f"up to {saved} pieces saved"
    )
    print(
        f"  {'colour':22s} {'plates':>7s} {'owned':>6s} {'bricks':>7s} {'owned':>6s}   "
        "extra L pieces that would be used"
    )
    order = sorted(by_colour, key=lambda c: (c is None, -sum(by_colour[c].values())))
    for colour in order:
        print(_demand_row(colour, by_colour[colour], stock))


# --- the swap ----------------------------------------------------------------------------------------------


def _colour_still_free(stock: Stock, c: Candidate) -> int | None:
    """Colour to build the candidate with now (an earlier swap may have used the owned L of its colour up):
    the planned colour, another owned colour for hidden-only places, or None."""
    if c.colour is not None and stock.l_free(c.h, c.colour) > 0:
        return c.colour
    if c.visible_colour is not None:
        return None
    return stock.l_any_colour(c.h)


def _visible_l(b: PlacedBrick, visible: NDArray[np.bool_] | None) -> bool:
    if visible is None:
        return True
    region = visible[b.z : b.z + b.h_plates, b.y : b.y + b.d, b.x : b.x + b.w]
    return bool(region[:, piece_mask(b)].any())


def _log_result(
    placed: list[tuple[PlacedBrick, int]], visible: NDArray[np.bool_] | None, before: int, after: int
) -> None:
    n_visible = sum(1 for b, _ in placed if _visible_l(b, visible))
    print(
        f"L pieces: {len(placed)} placed ({n_visible} on visible cells, {len(placed) - n_visible} hidden), "
        f"pieces {before} -> {after}"
    )
    for b, n_replaced in placed:
        print(
            f"  L {'brick' if b.h_plates == BRICK_HEIGHT else 'plate'} at layer {b.z + 1}, x {b.x}, y {b.y}, "
            f"replaces {n_replaced} piece(s), colour {b.color}"
        )


def use_l_pieces(
    bricks: Sequence[PlacedBrick],
    grid: NDArray[np.bool_],
    visible: NDArray[np.bool_] | None,
    stock: Stock | None,
    options: LOptions | None = None,
) -> tuple[list[PlacedBrick], int]:
    """Swap groups of small pieces for owned L pieces. Returns (new brick list, number of L pieces placed).
    `stock` is the running ledger and IS UPDATED (pieces given back, L pieces taken); stock None = nothing happens.
    options None = LOptions() defaults."""
    options = options if options is not None else LOptions()
    if stock is None:
        return list(bricks), 0
    if options.verbose:
        places, saved = demand(bricks, grid, visible, options.skip)
        print_demand(places, saved, stock)
    if not stock.l_have:
        if options.verbose:
            print("L pieces: none in the stock (add 'corner plate / brick 2x2 (L, 3 studs)' rows), skipped")
        return list(bricks), 0
    candidates = find_candidates(bricks, grid, visible, stock, options.skip)
    if options.verbose:
        print(f"L pieces: {sum(stock.l_have.values())} owned, {len(candidates)} possible place(s) found")
    slots: list[PlacedBrick | None] = list(bricks)  # indices stay valid; swapped pieces become None
    placed = _swap_all(slots, candidates, grid, stock, options)
    result = [b for b in slots if b is not None]
    if options.verbose:
        _log_result(placed, visible, len(bricks), len(result))
    return result, len(placed)


def _swap_all(
    slots: list[PlacedBrick | None],
    candidates: Sequence[Candidate],
    grid: NDArray[np.bool_],
    stock: Stock,
    options: LOptions,
) -> list[tuple[PlacedBrick, int]]:
    """Apply the candidates best first (in place on `slots` and the ledger); returns [(L piece, pieces replaced)]."""
    score = stability_score([b for b in slots if b is not None], grid, options.base_layers, options.hang_min)
    swapped: set[int] = set()
    placed: list[tuple[PlacedBrick, int]] = []
    for c in candidates:
        if swapped.intersection(c.ids):
            continue
        colour = _colour_still_free(stock, c)
        if colour is None:
            continue
        new = PlacedBrick(c.x, c.y, c.z, L_BOX, L_BOX, c.h, colour, c.notch)
        trial = [b for i, b in enumerate(slots) if b is not None and i not in c.ids] + [new]
        new_score = stability_score(trial, grid, options.base_layers, options.hang_min)
        if not is_not_worse(new_score, score):
            continue  # would make the build less stable
        for i in c.ids:
            replaced = slots[i]
            if replaced is None:
                raise ValueError(f"piece {i} was already swapped for an L piece")
            stock.give_brick(replaced)
            slots[i] = None
        slots.append(new)
        stock.take_brick(new)
        swapped.update(c.ids)
        score = new_score
        placed.append((new, len(c.ids)))
    return placed
