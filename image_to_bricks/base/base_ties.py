"""Sparse tie layer of a layered base (--base-sparse-layers): pieces only where they are needed.

As in the LEGO Architecture Sagrada Familia base: under a layer of plates, the next layer is not a full slab
but a few "ties" laid across the seams between those plates, plus feet at the corners so the base stands
flat and keeps its tip angle. A seam is the border between two neighbouring pieces of the layer above; it
is cut into chunks of `tie_every` studs and every chunk gets at least one tie (a piece that holds both
sides with at least `min_studs` studs each, or the whole piece if it is smaller).

Coordinates: one layer of the model grid, arrays [y, x]. Pieces are model.PlacedBrick.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
from numpy.typing import NDArray

from image_to_bricks.model import PlacedBrick
from image_to_bricks.stock import PLATE_HEIGHT_PLATES, Stock

BASE_TIE_EVERY = 4  # studs of seam per tie (--base-tie-every); smaller = stiffer, more pieces
BASE_TIE_MIN_STUDS = 2  # studs a tie must hold on each side of a seam (a smaller piece above: all of it)
SEAM, FOOT = "seam", "foot"  # Requirement.kind

# Largest tie (studs of area) per greedy pass; the pass with the fewest unmet requirements, then the fewest
# pieces, wins. Big ties meet many seams at once but can block the cells a remaining seam needs.
TIE_AREA_CAPS = (None, 16, 8, 4)

type Cell = tuple[int, int]  # (y, x)
type Edge = tuple[Cell, Cell]  # two neighbouring cells (y, x) on either side of a seam
type Rectangle = tuple[int, int, int, int]  # (x, y, width, depth) in studs


@dataclass(frozen=True)
class Requirement:
    """Something one tie piece must provide. kind SEAM: hold pieces first_piece and second_piece (indices of the
    layer above) across one of `edges`; kind FOOT: cover `cell` (a corner of the base outline, so the base stands
    on its corners)."""

    kind: str
    edges: tuple[Edge, ...] = ()
    first_piece: int = -1
    second_piece: int = -1
    cell: Cell = (-1, -1)


@dataclass(frozen=True)
class TieCandidate:
    """A plate rectangle (x, y, w, d in studs) and the requirements it would satisfy."""

    x: int
    y: int
    w: int
    d: int
    meets: frozenset[int]  # indices of the requirements this rectangle satisfies


@dataclass(frozen=True)
class TieSupply:
    """Where tie pieces come from: owned pieces from `stock` (a FORK, taken from; None = nothing owned) or bought in
    buy_sizes ((long, short) studs); bought ties get fallback_colour."""

    stock: Stock | None
    buy_sizes: frozenset[tuple[int, int]]
    fallback_colour: int


@dataclass(frozen=True)
class TieRules:
    """What a tie layer needs: plate sizes (w, d) it may use, studs of seam per tie, studs held on each side."""

    sizes: Sequence[tuple[int, int]]
    tie_every: int = BASE_TIE_EVERY
    min_studs: int = BASE_TIE_MIN_STUDS


type ChosenTie = tuple[TieCandidate, int | None]  # (candidate, owned colour or None = to buy)


def seam_edges(upper_ids: NDArray[Any], mask: NDArray[np.bool_]) -> dict[tuple[int, int], list[Edge]]:
    """{(a, b): [edges]} for every pair of different upper pieces a < b that touch over cells of `mask`."""
    seams: dict[tuple[int, int], list[Edge]] = {}
    depth, width = upper_ids.shape
    for y, x in zip(*np.nonzero(mask), strict=True):
        for neighbour_y, neighbour_x in ((y, x + 1), (y + 1, x)):
            if neighbour_y >= depth or neighbour_x >= width or not mask[neighbour_y, neighbour_x]:
                continue
            a, b = int(upper_ids[y, x]), int(upper_ids[neighbour_y, neighbour_x])
            if a >= 0 and b >= 0 and a != b:
                edge = ((int(y), int(x)), (int(neighbour_y), int(neighbour_x)))
                seams.setdefault((min(a, b), max(a, b)), []).append(edge)
    return seams


def _chunks(edges: list[Edge], tie_every: int) -> list[tuple[Edge, ...]]:
    """A seam's edges sorted along the seam, cut into runs of tie_every edges."""
    ordered = sorted(edges)
    return [tuple(ordered[i : i + tie_every]) for i in range(0, len(ordered), tie_every)]


def corner_cells(mask: NDArray[np.bool_]) -> list[Cell]:
    """The outline cell (y, x) nearest each corner of the outline's bounding box (4 cells, may repeat)."""
    ys, xs = np.nonzero(mask)
    corners = []
    for corner_y, corner_x in product((ys.min(), ys.max()), (xs.min(), xs.max())):
        nearest = int(np.argmin(np.abs(ys - corner_y) + np.abs(xs - corner_x)))
        corners.append((int(ys[nearest]), int(xs[nearest])))
    return corners


def requirements(
    upper_ids: NDArray[Any], mask: NDArray[np.bool_], tie_every: int, *, with_feet: bool
) -> list[Requirement]:
    """Seam chunks to tie (every tie_every studs of every seam) and, with_feet, the 4 corner feet."""
    found = [
        Requirement(SEAM, chunk, a, b)
        for (a, b), edges in sorted(seam_edges(upper_ids, mask).items())
        for chunk in _chunks(edges, tie_every)
    ]
    if with_feet:
        found += [Requirement(FOOT, cell=cell) for cell in dict.fromkeys(corner_cells(mask))]
    return found


def _meets(requirement: Requirement, rect: Rectangle, held: dict[int, int], need: dict[int, int]) -> bool:
    """True if the rectangle satisfies the requirement; held = {upper piece: studs under the rectangle},
    need = {upper piece: studs a tie must hold of it}."""
    x, y, w, d = rect

    def is_inside(cell: Cell) -> bool:
        return y <= cell[0] < y + d and x <= cell[1] < x + w

    if requirement.kind == FOOT:
        return is_inside(requirement.cell)
    first, second = requirement.first_piece, requirement.second_piece
    if held.get(first, 0) < need[first] or held.get(second, 0) < need[second]:
        return False
    return any(is_inside(cell) and is_inside(other_cell) for cell, other_cell in requirement.edges)


def _rectangles(mask: NDArray[np.bool_], sizes: Sequence[tuple[int, int]]) -> Iterator[Rectangle]:
    """(x, y, w, d) of every rectangle of `sizes` that lies inside `mask`."""
    depth, width = mask.shape
    for w, d in sizes:
        for y, x in product(range(depth - d + 1), range(width - w + 1)):
            if mask[y : y + d, x : x + w].all():
                yield x, y, w, d


def candidates(
    mask: NDArray[np.bool_],
    upper_ids: NDArray[Any],
    sizes: Sequence[tuple[int, int]],
    reqs: Sequence[Requirement],
    min_studs: int,
) -> tuple[list[TieCandidate], list[TieCandidate]]:
    """Rectangles of `sizes` inside `mask` that satisfy at least one requirement: (strict, relaxed). strict ties
    hold min_studs on each side of a seam (or the whole piece above if smaller); relaxed ones only 1 stud, the
    fallback for a seam the strict ones cannot reach (a short seam in a rounded corner, a cell another tie took)."""
    ids, counts = np.unique(upper_ids[upper_ids >= 0], return_counts=True)
    strict_need = {int(i): min(min_studs, int(c)) for i, c in zip(ids, counts, strict=True)}
    relaxed_need = {int(i): 1 for i in ids}
    strict: list[TieCandidate] = []
    relaxed: list[TieCandidate] = []
    for rect in _rectangles(mask, sizes):
        x, y, w, d = rect
        under = upper_ids[y : y + d, x : x + w]
        held_ids, held_counts = np.unique(under[under >= 0], return_counts=True)
        held = dict(zip(held_ids.tolist(), held_counts.tolist(), strict=True))
        for need, found in ((strict_need, strict), (relaxed_need, relaxed)):
            meets = frozenset(i for i, r in enumerate(reqs) if _meets(r, rect, held, need))
            if meets:
                found.append(TieCandidate(x, y, w, d, meets))
    return strict, relaxed


def _owned_colour(stock: Stock | None, w: int, d: int) -> int | None:
    """Colour of an unused owned plate of this size (None without stock or if none is left)."""
    return stock.any_owned_colour(PLATE_HEIGHT_PLATES, w, d) if stock is not None else None


def _greedy(
    free: NDArray[np.bool_], cands: Sequence[TieCandidate], open_reqs: set[int], supply: TieSupply
) -> list[ChosenTie]:
    """Add non-overlapping candidates to `free` / `open_reqs` (both changed in place), most open requirements
    met first, owned before bought, then the smaller piece. Returns [(candidate, owned colour or None)]."""
    chosen: list[ChosenTie] = []
    while open_reqs:
        best: ChosenTie | None = None
        best_key: tuple[int, bool, int] | None = None
        for candidate in cands:
            gain = len(candidate.meets & open_reqs)
            region = free[candidate.y : candidate.y + candidate.d, candidate.x : candidate.x + candidate.w]
            if not gain or not region.all():
                continue
            colour = _owned_colour(supply.stock, candidate.w, candidate.d)
            if (
                colour is None
                and (max(candidate.w, candidate.d), min(candidate.w, candidate.d)) not in supply.buy_sizes
            ):
                continue
            key = (gain, colour is not None, -candidate.w * candidate.d)
            if best_key is None or key > best_key:
                best, best_key = (candidate, colour), key
        if best is None:
            return chosen
        candidate, colour = best
        free[candidate.y : candidate.y + candidate.d, candidate.x : candidate.x + candidate.w] = False
        open_reqs -= candidate.meets
        if supply.stock is not None and colour is not None:
            supply.stock.take(PLATE_HEIGHT_PLATES, candidate.w, candidate.d, colour)
        chosen.append(best)
    return chosen


def _cover(
    mask: NDArray[np.bool_], chosen: Sequence[ChosenTie], requirement_count: int
) -> tuple[NDArray[np.bool_], set[int]]:
    """(free cells, open requirements) left by the chosen candidates."""
    free, open_reqs = mask.copy(), set(range(requirement_count))
    for candidate, _ in chosen:
        free[candidate.y : candidate.y + candidate.d, candidate.x : candidate.x + candidate.w] = False
        open_reqs -= candidate.meets
    return free, open_reqs


def _supply_without(supply: TieSupply, removed: ChosenTie) -> TieSupply:
    """A supply on a fork of the stock with the removed tie's owned piece given back."""
    if supply.stock is None:
        return supply
    fork = supply.stock.fork()
    fork.used = fork.used.copy()
    candidate, colour = removed
    if colour is not None:
        fork.give(PLATE_HEIGHT_PLATES, candidate.w, candidate.d, colour)
    return TieSupply(fork, supply.buy_sizes, supply.fallback_colour)


@dataclass
class _TieSearch:
    """The requirements to cover and the candidates to cover them with (for the repair pass)."""

    mask: NDArray[np.bool_]
    cands: Sequence[TieCandidate]
    requirement_count: int
    supply: TieSupply
    chosen: list[ChosenTie] = field(default_factory=list)

    def improve_once(self, open_now: set[int]) -> bool:
        """Try removing each chosen tie and re-covering what is open (the stuck seams first); keep the first change
        that leaves fewer requirements open (then fewer pieces). True if one was kept."""
        for k in range(len(self.chosen)):
            trial = self.chosen[:k] + self.chosen[k + 1 :]
            trial_supply = _supply_without(self.supply, self.chosen[k])
            free, open_reqs = _cover(self.mask, trial, self.requirement_count)
            blocked_first = [c for c in self.cands if c.meets & open_now]  # the seams that were stuck go first
            trial += _greedy(free, blocked_first, open_reqs, trial_supply)
            trial += _greedy(free, self.cands, open_reqs, trial_supply)
            if (len(open_reqs), len(trial)) < (len(open_now), len(self.chosen)):
                self.chosen = trial
                if self.supply.stock is not None and trial_supply.stock is not None:
                    self.supply.stock.used = trial_supply.stock.used
                return True
        return False


def _repair(search: _TieSearch) -> list[ChosenTie]:
    """Greedy blocks itself when a big tie takes a cell a later seam needs. For each piece, try removing it and
    re-covering what is open; keep the change if fewer requirements stay open (then fewer pieces)."""
    while True:
        _, open_now = _cover(search.mask, search.chosen, search.requirement_count)
        if not open_now or not search.improve_once(open_now):
            return search.chosen


def choose_ties(
    mask: NDArray[np.bool_],
    strict: Sequence[TieCandidate],
    relaxed: Sequence[TieCandidate],
    requirement_count: int,
    supply: TieSupply,
) -> tuple[list[ChosenTie], list[int]]:
    """Cover the requirements with non-overlapping candidates: strict greedy, a repair pass (see `_repair`), then
    relaxed (1-stud) ties for what is still open. Owned pieces are taken from supply.stock (a FORK).
    Returns ([(candidate, owned colour or None = to buy)], open requirement indices that no piece could meet)."""
    free, open_reqs = mask.copy(), set(range(requirement_count))
    chosen = _greedy(free, strict, open_reqs, supply)
    chosen = _repair(_TieSearch(mask, strict, requirement_count, supply, chosen))
    free, open_reqs = _cover(mask, chosen, requirement_count)
    chosen += _greedy(free, relaxed, open_reqs, supply)
    return chosen, sorted(open_reqs)


def _place_ties(chosen: Sequence[ChosenTie], supply: TieSupply, z: int) -> list[PlacedBrick]:
    """Plates of the chosen ties in layer z; bought ones get the fallback colour (taken from the stock too)."""
    pieces = []
    for candidate, owned_colour in chosen:
        colour = owned_colour
        if colour is None:
            colour = supply.fallback_colour
            if supply.stock is not None:
                supply.stock.take(PLATE_HEIGHT_PLATES, candidate.w, candidate.d, colour)
        pieces.append(
            PlacedBrick(
                candidate.x, candidate.y, z, candidate.w, candidate.d, h_plates=PLATE_HEIGHT_PLATES, color=colour
            )
        )
    return pieces


def pack_tie_layer(
    mask: NDArray[np.bool_], upper_ids: NDArray[Any], z: int, supply: TieSupply, rules: TieRules
) -> tuple[list[PlacedBrick], int, list[Requirement]]:
    """Sparse layer z under the pieces `upper_ids` [y, x] (-1 = none): ties across every seam chunk, plus corner
    feet when z == 0 (the layer the base stands on). supply.stock: a stock.Stock FORK (taken from) or None.
    Returns (pieces, number of requirements, open requirements)."""
    reqs = requirements(upper_ids, mask, rules.tie_every, with_feet=z == 0)
    strict, relaxed = candidates(mask, upper_ids, rules.sizes, reqs, rules.min_studs)
    best: tuple[tuple[int, int, int], list[PlacedBrick], list[int], Stock | None] | None = None
    for cap in TIE_AREA_CAPS:
        fork = supply.stock.fork() if supply.stock is not None else None
        pass_supply = TieSupply(fork, supply.buy_sizes, supply.fallback_colour)
        allowed = [c for c in strict if cap is None or c.w * c.d <= cap]
        chosen, open_reqs = choose_ties(mask, allowed, relaxed, len(reqs), pass_supply)
        pieces = _place_ties(chosen, pass_supply, z)
        bought = fork.need_total() if fork is not None else 0
        key = (len(open_reqs), len(pieces), bought)
        if best is None or key < best[0]:
            best = (key, pieces, open_reqs, fork)
    if best is None:
        raise ValueError("TIE_AREA_CAPS is empty")
    _, pieces, open_reqs, fork = best
    if supply.stock is not None and fork is not None:
        supply.stock.used = fork.used
    return pieces, len(reqs), [reqs[i] for i in open_reqs]
