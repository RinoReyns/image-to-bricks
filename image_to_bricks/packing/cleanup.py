"""Lossless clean-up of small pieces (--cleanup-small).

The greedy packer leaves many 1x1 pieces next to pieces of the same colour. This pass re-tiles a small
group of same-colour pieces (same layer, same height, touching the 1x1) with an exact search that
minimises 1x1 pieces, then the piece count. Nothing visible changes: same cells, same colours.

A change is kept only if ALL of these hold (else it is rolled back):
  * fewer 1x1 pieces (or equal and fewer pieces) in the group,
  * no more "aligned seams" (a joint between two pieces lying exactly above a joint of the layer below) and
    no more exact stacks (a piece on an identical piece) around the group - the wall is not weaker,
  * the tie-repair score (loose pieces, weak joints, footings) is not worse,
  * with --stock: no owned piece goes unused and nothing more has to be bought.
"""

from collections.abc import Iterable, Sequence
from typing import NamedTuple

import numpy as np

from image_to_bricks.model import PlacedBrick
from image_to_bricks.packing.assembly import Assembly, piece_sizes, score

MAX_GROUP_CELLS = 30  # max cells in a re-tiled group
NEIGHBOUR_RINGS = 2  # how many neighbour rings are added around the 1x1
MIN_GROUP_PIECES = 2  # a group must hold the 1x1 and at least one neighbour


class TileVariant(NamedTuple):
    """Which corner the exact tiling search starts from (different variants give different tilings)."""

    flips_x: bool
    flips_y: bool
    swaps_axes: bool


TILE_VARIANTS = tuple(
    TileVariant(bool(flips_x), bool(flips_y), bool(swaps_axes))
    for swaps_axes in (0, 1)
    for flips_y in (0, 1)
    for flips_x in (0, 1)
)

type Cell = tuple[int, int]  # (x, y)
type Footprint = tuple[int, int, int, int]  # (x, y, width, depth)
type TilingScore = tuple[int, int]  # (1x1 pieces, pieces): smaller is better
type Solution = tuple[TilingScore, tuple[Footprint, ...]]


class Window(NamedTuple):
    """Rows y0..y1-1 and columns x0..x1-1 of the grid."""

    y0: int
    y1: int
    x0: int
    x1: int


def _count_single_studs(footprints: Iterable[Footprint]) -> int:
    """Number of 1x1 footprints."""
    return sum(1 for _, _, w, d in footprints if w == 1 and d == 1)


def _to_search(cell: Cell, variant: TileVariant) -> Cell:
    """Cell (x, y) in the search coordinates of the variant (axes swapped first, then flipped)."""
    x, y = cell
    if variant.swaps_axes:
        x, y = y, x
    return (-x if variant.flips_x else x, -y if variant.flips_y else y)


def _from_search(cell: Cell, variant: TileVariant) -> Cell:
    """Inverse of _to_search."""
    x, y = cell
    x, y = (-x if variant.flips_x else x), (-y if variant.flips_y else y)
    return (y, x) if variant.swaps_axes else (x, y)


class _TilingSearch:
    """Memoised exact tiling: the remaining cells' lowest-row, lowest-column cell is covered by every fitting size."""

    def __init__(self, sizes: Sequence[tuple[int, int]]) -> None:
        self.sizes = sizes
        self.memo: dict[frozenset[Cell], Solution | None] = {}

    def solve(self, remaining: frozenset[Cell]) -> Solution | None:
        """Best (score, footprints) tiling of `remaining`, or None if it cannot be tiled."""
        if not remaining:
            return (0, 0), ()
        if remaining in self.memo:
            return self.memo[remaining]
        x, y = min(remaining, key=lambda cell: (cell[1], cell[0]))
        best: Solution | None = None
        for w, d in self.sizes:
            covered = [(x + i, y + j) for j in range(d) for i in range(w)]
            if not all(cell in remaining for cell in covered):
                continue
            rest = self.solve(remaining - frozenset(covered))
            if rest is None:
                continue
            total = (rest[0][0] + (w * d == 1), rest[0][1] + 1)
            if best is None or total < best[0]:
                best = (total, ((x, y, w, d), *rest[1]))
        self.memo[remaining] = best
        return best


def _tile(cells: Iterable[Cell], sizes: Sequence[tuple[int, int]], variant: TileVariant) -> list[Footprint] | None:
    """Exact tiling of `cells` {(x, y)} by rectangles of `sizes` [(w, d)] minimising (1x1 count, pieces), searched
    from the variant's corner. Returns [(x, y, w, d)] in the original coordinates, or None."""
    search_sizes = [(d, w) if variant.swaps_axes else (w, d) for (w, d) in sizes]
    solution = _TilingSearch(search_sizes).solve(frozenset(_to_search(cell, variant) for cell in cells))
    if solution is None:
        return None
    footprints = []
    for x, y, w, d in solution[1]:
        covered = [_from_search((x + i, y + j), variant) for j in range(d) for i in range(w)]
        xs, ys = [c[0] for c in covered], [c[1] for c in covered]
        footprints.append((min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1))
    return footprints


def _quality(asm: Assembly, boundaries: Iterable[int], window: Window) -> tuple[int, int]:
    """(aligned seams, exact stacks) in the window at the given layer boundaries (layer L on layer L - 1)."""
    aligned = stacked = 0
    rows, columns = slice(window.y0, window.y1), slice(window.x0, window.x1)
    for layer in boundaries:
        if layer < 1 or asm.nz <= layer:
            continue
        upper = asm.owner[layer, rows, columns]
        lower = asm.owner[layer - 1, rows, columns]
        changes = (upper != lower) & (upper >= 0) & (lower >= 0)
        for first, second in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1], np.s_[1:])):
            upper_first, upper_second = upper[first], upper[second]
            lower_first, lower_second = lower[first], lower[second]
            aligned += int(
                (
                    (upper_first != upper_second)
                    & (upper_first >= 0)
                    & (upper_second >= 0)
                    & (lower_first != lower_second)
                    & (lower_first >= 0)
                    & (lower_second >= 0)
                    & changes[first]
                    & changes[second]
                ).sum()
            )
        for index in np.unique(upper[upper >= 0]).tolist():
            piece = asm.bricks[index]
            if piece is None or piece.z != layer:
                continue
            below = np.unique(asm.owner[layer - 1, piece.y : piece.y + piece.d, piece.x : piece.x + piece.w])
            below = below[below >= 0]
            if len(below) == 1:
                under = asm.bricks[int(below[0])]
                if under is not None:
                    stacked += (under.x, under.y, under.w, under.d) == (piece.x, piece.y, piece.w, piece.d)
    return aligned, stacked


def _edge_ring(piece: PlacedBrick) -> list[Cell]:
    """Cells (x, y) just outside the piece's four edges (left, right, top, bottom)."""
    return (
        [(piece.x - 1, y) for y in range(piece.y, piece.y + piece.d)]
        + [(piece.x + piece.w, y) for y in range(piece.y, piece.y + piece.d)]
        + [(x, piece.y - 1) for x in range(piece.x, piece.x + piece.w)]
        + [(x, piece.y + piece.d) for x in range(piece.x, piece.x + piece.w)]
    )


def _group(asm: Assembly, index: int) -> list[int]:
    """Piece `index` plus same-colour neighbours (same layer and height, sharing an edge), up to NEIGHBOUR_RINGS
    rings / MAX_GROUP_CELLS cells."""
    seed = asm.bricks[index]
    assert seed is not None
    ids, cell_count = [index], seed.w * seed.d
    frontier = [index]
    for _ in range(NEIGHBOUR_RINGS):
        next_frontier = []
        for member_index in frontier:
            member = asm.bricks[member_index]
            assert member is not None
            for x, y in _edge_ring(member):
                if not (0 <= x < asm.nx and 0 <= y < asm.ny):
                    continue
                j = int(asm.owner[seed.z, y, x])
                if j < 0 or j in ids:
                    continue
                neighbour = asm.bricks[j]
                if (
                    neighbour is not None
                    and neighbour.z == seed.z
                    and neighbour.h_plates == seed.h_plates
                    and neighbour.color == seed.color
                    and cell_count + neighbour.w * neighbour.d <= MAX_GROUP_CELLS
                ):
                    ids.append(j)
                    cell_count += neighbour.w * neighbour.d
                    next_frontier.append(j)
        frontier = next_frontier
    return ids


def _better_tilings(cells: set[Cell], sizes: Sequence[tuple[int, int]], current: TilingScore) -> list[list[Footprint]]:
    """Distinct exact tilings of the cells (one per TILE_VARIANTS entry) that score better than `current`, best
    first (stable among equal scores)."""
    seen: set[frozenset[Footprint]] = set()
    candidates: list[tuple[TilingScore, list[Footprint]]] = []
    for variant in TILE_VARIANTS:
        tiling = _tile(cells, sizes, variant)
        if tiling is None:
            continue
        key = frozenset(tiling)
        if key in seen:
            continue
        seen.add(key)
        score = (_count_single_studs(tiling), len(tiling))
        if score < current:
            candidates.append((score, tiling))
    candidates.sort(key=lambda candidate: candidate[0])
    return [tiling for _, tiling in candidates]


class _Allowance(NamedTuple):
    """Extra exact stacks / aligned-seam segments a re-tiling may add per group (0 = never weaker)."""

    stacks: int
    seams: int


def _owned_used(asm: Assembly) -> int:
    """Owned pieces in use (0 without stock)."""
    return asm.stock.owned_used() if asm.stock is not None else 0


def _need_total(asm: Assembly) -> int:
    """Pieces to buy (0 without stock)."""
    return asm.stock.need_total() if asm.stock is not None else 0


def _retile_group(asm: Assembly, ids: list[int], hang_min: int, allowance: _Allowance) -> bool:
    """Replace the group's pieces by the first better tiling that keeps every acceptance rule (see module doc);
    the assembly is rolled back for each rejected tiling. True if a tiling was kept."""
    olds = [piece for piece in (asm.bricks[j] for j in ids) if piece is not None]
    first = olds[0]
    cells = {(x, y) for b in olds for y in range(b.y, b.y + b.d) for x in range(b.x, b.x + b.w)}
    xs, ys = [c[0] for c in cells], [c[1] for c in cells]
    width, depth = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
    sizes = [size for size in piece_sizes(asm) if size[0] <= width and size[1] <= depth]
    current = (_count_single_studs((b.x, b.y, b.w, b.d) for b in olds), len(olds))
    tilings = _better_tilings(cells, sizes, current)
    if not tilings:
        return False
    z, height = first.z, first.h_plates
    window = Window(max(0, min(ys) - 1), min(asm.ny, max(ys) + 2), max(0, min(xs) - 1), min(asm.nx, max(xs) + 2))
    boundaries = {z, z + height}
    seams_before, stacks_before = _quality(asm, boundaries, window)
    score_before, owned_before, need_before = score(asm, hang_min), _owned_used(asm), _need_total(asm)
    for tiling in tilings:
        asm.log = []
        for j in ids:
            asm.remove(j)
        for x, y, w, d in tiling:
            asm.add(PlacedBrick(x, y, z, w, d, height, first.color))
        seams_after, stacks_after = _quality(asm, boundaries, window)
        if (
            seams_after <= seams_before + allowance.seams
            and stacks_after <= stacks_before + allowance.stacks
            and _owned_used(asm) >= owned_before
            and _need_total(asm) <= need_before
            and all(a <= b for a, b in zip(score(asm, hang_min), score_before, strict=True))
        ):
            asm.log = None
            return True
        asm.rollback()
    return False


def _count_pieces(asm: Assembly) -> tuple[int, int]:
    """(1x1 pieces, pieces) of the assembly."""
    live = [b for b in asm.bricks if b is not None]
    return sum(1 for b in live if b.w == 1 and b.d == 1), len(live)


def cleanup(asm: Assembly, hang_min: int, *, verbose: bool = True, extra_stacks: int = 0, extra_seams: int = 0) -> int:
    """Re-tile groups around 1x1 pieces. Returns the number of groups changed. extra_stacks / extra_seams: exact
    stacks / aligned-seam segments a re-tiling may add per group (0 = never weaker)."""
    allowance = _Allowance(extra_stacks, extra_seams)
    single_before, total_before = _count_pieces(asm)
    changed = 0
    for i in range(len(asm.bricks)):
        piece = asm.bricks[i]
        if piece is None or piece.w != 1 or piece.d != 1:
            continue
        ids = _group(asm, i)
        if len(ids) >= MIN_GROUP_PIECES and _retile_group(asm, ids, hang_min, allowance):
            changed += 1
    if verbose:
        single_after, total_after = _count_pieces(asm)
        print(
            f"Clean-up (lossless): {changed} groups re-tiled, 1x1 pieces {single_before} -> {single_after}, "
            f"pieces {total_before} -> {total_after}"
        )
    return changed
