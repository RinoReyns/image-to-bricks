"""Fewer pieces that hang in the air while building (--anchor-floating).

Built layer by layer, a plate with nothing directly under it floats until a piece of a higher layer clamps it
(the PDF marks it with "!"). Many of these sit right next to a piece of the SAME colour in the same layer that
does rest on the layer below (boot and leg outlines that step outwards). Such a floating plate and that neighbour
are tiled again together (exact cover of their cells, same colour, plate sizes the packer may use) so that every
new piece has at least one stud on the layer below. If the neighbour is a 3-plate brick, it is split: its slice in
the floating plate's layer joins the re-tiling, its other two slices become plates of its size (2 more pieces).
A change is kept only if the stability score (loose pieces, big single-stud joints, single-stud footings) does not
get worse; cells and colours never change. Floating pieces with no such neighbour (a tail tip, a hat brim) stay
floating.

Pieces: model.PlacedBrick in the model grid; grid bool [z, y, x].
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from image_to_bricks.model import PlacedBrick, owner_grid, piece_cells
from image_to_bricks.packing.retile_setting import RetileSetting, StabilityScore, is_not_worse

ANCHOR_MAX_REGION_CELLS = 16  # largest region (floating piece + neighbour) that is tiled again
ANCHOR_MAX_TILINGS = 400  # tilings enumerated per region
ANCHOR_EXTRA_SIZES = [(6, 1), (1, 6), (8, 1), (1, 8), (6, 2), (2, 6), (8, 2), (2, 8)]  # long standard plates
SIDE_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))  # (step x, step y) to the four side neighbours of a stud

type Cell = tuple[int, int]  # (x, y) in studs
type Rectangle = tuple[int, int, int, int]  # (x, y, width, depth) in studs


@dataclass(frozen=True)
class AnchorResult:
    """Floating pieces before / after and how many re-tilings were kept."""

    floating_before: int
    floating_after: int
    retilings: int


@dataclass
class _Workspace:
    """The pieces during the pass (a removed piece stays as None, so every index stays valid), the owner grid of
    the live pieces and their current stability score."""

    bricks: list[PlacedBrick | None]
    setting: RetileSetting
    current_score: StabilityScore
    owner: NDArray[np.int32] = field(init=False)

    def __post_init__(self) -> None:
        self.owner = owner_grid(self.bricks, self.setting.grid_shape)

    def live_piece(self, index: int) -> PlacedBrick:
        """The piece at `index`; it must not have been removed (the owner grid only points at live pieces)."""
        brick = self.bricks[index]
        if brick is None:
            raise ValueError(f"piece {index} was removed by an earlier re-tiling")
        return brick


def floating_pieces(bricks: Sequence[PlacedBrick | None], owner: NDArray[np.int32]) -> set[int]:
    """Indices of pieces above layer 0 with no piece directly under any of their studs (None entries skipped)."""
    return {
        index
        for index, brick in enumerate(bricks)
        if brick is not None and brick.z > 0 and all(owner[brick.z - 1, y, x] < 0 for x, y in piece_cells(brick))
    }


def enumerate_tilings(
    region: set[Cell], sizes: Sequence[tuple[int, int]], limit: int = ANCHOR_MAX_TILINGS
) -> list[list[Rectangle]]:
    """Exact covers of region by rectangles (x, y, w, d) of the given sizes, each anchored at the first free
    cell in (y, x) order; at most `limit` of them."""
    found: list[list[Rectangle]] = []

    def extend(free: set[Cell], placed: list[Rectangle]) -> None:
        if len(found) >= limit:
            return
        if not free:
            found.append(list(placed))
            return
        x, y = min(free, key=lambda cell: (cell[1], cell[0]))
        for width, depth in sizes:
            cells = {(x + i, y + j) for j in range(depth) for i in range(width)}
            if cells <= free:
                placed.append((x, y, width, depth))
                extend(free - cells, placed)
                placed.pop()

    extend(set(region), [])
    return found


def _best_anchored_tiling(
    region: set[Cell], is_supported: Callable[[Cell], bool], sizes: Sequence[tuple[int, int]]
) -> list[Rectangle] | None:
    """Tiling of region where every piece has a supported cell; fewest pieces, then fewest 1x1; or None."""
    best: tuple[tuple[int, int], list[Rectangle]] | None = None
    for tiling in enumerate_tilings(region, sizes):
        if all(any(is_supported((x + i, y + j)) for j in range(d) for i in range(w)) for x, y, w, d in tiling):
            key = (len(tiling), sum(1 for _, _, w, d in tiling if w * d == 1))
            if best is None or key < best[0]:
                best = (key, tiling)
    return None if best is None else best[1]


def _side_neighbours(owner: NDArray[np.int32], brick: PlacedBrick) -> list[int]:
    """Owner indices of the cells beside the piece in its layer (with repeats, -1 = empty), in stud order."""
    found = []
    for x, y in piece_cells(brick):
        for step_x, step_y in SIDE_STEPS:
            is_inside = 0 <= y + step_y < owner.shape[1] and 0 <= x + step_x < owner.shape[2]
            found.append(int(owner[brick.z, y + step_y, x + step_x]) if is_inside else -1)
    return found


def _can_anchor(neighbour: PlacedBrick, brick: PlacedBrick, max_neighbour_plates: int) -> bool:
    """True if neighbour is a rectangle of brick's colour, at most max_neighbour_plates tall, in brick's layer."""
    return (
        neighbour.notch < 0
        and neighbour.color == brick.color
        and neighbour.h_plates <= max_neighbour_plates
        and neighbour.z <= brick.z < neighbour.z + neighbour.h_plates
    )


def _anchoring_neighbours(work: _Workspace, index: int, floating: set[int], max_neighbour_plates: int) -> list[int]:
    """Same-colour rectangular pieces beside piece `index` in its layer that are not floating and at most
    max_neighbour_plates tall (1 = plates only, 3 = bricks too); plates first, then the smallest."""
    brick = work.live_piece(index)
    found: list[int] = []
    for other in _side_neighbours(work.owner, brick):
        if other < 0 or other == index or other in floating or other in found:
            continue
        if _can_anchor(work.live_piece(other), brick, max_neighbour_plates):
            found.append(other)
    return sorted(found, key=lambda i: (work.live_piece(i).h_plates, work.live_piece(i).w * work.live_piece(i).d))


def anchor_floating_pieces(
    bricks: list[PlacedBrick], setting: RetileSetting, max_neighbour_plates: int = 1
) -> tuple[list[PlacedBrick], AnchorResult]:
    """Re-tiles floating 1-plate pieces with a supported same-colour neighbour (see the module docstring);
    max_neighbour_plates 1 = only plate neighbours, 3 = a brick neighbour is split too. A change is kept only if
    the score is not worse in any entry. Returns (new bricks, result); the stock ledger follows every kept change.
    """
    sizes = sorted(set(setting.sizes) | set(ANCHOR_EXTRA_SIZES), key=lambda size: -size[0] * size[1])
    work = _Workspace(list(bricks), setting, setting.score(bricks))
    floating = floating_pieces(work.bricks, work.owner)
    retilings = 0
    for index in sorted(floating, key=lambda i: (bricks[i].z, bricks[i].y, bricks[i].x)):
        brick = work.bricks[index]
        if brick is None or brick.h_plates != 1 or brick.notch >= 0:
            continue
        for neighbour in _anchoring_neighbours(work, index, floating, max_neighbour_plates):
            if _try_retile(work, index, neighbour, sizes):
                retilings += 1
                break
    live_bricks = [brick for brick in work.bricks if brick is not None]
    floating_after = len(floating_pieces(live_bricks, owner_grid(live_bricks, setting.grid_shape)))
    return live_bricks, AnchorResult(len(floating), floating_after, retilings)


def _retiled_pieces(floating_brick: PlacedBrick, neighbour: PlacedBrick, tiling: list[Rectangle]) -> list[PlacedBrick]:
    """Plates of the tiling in the floating piece's layer, plus the other slices of a split brick neighbour."""
    layer = floating_brick.z
    new_pieces = [PlacedBrick(x, y, layer, w, d, 1, floating_brick.color) for x, y, w, d in tiling]
    new_pieces += [
        PlacedBrick(neighbour.x, neighbour.y, slice_z, neighbour.w, neighbour.d, 1, neighbour.color)
        for slice_z in range(neighbour.z, neighbour.z + neighbour.h_plates)
        if slice_z != layer
    ]
    return new_pieces


def _try_retile(work: _Workspace, index: int, neighbour_index: int, sizes: Sequence[tuple[int, int]]) -> bool:
    """Re-tiles piece `index` with its neighbour (a brick is split into plates) in `work` if that anchors it and
    does not worsen the score; returns True if the change was kept. Removed pieces become None in place."""
    floating_brick, neighbour = work.live_piece(index), work.live_piece(neighbour_index)
    region = set(piece_cells(floating_brick)) | set(piece_cells(neighbour))
    if len(region) > ANCHOR_MAX_REGION_CELLS:
        return False
    below = floating_brick.z - 1
    tiling = _best_anchored_tiling(region, lambda cell: bool(work.owner[below, cell[1], cell[0]] >= 0), sizes)
    if tiling is None:
        return False
    new_pieces = _retiled_pieces(floating_brick, neighbour, tiling)
    removed = (index, neighbour_index)
    trial = [b for i, b in enumerate(work.bricks) if i not in removed and b is not None] + new_pieces
    trial_score = work.setting.score(trial)
    if not is_not_worse(trial_score, work.current_score):
        return False
    stock = work.setting.stock
    if stock is not None:
        stock.give_brick(floating_brick)
        stock.give_brick(neighbour)
        for piece in new_pieces:
            stock.take_brick(piece)
    for removed_index in removed:
        work.bricks[removed_index] = None
    work.bricks.extend(new_pieces)
    work.current_score = trial_score
    work.owner = owner_grid(work.bricks, work.setting.grid_shape)
    return True
