"""Merge neighbouring plates into one bigger plate (--merge-plates).

After packing, tie repair, clean-up and the floating-piece steps, a layer can still hold groups of 2 or 3 plates
that together fill a standard plate exactly: two 1x1 side by side (-> 1x2), a 1x2 + two 2x1 (-> 2x3), and so on.
The packer leaves them on purpose (staggered seams, an owned piece kept in use) or a later step created them
(a split brick). Each such group becomes one plate when
- every visible cell keeps its colour: the group is one colour, or pieces of another colour cover only hidden
  cells (they take the visible colour, as hidden cells may be any colour);
- pieces to buy do not go up (an owned piece may drop out if the merged plate saves buying);
- the stability score (loose pieces, big single-stud joints, single-stud footings) is not worse.
Biggest saving first, repeated until nothing changes.

Pieces: model.PlacedBrick in the model grid; visible: bool [z, y, x] surface cells (None = all visible).
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from image_to_bricks.model import PlacedBrick, owner_grid, piece_cells
from image_to_bricks.packing.retile_setting import RetileSetting, StabilityScore, is_not_worse
from image_to_bricks.stock import PLATE_HEIGHT_PLATES, PieceKey, Stock, make_key

MERGE_GROUP_PIECES = 3  # largest group (pieces) merged into one plate
SIDE_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))  # (step x, step y) to the four side neighbours of a stud


@dataclass(frozen=True)
class MergeResult:
    """Pieces before / after and the number of merges kept."""

    pieces_before: int
    pieces_after: int
    merges: int


def _rectangle_key(piece: PlacedBrick) -> PieceKey:
    """Stock key of a rectangular piece."""
    return make_key(piece.h_plates, piece.w, piece.d, piece.color)


def _union_rectangle(pieces: Sequence[PlacedBrick]) -> tuple[int, int, int, int] | None:
    """(x, y, w, d) if the pieces' cells form one full rectangle, else None."""
    cells = [cell for piece in pieces for cell in piece_cells(piece)]
    xs, ys = [x for x, _ in cells], [y for _, y in cells]
    x0, y0, width, depth = min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
    return (x0, y0, width, depth) if len(set(cells)) == len(cells) == width * depth else None


def _merged_colour(pieces: Sequence[PlacedBrick], visible: NDArray[np.bool_] | None) -> int | None:
    """The one colour the merged plate can take without changing a visible cell, or None."""
    visible_colours = set()
    for piece in pieces:
        if visible is None or any(visible[piece.z, y, x] for x, y in piece_cells(piece)):
            visible_colours.add(piece.color)
    if len(visible_colours) == 1:
        return visible_colours.pop()
    if not visible_colours and len({p.color for p in pieces}) == 1:
        return pieces[0].color
    return None


def _is_mergeable_plate(piece: PlacedBrick | None, layer: int) -> bool:
    """True if piece is a rectangular 1-plate piece in `layer`."""
    return piece is not None and piece.z == layer and piece.h_plates == 1 and piece.notch < 0


def _neighbour_groups(
    bricks: Sequence[PlacedBrick | None], owner: NDArray[np.int32], index: int
) -> list[tuple[int, ...]]:
    """Groups (2..MERGE_GROUP_PIECES pieces) of 1-plate rectangles in the layer of piece `index`, containing it,
    built from it and its side neighbours."""
    piece = bricks[index]
    if piece is None:
        return []
    neighbours = set()
    for x, y in piece_cells(piece):
        for step_x, step_y in SIDE_STEPS:
            neighbour_y, neighbour_x = y + step_y, x + step_x
            if 0 <= neighbour_y < owner.shape[1] and 0 <= neighbour_x < owner.shape[2]:
                other = int(owner[piece.z, neighbour_y, neighbour_x])
                if other >= 0 and other != index and _is_mergeable_plate(bricks[other], piece.z):
                    neighbours.add(other)
    groups = []
    for size in range(1, MERGE_GROUP_PIECES):
        groups += [(index, *combo) for combo in combinations(sorted(neighbours), size)]
    return groups


def _count_to_buy(stock: Stock, used: Counter[PieceKey]) -> int:
    """Pieces of `used` beyond what the stock owns."""
    return sum(max(0, count - stock.have.get(key, 0)) for key, count in used.items())


@dataclass
class _Merger:
    """One merge pass: the pieces (a merged-away piece stays as None, so every index stays valid), the stock keys
    they use and their current stability score."""

    setting: RetileSetting
    visible: NDArray[np.bool_] | None
    work: list[PlacedBrick | None]
    current_score: StabilityScore
    allowed_sizes: set[tuple[int, ...]] = field(init=False)
    used: Counter[PieceKey] = field(init=False)

    def __post_init__(self) -> None:
        self.allowed_sizes = {tuple(sorted(size)) for size in self.setting.sizes}
        self.used = Counter(_rectangle_key(b) for b in self.work if b is not None and b.notch < 0)

    def merge_order(self, first_layer: int) -> list[int]:
        """Indices of the plates that may start a group, smallest first, then lowest layer."""
        return sorted(
            (
                i
                for i, b in enumerate(self.work)
                if b is not None and b.z >= first_layer and b.h_plates == 1 and b.notch < 0
            ),
            key=lambda i: (self._piece(i).w * self._piece(i).d, self._piece(i).z),
        )

    def _piece(self, index: int) -> PlacedBrick:
        piece = self.work[index]
        if piece is None:
            raise ValueError(f"piece {index} was merged away")
        return piece

    def try_merge(self, group: tuple[int, ...]) -> bool:
        """Replaces the group by one plate in `work` if every rule holds; returns True if it did."""
        pieces = [self._piece(i) for i in group]
        merged = self._merged_plate(pieces)
        if merged is None:
            return False
        trial_used = Counter(self.used)
        for piece in pieces:
            trial_used[_rectangle_key(piece)] -= 1
        trial_used[_rectangle_key(merged)] += 1
        stock = self.setting.stock
        if stock is not None and _count_to_buy(stock, trial_used) > _count_to_buy(stock, self.used):
            return False
        trial = [b for i, b in enumerate(self.work) if b is not None and i not in group] + [merged]
        trial_score = self.setting.score(trial)
        if not is_not_worse(trial_score, self.current_score):
            return False
        self._apply(group, pieces, merged, trial_used)
        self.current_score = trial_score
        return True

    def _merged_plate(self, pieces: Sequence[PlacedBrick]) -> PlacedBrick | None:
        """The one plate that can replace the pieces (allowed size, no visible colour changes), or None."""
        rectangle = _union_rectangle(pieces)
        if rectangle is None or tuple(sorted(rectangle[2:])) not in self.allowed_sizes:
            return None
        colour = _merged_colour(pieces, self.visible)
        if colour is None:
            return None
        x, y, width, depth = rectangle
        return PlacedBrick(x, y, pieces[0].z, width, depth, PLATE_HEIGHT_PLATES, colour)

    def _apply(
        self, group: tuple[int, ...], pieces: Sequence[PlacedBrick], merged: PlacedBrick, used: Counter[PieceKey]
    ) -> None:
        """Puts the merged plate in place of the group; the used counts and the stock ledger follow."""
        for i in group:
            self.work[i] = None
        self.work.append(merged)
        self.used = used
        stock = self.setting.stock
        if stock is not None:
            for piece in pieces:
                stock.give_brick(piece)
            stock.take_brick(merged)


def merge_plates(
    bricks: list[PlacedBrick],
    setting: RetileSetting,
    visible: NDArray[np.bool_] | None = None,
    first_layer: int = 0,
) -> tuple[list[PlacedBrick], MergeResult]:
    """Merges plate groups (see the module docstring) in layers >= first_layer; setting.sizes are the plate sizes
    a merged plate may have. Returns (new bricks, result); the stock ledger follows every kept merge."""
    merger = _Merger(setting, visible, list(bricks), setting.score(bricks))
    merges = 0
    is_changed = True
    while is_changed:
        is_changed = False
        owner = owner_grid(merger.work, setting.grid_shape)
        for index in merger.merge_order(first_layer):
            if merger.work[index] is None:
                continue
            for group in sorted(_neighbour_groups(merger.work, owner, index), key=len, reverse=True):
                if merger.try_merge(group):
                    merges += 1
                    is_changed = True
                    owner = owner_grid(merger.work, setting.grid_shape)
                    break
    result_bricks = [b for b in merger.work if b is not None]
    return result_bricks, MergeResult(len(bricks), len(result_bricks), merges)
