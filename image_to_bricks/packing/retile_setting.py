"""What a re-tiling pass after packing works with (--anchor-floating, --merge-plates).

Both passes replace a few pieces by others over exactly the same cells and keep the change only if the stability
score does not get worse; the stock ledger follows every kept change.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from image_to_bricks.model import PlacedBrick
from image_to_bricks.stock import Stock

type StabilityScore = tuple[int, ...]  # e.g. (loose pieces, big single-stud joints, single-stud footings)
type ScoreFunction = Callable[[list[PlacedBrick]], StabilityScore]  # lower is better in every entry


@dataclass(frozen=True)
class RetileSetting:
    """The model a re-tiling pass changes and the rules a change must keep."""

    grid_shape: tuple[int, int, int]  # (z, y, x) of the model grid
    sizes: Sequence[tuple[int, int]]  # (width, depth) in studs of the plates the packer may use
    score: ScoreFunction  # stability score of a piece list
    stock: Stock | None = None  # ledger that follows every kept change (None = no stock)


def is_not_worse(new_score: StabilityScore, old_score: StabilityScore) -> bool:
    """True if no entry of new_score is higher than the same entry of old_score."""
    return all(new <= old for new, old in zip(new_score, old_score, strict=True))
