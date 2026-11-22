"""Sturdier --accent: build the accent plate from smaller plates of the wanted colour when the packer did not
make one plate of the accent size.

The accent (e.g. the chrome ball plate on the sword guard) needs one W x D plate of a given colour inside its box.
The packer may cover the same cells with 1x1 plates instead. Plates of that colour lying ENTIRELY inside a W x D
footprint of one layer that they fill exactly are merged into one W x D plate: same cells, same colour, and stud
contacts only grow (the merged plate touches everything the small ones touched), so the build stays as strong.
Coordinates: model grid (PlacedBrick x, y, z); the box test is done by the caller.
"""

from collections.abc import Callable
from dataclasses import dataclass

from image_to_bricks.model import PlacedBrick, piece_cells
from image_to_bricks.stock import Stock

MIN_PLATES_TO_MERGE = 2  # a merge replaces at least two plates (one plate of the size is used as it is)


@dataclass(frozen=True)
class MergedPlate:
    """The new plate and the plates it replaces."""

    plate: PlacedBrick
    replaced: list[PlacedBrick]


def _plates_covering_footprint(
    cell_owner: dict[tuple[int, int], PlacedBrick], cells: list[tuple[int, int]]
) -> list[PlacedBrick] | None:
    """The plates covering `cells` if every cell is covered and every such plate lies inside `cells`, else None."""
    owners: list[PlacedBrick] = []
    for cell in cells:
        owner = cell_owner.get(cell)
        if owner is None:
            return None
        if all(owner is not known for known in owners):
            owners.append(owner)
    inside = set(cells)
    return owners if all(set(piece_cells(owner)) <= inside for owner in owners) else None


def find_mergeable_footprint(
    bricks: list[PlacedBrick], width: int, depth: int, colour: int, is_in_box: Callable[[int, int, int], bool]
) -> MergedPlate | None:
    """Highest W x D footprint (either orientation) inside the box that is covered exactly by two or more
    rectangular 1-plate pieces of `colour`; None if there is none. is_in_box(z, y, x) tests the footprint corner
    (model grid)."""
    plates_by_layer: dict[int, list[PlacedBrick]] = {}
    for brick in bricks:
        if brick.h_plates == 1 and brick.notch < 0 and brick.color == colour:
            plates_by_layer.setdefault(brick.z, []).append(brick)
    for z in sorted(plates_by_layer, reverse=True):
        cell_owner = {cell: plate for plate in plates_by_layer[z] for cell in piece_cells(plate)}
        for x, y in sorted(cell_owner, key=lambda cell: (cell[1], cell[0])):
            for plate_width, plate_depth in {(width, depth), (depth, width)}:
                if not is_in_box(z, y, x):
                    continue
                cells = [(x + i, y + j) for j in range(plate_depth) for i in range(plate_width)]
                owners = _plates_covering_footprint(cell_owner, cells)
                if owners and len(owners) >= MIN_PLATES_TO_MERGE:
                    return MergedPlate(PlacedBrick(x, y, z, plate_width, plate_depth, 1, colour), owners)
    return None


def apply_merge(bricks: list[PlacedBrick], merge: MergedPlate, stock: Stock | None = None) -> PlacedBrick:
    """Replaces merge.replaced by merge.plate in `bricks` (in place) and in the running stock ledger."""
    for old in merge.replaced:
        bricks.remove(old)
        if stock is not None:
            stock.give_brick(old)
    bricks.append(merge.plate)
    if stock is not None:
        stock.take_brick(merge.plate)
    return merge.plate
