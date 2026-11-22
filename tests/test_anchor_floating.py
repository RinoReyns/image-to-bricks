"""--anchor-floating (anchor_floating.py): a plate with nothing under it is re-tiled with a supported same-colour
neighbour so it rests on the layer below; level 2 also splits a neighbouring brick; cells / colours unchanged."""

import contextlib
import io
import sys

import support

support.stub_mesh_libraries()
import numpy as np

from image_to_bricks.model import PlacedBrick, piece_cells
from image_to_bricks.packing import anchor_floating as af
from image_to_bricks.packing import assembly, placement
from image_to_bricks.packing.retile_setting import RetileSetting
from image_to_bricks.stock import Stock

BLACK, ORANGE = 6, 9
SHAPE = (4, 3, 5)
failed = False


def check(name, ok, info=None):
    global failed
    print(("ok   " if ok else "FAIL ") + name + ("" if ok or info is None else f"  {info}"))
    failed |= not ok


def never_worse(_bricks):
    return (0,)


def setting(score=never_worse, stock=None):
    return RetileSetting(SHAPE, assembly.SIZES, score, stock)


def cells(bricks):
    return sorted((z, y, x, b.color) for b in bricks for x, y in piece_cells(b) for z in range(b.z, b.z + b.h_plates))


def boot_step(neighbour_colour=BLACK):
    """Layer 0: Black 2x1 plate. Layer 1: Black 2x1 plate on it + a 1x1 plate one stud further out (the boot
    stepping outwards) with nothing under it."""
    return [
        PlacedBrick(0, 0, 0, 2, 1, 1, BLACK),
        PlacedBrick(0, 0, 1, 2, 1, 1, neighbour_colour),
        PlacedBrick(2, 0, 1, 1, 1, 1, BLACK),
    ]


def test_floating_plate_joins_its_supported_neighbour():
    bricks = boot_step()
    new, result = af.anchor_floating_pieces(list(bricks), setting())
    check("1 floating -> 0", (result.floating_before, result.floating_after) == (1, 0), result)
    check(
        "layer 1 is now one 3x1 plate",
        sorted((b.z, b.x, b.w, b.d) for b in new) == [(0, 0, 2, 1), (1, 0, 3, 1)],
        [(b.z, b.x, b.w, b.d) for b in new],
    )
    check("cells and colours unchanged", cells(new) == cells(bricks))


def test_other_colour_neighbour_is_not_used():
    bricks = boot_step(neighbour_colour=ORANGE)
    new, result = af.anchor_floating_pieces(list(bricks), setting())
    check("different colour: nothing changes", result.retilings == 0 and cells(new) == cells(bricks))


def brick_step():
    """A Black 2x1 BRICK (layers 0-2) and a 1x1 plate beside its middle slice with nothing under it."""
    return [PlacedBrick(0, 0, 0, 2, 1, 3, BLACK), PlacedBrick(2, 0, 1, 1, 1, 1, BLACK)]


def test_brick_neighbour_only_split_at_level_2():
    bricks = brick_step()
    _, plates_only = af.anchor_floating_pieces(list(bricks), setting(), 1)
    check("level 1 leaves a brick neighbour alone", plates_only.retilings == 0, plates_only)
    new, split = af.anchor_floating_pieces(list(bricks), setting(), 3)
    check("level 2 splits the brick: 0 floating", split.floating_after == 0, split)
    check(
        "brick -> 2x1, 3x1, 2x1 plates",
        sorted((b.z, b.w, b.h_plates) for b in new) == [(0, 2, 1), (1, 3, 1), (2, 2, 1)],
        [(b.z, b.w, b.h_plates) for b in new],
    )
    check("cells and colours unchanged after the split", cells(new) == cells(bricks))


def test_change_that_weakens_the_build_is_rejected():
    calls = []

    def worse_after_change(_bricks):
        calls.append(1)
        return (0,) if len(calls) == 1 else (1,)

    new, result = af.anchor_floating_pieces(boot_step(), setting(worse_after_change))
    check("score got worse -> kept as it was", result.retilings == 0 and len(new) == 3)


def test_stock_ledger_follows():
    bricks = boot_step()
    stock = Stock(have={})
    for piece in bricks:
        stock.take_brick(piece)
    new, _ = af.anchor_floating_pieces(list(bricks), setting(stock=stock), 1)
    check("ledger == final pieces", +stock.recount(new).used == +stock.used)


def test_tilings_are_exact_covers():
    region = {(0, 0), (1, 0), (2, 0), (0, 1)}
    tilings = af.enumerate_tilings(region, assembly.SIZES)
    covers = [sorted((x + i, y + j) for x, y, w, d in t for j in range(d) for i in range(w)) for t in tilings]
    check("every tiling covers the region exactly once", tilings and all(c == sorted(region) for c in covers))


def test_pipeline_step_and_cli():
    grid = np.zeros(SHAPE, bool)
    for piece in brick_step():
        for x, y in piece_cells(piece):
            grid[piece.z : piece.z + piece.h_plates, y, x] = True
    with contextlib.redirect_stdout(io.StringIO()) as log:
        new = placement._anchor_floating(brick_step(), placement.PassContext(grid, None, 0, 5), 2)
    check("console line", "Anchor floating: pieces with nothing under them 1 -> 0" in log.getvalue(), log.getvalue())
    check("pipeline step returns the anchored pieces", len(new) == 3)


if __name__ == "__main__":
    for test in [value for key, value in list(globals().items()) if key.startswith("test_")]:
        test()
    sys.exit(1 if failed else 0)
