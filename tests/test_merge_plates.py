"""--merge-plates (merge_plates.py): neighbouring plates that fill a standard plate exactly become one plate;
visible colours never change, hidden cells may, buying never grows, stability never gets worse."""

import contextlib
import io
import sys

import support

support.stub_mesh_libraries()
import numpy as np

from image_to_bricks.model import PlacedBrick, piece_cells
from image_to_bricks.packing import assembly, placement
from image_to_bricks.packing import merge_plates as mp
from image_to_bricks.packing.retile_setting import RetileSetting
from image_to_bricks.stock import Stock, make_key

BLACK, WHITE, ORANGE = 6, 1, 9
SHAPE = (2, 4, 4)
SIZES = assembly.SIZES
failed = False


def check(name, ok, info=None):
    global failed
    print(("ok   " if ok else "FAIL ") + name + ("" if ok or info is None else f"  {info}"))
    failed |= not ok


def never_worse(_bricks):
    return (0,)


def setting(score=never_worse, stock=None, shape=SHAPE):
    return RetileSetting(shape, SIZES, score, stock)


def layout(bricks):
    return sorted((b.z, b.x, b.y, b.w, b.d, b.color) for b in bricks)


def test_two_1x1_become_1x2():
    bricks = [PlacedBrick(0, 0, 0, 1, 1, 1, BLACK), PlacedBrick(0, 1, 0, 1, 1, 1, BLACK)]
    new, result = mp.merge_plates(bricks, setting())
    check("two 1x1 -> one 1x2", layout(new) == [(0, 0, 0, 1, 2, BLACK)] and result.merges == 1, layout(new))


def layer_6_case(white_visible: bool):
    """Step-70 figure layer 6: Black 1x2 + Black 2x1 + White 2x1 (hidden) fill a 3x2."""
    bricks = [
        PlacedBrick(0, 0, 0, 1, 2, 1, BLACK),
        PlacedBrick(1, 1, 0, 2, 1, 1, BLACK),
        PlacedBrick(1, 0, 0, 2, 1, 1, WHITE),
    ]
    visible = np.zeros(SHAPE, bool)
    visible[0, :, 0] = True
    visible[0, 0, 1:3] = white_visible
    return bricks, visible


def test_hidden_other_colour_joins():
    bricks, visible = layer_6_case(white_visible=False)
    new, _ = mp.merge_plates(bricks, setting(), visible)
    check("1x2 + 2x1 + hidden White 2x1 -> one Black 3x2", layout(new) == [(0, 0, 0, 3, 2, BLACK)], layout(new))


def test_visible_other_colour_stays():
    bricks, visible = layer_6_case(white_visible=True)
    new, _ = mp.merge_plates(bricks, setting(), visible)
    cells = {(x, y): b.color for b in new for x, y in piece_cells(b)}
    check("visible White cells stay White", cells[(1, 0)] == WHITE and cells[(2, 0)] == WHITE, layout(new))


def test_non_rectangle_and_bricks_untouched():
    bricks = [PlacedBrick(0, 0, 0, 1, 1, 1, BLACK), PlacedBrick(1, 1, 0, 1, 1, 1, BLACK)]  # diagonal
    new, result = mp.merge_plates(bricks, setting())
    check("diagonal 1x1 pair not merged", result.merges == 0)
    bricks = [PlacedBrick(0, 0, 0, 1, 1, 3, BLACK), PlacedBrick(1, 0, 0, 1, 1, 1, BLACK)]
    new, result = mp.merge_plates(bricks, setting(shape=(3, 4, 4)))
    check("a brick is never merged with a plate", result.merges == 0)


def test_buying_never_grows():
    bricks = [PlacedBrick(0, 0, 0, 1, 1, 1, BLACK), PlacedBrick(0, 1, 0, 1, 1, 1, BLACK)]
    stock = Stock(have={make_key(1, 1, 1, BLACK): 2})  # both 1x1 owned, a 1x2 would be bought
    for piece in bricks:
        stock.take_brick(piece)
    new, result = mp.merge_plates(bricks, setting(stock=stock))
    check("owned pieces kept when the merged plate must be bought", result.merges == 0)
    stock = Stock(have={make_key(1, 1, 2, BLACK): 1})
    for piece in bricks:
        stock.take_brick(piece)
    new, result = mp.merge_plates(bricks, setting(stock=stock))
    check(
        "merged when it saves buying; ledger == pieces", result.merges == 1 and +stock.recount(new).used == +stock.used
    )


def test_weaker_result_rejected_and_base_skipped():
    calls = []

    def worse(_bricks):
        calls.append(1)
        return (0,) if len(calls) == 1 else (1,)

    bricks = [PlacedBrick(0, 0, 0, 1, 1, 1, BLACK), PlacedBrick(0, 1, 0, 1, 1, 1, BLACK)]
    _, result = mp.merge_plates(bricks, setting(worse))
    check("score worse -> not merged", result.merges == 0)
    _, result = mp.merge_plates(bricks, setting(), first_layer=1)
    check("layers below first_layer (base) untouched", result.merges == 0)


def test_pipeline_step_and_cli():
    grid = np.zeros((3, 4, 4), bool)
    grid[0, 0, 0:2] = True  # base plate 1x2
    grid[1, 0, 0:2] = True
    bricks = [
        PlacedBrick(0, 0, 0, 2, 1, 1, BLACK),
        PlacedBrick(0, 0, 1, 1, 1, 1, ORANGE),
        PlacedBrick(1, 0, 1, 1, 1, 1, ORANGE),
    ]
    with contextlib.redirect_stdout(io.StringIO()) as log:
        new = placement._merge_plates(bricks, placement.PassContext(grid, None, 1, 5))
    check("console line", "Merge plates: 1 group(s)" in log.getvalue() and len(new) == 2, log.getvalue())


if __name__ == "__main__":
    for test in [value for key, value in list(globals().items()) if key.startswith("test_")]:
        test()
    sys.exit(1 if failed else 0)
