"""--prefer-bricks (PackingOptions.prefer_bricks): bricks first over the whole figure, in 3 band shifts, bought even
when owned plates of that colour are left; plates only for the rest. Off = the old band packing."""

import contextlib
import io
import sys

import support

support.stub_mesh_libraries()
import numpy as np

from image_to_bricks.options import BuildOptions, PackingOptions, RepairOptions
from image_to_bricks.packing import placement
from image_to_bricks.stock import Stock, make_key

BLACK, YELLOW = 6, 5
failed = False


def check(name, ok, info=None):
    global failed
    print(("ok   " if ok else "FAIL ") + name + ("" if ok or info is None else f"  {info}"))
    failed |= not ok


def pack(grid, colours, prefer_bricks, stock=None):
    """Pack with tie repair and every later step off; returns the pieces (exact cover is asserted)."""
    options = BuildOptions(packing=PackingOptions(prefer_bricks=prefer_bricks), repair=RepairOptions(tie_repair=False))
    with contextlib.redirect_stdout(io.StringIO()):
        model = placement.optimize_brick_placement(
            placement.PackingCells(grid, colours, visible=grid.copy()), options=options, stock=stock
        )
    cover = np.zeros(grid.shape, int)
    for piece in model.bricks:
        cover[piece.z : piece.z + piece.h_plates, piece.y : piece.y + piece.d, piece.x : piece.x + piece.w] += 1
    assert (cover == grid.astype(int)).all(), "exact cover"
    return model.bricks


def count_by_height(pieces):
    return sum(p.h_plates == 3 for p in pieces), sum(p.h_plates == 1 for p in pieces)


def colours_of(pieces, shape):
    painted = np.zeros(shape, np.uint8)
    for p in pieces:
        painted[p.z : p.z + p.h_plates, p.y : p.y + p.d, p.x : p.x + p.w] = p.color
    return painted


def test_owned_plates_no_longer_block_bricks():
    grid = np.ones((3, 2, 4), bool)
    colours = np.full(grid.shape, BLACK, np.uint8)
    owned_plates = Stock({make_key(1, 4, 2, BLACK): 3})  # exactly enough plates for the block
    default = pack(grid, colours, False, Stock({make_key(1, 4, 2, BLACK): 3}))
    preferred = pack(grid, colours, True, owned_plates)
    check("default: owned plates first, no bought brick", count_by_height(default) == (0, 3), count_by_height(default))
    check("prefer bricks: one bought 4x2 brick", count_by_height(preferred) == (1, 0), count_by_height(preferred))
    check(
        "ledger follows: the brick is taken, plates stay unused",
        owned_plates.used.get(make_key(3, 4, 2, BLACK), 0) == 1
        and owned_plates.used.get(make_key(1, 4, 2, BLACK), 0) == 0,
    )


def test_bricks_fit_off_the_fixed_bands():
    grid = np.zeros((4, 2, 2), bool)
    grid[1:4] = True  # layers 1-3: off the 0-2 / 3-5 bands
    colours = np.full(grid.shape, YELLOW, np.uint8)
    check("default: no brick fits the fixed bands", count_by_height(pack(grid, colours, False))[0] == 0)
    preferred = pack(grid, colours, True)
    check(
        "prefer bricks: one brick on the shifted band (layers 1-3)",
        count_by_height(preferred) == (1, 0) and preferred[0].z == 1,
        [(p.z, p.h_plates) for p in preferred],
    )


def test_colours_kept_and_fewer_plates():
    grid = np.ones((7, 4, 6), bool)
    colours = np.full(grid.shape, BLACK, np.uint8)
    colours[2:5, :, 3:] = YELLOW  # a yellow patch across the band border
    default, preferred = pack(grid, colours, False), pack(grid, colours, True)
    check("prefer bricks: every cell keeps its colour", (colours_of(preferred, grid.shape) == colours).all())
    check(
        "prefer bricks: fewer plates and more bricks than default",
        count_by_height(preferred)[1] < count_by_height(default)[1]
        and count_by_height(preferred)[0] >= count_by_height(default)[0],
        (count_by_height(default), count_by_height(preferred)),
    )


def test_off_by_default_and_cli():
    check("off by default", PackingOptions().prefer_bricks is False)


if __name__ == "__main__":
    for test in [value for key, value in list(globals().items()) if key.startswith("test_")]:
        test()
    sys.exit(1 if failed else 0)
