"""--front-oval (fur_ovals.py): concentric oval bands on the front fur, middle colour first; only fur cells (one of the
band colours) change, the wall behind each front cell gets the same colour; the side the face looks to (--front)
decides which cells are the front."""

import sys

import support

support.stub_mesh_libraries()
import numpy as np

from image_to_bricks.colours import fur_ovals

YELLOW, ORANGE, DARK_ORANGE, BLACK = 5, 9, 19, 6
failed = False


def check(name, ok, info=None):
    global failed
    print(("ok   " if ok else "FAIL ") + name + ("" if ok or info is None else f"  {info}"))
    failed |= not ok


def block():
    """Solid block z 0..8, y 0..3, x 0..8, all Dark Orange fur, with a Black belt row at z 4."""
    grid = np.ones((9, 4, 9), bool)
    colours = np.full(grid.shape, DARK_ORANGE, np.uint8)
    colours[4] = BLACK
    return grid, colours


OVAL = fur_ovals.FrontOval(0, 9, 0, 9, (YELLOW, ORANGE, DARK_ORANGE))


def test_bands_go_from_the_middle_outwards():
    check("middle cell = first colour", fur_ovals.band_index(4, 4, OVAL) == 0)
    check("rim cell = last colour", fur_ovals.band_index(0, 4, OVAL) == 2 and fur_ovals.band_index(4, 0, OVAL) == 2)
    check("box corner is outside the oval", fur_ovals.band_index(0, 0, OVAL) is None)


def test_front_wall_painted_and_belt_kept():
    grid, colours = block()
    painted, counts = fur_ovals.apply_front_ovals(colours, grid, [OVAL], "-y", 2)
    check("front cell in the middle row above the belt gets the middle colour", painted[3, 0, 4] == YELLOW)
    check("wall cell behind it too (2 cells deep)", painted[3, 1, 4] == YELLOW and painted[3, 2, 4] == DARK_ORANGE)
    check("black belt not repainted", (painted[4] == BLACK).all())
    check("corner outside the oval unchanged", painted[0, 0, 0] == DARK_ORANGE)
    check("input not changed in place", (colours[3, 0, 4] == DARK_ORANGE))


def test_front_side_decides_the_painted_face():
    grid, colours = block()
    painted, _ = fur_ovals.apply_front_ovals(colours, grid, [OVAL], "+y", 1)
    check("+y front: the back face (high y) is painted", painted[3, 3, 4] == YELLOW and painted[3, 0, 4] == DARK_ORANGE)
    painted, _ = fur_ovals.apply_front_ovals(
        colours, grid, [fur_ovals.FrontOval(0, 9, 0, 4, (YELLOW, ORANGE, DARK_ORANGE))], "-x", 1
    )
    check(
        "-x front: horizontal axis is y, low-x face painted",
        painted[4 - 1, 1, 0] in (YELLOW, ORANGE) and painted[3, 1, 1] == DARK_ORANGE,
    )


if __name__ == "__main__":
    for test in [value for key, value in list(globals().items()) if key.startswith("test_")]:
        test()
    sys.exit(1 if failed else 0)
