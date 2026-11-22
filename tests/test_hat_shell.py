"""--join-corners (corner_joins.py) and --dark-enclaves (color_cleanup.dark_enclaves): a thin diagonal hat flap
becomes side-connected, real corner gaps stay open, small fur-coloured spots inside a black part become Black."""

import contextlib
import io
import sys

import support

support.stub_mesh_libraries()
import numpy as np
from scipy.ndimage import label

from image_to_bricks.colours import color_cleanup, corner_joins
from image_to_bricks.mesh import shape_edits, voxelize
from image_to_bricks.options import ShapeOptions

BLACK, YELLOW, ORANGE, RED = 6, 5, 9, 2
SIDE_CONNECTED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
failed = False


def check(name, ok, info=None):
    global failed
    print(("ok   " if ok else "FAIL ") + name + ("" if ok or info is None else f"  {info}"))
    failed |= not ok


def hat_with_diagonal_flap():
    """Crown 4 x 4 (z 0-5, y 0-3, x 0-3) and a one-cell flap running diagonally away from its corner:
    (y 4, x 4), (y 5, x 5), (y 6, x 6). In most layers the flap touches the crown and itself only at corners; in
    layers 1 and 4 a seam joins them (as the real tilted flap does every few layers), so it is one body."""
    grid = np.zeros((6, 8, 8), bool)
    grid[:, 0:4, 0:4] = True
    for step in range(4, 7):
        grid[:, step, step] = True
    for seam_layer in (1, 4):
        grid[seam_layer, 3, 4] = grid[seam_layer, 4, 5] = grid[seam_layer, 5, 6] = True
    return grid


def test_diagonal_flap_becomes_side_connected():
    grid = hat_with_diagonal_flap()
    joined, added = corner_joins.join_corners(grid, 3)
    parts_per_layer = [label(joined[z], structure=SIDE_CONNECTED)[1] for z in range(joined.shape[0])]
    check("flap + crown one side-connected part in every layer", parts_per_layer == [1] * 6, parts_per_layer)
    check(
        "3 cells added in each layer without a seam",
        len(added) == 12 and sorted({z for z, _, _ in added}) == [0, 2, 3, 5],
        added,
    )
    check("nothing removed", (joined | ~grid).all())


def test_flap_that_is_not_one_body_is_left_alone():
    grid = np.zeros((6, 8, 8), bool)
    grid[:, 0:4, 0:4] = True
    for step in range(4, 7):
        grid[:, step, step] = True  # no seam: touches the crown only at corners
    _, added = corner_joins.join_corners(grid, 3)
    check("corner-only contact everywhere (separate body) is not closed", added == [], added)


def test_two_big_parts_touching_at_a_corner_stay_apart():
    grid = np.zeros((4, 10, 10), bool)
    grid[:, 0:4, 0:4] = True  # boot 1
    grid[:, 4:8, 4:8] = True  # boot 2, touching boot 1 only at a corner
    grid[3, :, :8] = True  # one body above (hips)
    joined, added = corner_joins.join_corners(grid, 3)
    check("two 16-cell parts touching at a corner are not joined", added == [], added)


def test_separate_bodies_are_not_joined():
    lone = np.zeros((6, 6, 6), bool)
    lone[2, 0, 0] = True
    lone[2, 1, 1] = True  # two single cells, no body joins them nearby
    _, added = corner_joins.join_corners(lone, 3)
    check("cells that are not one body nearby are not joined", added == [], added)


def test_pipeline_step_colours_new_cells_and_is_off_by_default():
    grid = hat_with_diagonal_flap()
    lego = np.where(grid, BLACK, 0).astype(np.uint8)
    same_grid, same_lego = shape_edits.fix_gaps_and_pinholes(grid, lego, ShapeOptions())
    check("off: grid unchanged", (same_grid == grid).all() and (same_lego == lego).all())
    with contextlib.redirect_stdout(io.StringIO()) as log:
        new_grid, new_lego = shape_edits.fix_gaps_and_pinholes(grid, lego, ShapeOptions(join_corners_max_cells=3))
    added = new_grid & ~grid
    check("added cells take the neighbours' colour", added.any() and (new_lego[added] == BLACK).all())
    check("console line", "Join corners: +12 cells" in log.getvalue(), log.getvalue())


def black_wall_with_spots():
    """A 1 x 10 x 10 black wall (x 0, the surface) with a 2-cell yellow spot and a 6-cell orange patch."""
    grid = np.zeros((10, 10, 3), bool)
    grid[:, :, 0:2] = True
    lego = np.where(grid, BLACK, 0).astype(np.uint8)
    lego[4, 4:6, 0] = YELLOW
    lego[1:3, 6:9, 0] = ORANGE
    return grid, lego


def test_small_spot_inside_black_becomes_black():
    grid, lego = black_wall_with_spots()
    surface = voxelize.surface_mask(grid)
    nothing = np.zeros(grid.shape, bool)
    out, changed = color_cleanup.dark_enclaves(lego, surface, nothing, 5)
    check("2-cell yellow spot -> Black", (out[4, 4:6, 0] == BLACK).all())
    check("6-cell orange patch kept with N = 5", (out[1:3, 6:9, 0] == ORANGE).all(), changed)
    check("only the spot changed", changed == 2, changed)


def test_spot_next_to_hand_painted_cells_is_kept():
    grid, lego = black_wall_with_spots()
    surface = voxelize.surface_mask(grid)
    painted = np.zeros(grid.shape, bool)
    painted[5, 4, :] = True  # a painted pupil right above the spot
    out, _ = color_cleanup.dark_enclaves(lego, surface, painted, 5, painted)
    check("spot touching a painted cell is kept", (out[4, 4:6, 0] == YELLOW).all())
    locked = np.zeros(grid.shape, bool)
    locked[4, 4, 0] = True
    out, _ = color_cleanup.dark_enclaves(lego, surface, locked, 5)
    check("spot with a locked cell is kept", (out[4, 4:6, 0] == YELLOW).all())


def test_spot_in_colourful_part_is_kept():
    grid, lego = black_wall_with_spots()
    lego[grid] = ORANGE
    lego[4, 4:6, 0] = YELLOW
    out, changed = color_cleanup.dark_enclaves(lego, voxelize.surface_mask(grid), np.zeros(grid.shape, bool), 5)
    check("a spot inside orange fur is not touched", changed == 0, changed)


if __name__ == "__main__":
    for test in [value for key, value in list(globals().items()) if key.startswith("test_")]:
        test()
    sys.exit(1 if failed else 0)
