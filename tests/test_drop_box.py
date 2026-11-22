"""--drop-box (drop_cells.py): filled cells in a box of the figure grid are left out, after every automatic shape
fix (so ground overhangs cannot put them back); colours of removed cells are cleared; empty box = WARNING."""

import contextlib
import io
import sys

import support

support.stub_mesh_libraries()
import numpy as np

from image_to_bricks.mesh import drop_cells, shape_edits
from image_to_bricks.options import ShapeOptions

BLACK = 6
failed = False


def check(name, ok, info=None):
    global failed
    print(("ok   " if ok else "FAIL ") + name + ("" if ok or info is None else f"  {info}"))
    failed |= not ok


def wall_with_nub():
    """Wall x 0..3 on layers 0..5, plus a 1-cell bump at (z 2..4, y 1, x 1) hanging over air."""
    grid = np.zeros((6, 3, 4), bool)
    grid[:, 0, :] = True
    grid[2:5, 1, 1] = True
    return grid


def test_only_cells_in_the_box_are_removed():
    grid = wall_with_nub()
    new_grid, removed = drop_cells.drop_cells_in_boxes(grid, [(slice(2, 3), slice(1, 2), slice(1, 2))])
    check("the lowest bump cell is gone", removed == [[(2, 1, 1)]] and not new_grid[2, 1, 1], removed)
    check("everything else unchanged", (new_grid ^ grid).sum() == 1)
    check("input grid not changed in place", grid[2, 1, 1])


def test_box_over_empty_cells_removes_nothing():
    _, removed = drop_cells.drop_cells_in_boxes(wall_with_nub(), [(slice(0, 1), slice(2, 3), slice(0, 4))])
    check("empty box: nothing removed", removed == [[]])


def run_fix_step(grid, boxes, ground_overhangs=0):
    shape = ShapeOptions(drop_boxes=boxes, ground_overhangs_max_plates=ground_overhangs)
    lego = np.where(grid, BLACK, 0).astype(np.uint8)
    with contextlib.redirect_stdout(io.StringIO()) as log:
        new_grid, new_lego = shape_edits.fix_gaps_and_pinholes(grid, lego, shape)
    return new_grid, new_lego, log.getvalue()


def test_pipeline_step_clears_colour_and_prints():
    grid = wall_with_nub()
    new_grid, new_lego, log = run_fix_step(grid, [(slice(2, 3), slice(1, 2), slice(1, 2))])
    check("cell removed and its colour cleared", not new_grid[2, 1, 1] and new_lego[2, 1, 1] == 0)
    check("console line", "Drop box 2:3,1:2,1:2: 1 cell(s) removed (z, y, x): [(2, 1, 1)]" in log, log)
    _, _, log = run_fix_step(grid, [(slice(0, 1), slice(2, 3), slice(0, 4))])
    check("WARNING for a box with no filled cell", "WARNING Drop box 0:1,2:3,0:4" in log, log)


def test_drop_wins_over_ground_overhangs():
    grid = wall_with_nub()
    grid[2:6, 1, 1] = True  # 4-plate column starting 2 plates in the air
    filled, _, _ = run_fix_step(grid, [], ground_overhangs=2)
    dropped, _, _ = run_fix_step(grid, [(slice(0, 2), slice(1, 2), slice(1, 2))], ground_overhangs=2)
    check("ground overhangs alone extend the column down", filled[0, 1, 1] and filled[1, 1, 1])
    check("a drop box over those cells keeps them out", not dropped[0, 1, 1] and not dropped[1, 1, 1])
    check("no drop box and no other fix: grid untouched", (run_fix_step(grid, [])[0] == grid).all())


if __name__ == "__main__":
    for test in [value for key, value in list(globals().items()) if key.startswith("test_")]:
        test()
    sys.exit(1 if failed else 0)
