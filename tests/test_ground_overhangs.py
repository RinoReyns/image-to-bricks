"""--ground-overhangs (ground_overhangs.py): a tall wall column that starts 1-2 plates in the air is extended down;
short specks and long overhangs (brim, tail) stay; new cells take the column's colour in the pipeline step."""

import contextlib
import io
import os
import sys

import support

support.stub_mesh_libraries()
import numpy as np

from image_to_bricks.mesh import ground_overhangs as go
from image_to_bricks.mesh import shape_edits
from image_to_bricks.options import ShapeOptions

BLACK, ORANGE = 6, 9
failed = False


def check(name, ok, info=None):
    global failed
    print(("ok   " if ok else "FAIL ") + name + ("" if ok or info is None else f"  {info}"))
    failed |= not ok


def boot():
    """A 2 x 2 boot column z 0-9 at x 0-1, a wall column at x 2 starting at z 1 (one plate in the air, 9 tall),
    a 2-plate speck at x 3 starting at z 1, and a brim cell at x 4 starting at z 6 (6 plates in the air)."""
    grid = np.zeros((10, 2, 5), bool)
    grid[:, :, 0:2] = True
    grid[1:, 0, 2] = True
    grid[1:3, 0, 3] = True
    grid[6:8, 0, 4] = True
    return grid


def test_tall_column_one_plate_up_is_grounded():
    grid = boot()
    out, fills = go.ground_overhangs(grid, 2)
    check("wall column at x 2 reaches z 0", out[0, 0, 2] and (0, 0, 2, 1) in fills, fills)
    check("2-plate speck (shorter than 3) left alone", not out[0, 0, 3])
    check("brim 6 plates up left alone", not out[:6, 0, 4].any())
    check("nothing removed", (out | ~grid).all())


def test_gap_limit_and_column_limit():
    grid = boot()
    out, _ = go.ground_overhangs(grid, 6, min_column=2)
    check("gap limit 6 and min column 2 also ground the brim", out[:6, 0, 4].all(), out[:, 0, 4])
    out, _ = go.ground_overhangs(grid, 2, min_column=2)
    check("min column 2 grounds the speck", out[0, 0, 3])


def test_mid_air_dent_is_closed():
    grid = np.zeros((8, 1, 1), bool)
    grid[0:3, 0, 0] = True
    grid[5:8, 0, 0] = True  # 2-plate gap between two parts of one column
    out, fills = go.ground_overhangs(grid, 2)
    check("2-plate gap closed", out[:, 0, 0].all() and len(fills) == 2, fills)


def test_pipeline_step_uses_the_column_colour():
    grid = boot()
    lego = np.where(grid, ORANGE, 0).astype(np.uint8)
    lego[1:, 0, 2] = BLACK  # the wall column is black, the boot beside it orange
    shape = ShapeOptions(ground_overhangs_max_plates=2, ground_min_column_plates=3)
    with contextlib.redirect_stdout(io.StringIO()) as log:
        new_grid, new_lego = shape_edits.fix_gaps_and_pinholes(grid, lego, shape)
    check("new cell takes the column colour, not the neighbour's", new_grid[0, 0, 2] and new_lego[0, 0, 2] == BLACK)
    check("console line", "Ground overhangs: +1 cells" in log.getvalue(), log.getvalue())
    source = open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "image_to_bricks", "cli.py"), encoding="utf-8"
    ).read()
    check("CLI options", '"--ground-overhangs"' in source and '"--ground-min-column"' in source)


if __name__ == "__main__":
    for test in [value for key, value in list(globals().items()) if key.startswith("test_")]:
        test()
    sys.exit(1 if failed else 0)
