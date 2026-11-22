"""--thicken-sheet-box: a one-plate-thick sheet (hat brim) gets a second plate so its ring is no longer floating."""

import contextlib
import io
import sys

import support

support.stub_mesh_libraries()
import numpy as np

from image_to_bricks.analysis import stability
from image_to_bricks.mesh import shape_edits, voxelize
from image_to_bricks.options import BuildOptions, RepairOptions, ShapeOptions, scale_boxes
from image_to_bricks.packing import placement

failed = False


def check(name, ok, info=None):
    global failed
    print(("ok   " if ok else "FAIL ") + name + ("" if ok or info is None else f"  {info}"))
    failed |= not ok


def hat():
    g = np.zeros((16, 20, 20), bool)
    g[0:14, 7:13, 7:13] = True  # crown 6x6
    g[6, 2:18, 2:18] = True  # brim: one plate thick, 16 x 16
    return g


S = slice
g = hat()
g2, n = shape_edits.thicken_sheets(g, (S(0, 16), S(0, 20), S(0, 20)), 1)
check("only the brim ring gets cells (16*16 - 6*6 = 220)", n == 220, n)
check("the plate is above the sheet", g2[7, 2, 2] and not g2[5, 2, 2])
check("crown unchanged", (g2[:, 7:13, 7:13] == g[:, 7:13, 7:13]).all())
g3, n3 = shape_edits.thicken_sheets(g, (S(0, 16), S(0, 20), S(0, 20)), -1)
check("direction down", g3[5, 2, 2] and not g3[7, 2, 2] and n3 == 220)
g4, n4 = shape_edits.thicken_sheets(g, (S(0, 4), S(0, 20), S(0, 20)), 1)
check("box excludes the brim: nothing added", n4 == 0)
check("original not modified", g.sum() == hat().sum())


def build(grid):
    c = np.full(grid.shape, 6, np.uint8)
    with contextlib.redirect_stdout(io.StringIO()):
        m = placement.optimize_brick_placement(
            placement.PackingCells(grid, c, visible=voxelize.surface_mask(grid), base_layers=0, fill=None),
            options=BuildOptions(repair=RepairOptions(tie_max_recoloured_cells=10, tie_drop_max_pieces=6)),
        )
    return m


m1 = build(g)
m2 = build(g2)
lost1 = int(g.sum() - m1.grid.sum())
lost2 = int(g2.sum() - m2.grid.sum())
with contextlib.redirect_stdout(io.StringIO()):
    r1 = stability.report(m1, None, "a", verbose=False)
    r2 = stability.report(m2, None, "b", verbose=False)
print("brim cells lost without / with sheet thickening:", lost1, lost2)
check("without thickening the floating brim ring is removed (cells lost)", lost1 > 0, lost1)
check("with a second plate nothing is lost", lost2 == 0, lost2)
check(
    "with a second plate the model is one assembly",
    r2["components"] == 1 and r2["disconnected_pieces"] == 0,
    r2["components"],
)

scaled = scale_boxes(BuildOptions(shape=ShapeOptions(thicken_sheet_boxes=[(S(0, 12), S(0, 10), S(0, 10))])), 0.5)
check("--boxes-for-studs scales the sheet boxes", scaled.shape.thicken_sheet_boxes[0][0] == S(0, 6))
print("FAILED" if failed else "all passed")
sys.exit(1 if failed else 0)
