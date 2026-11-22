import support

support.stub_mesh_libraries()
import numpy as np

from image_to_bricks.analysis import stability
from image_to_bricks.base import stand
from image_to_bricks.model import LegoModel
from image_to_bricks.model import PlacedBrick as P
from image_to_bricks.options import BaseOptions
from image_to_bricks.packing import tie_repair

# 1. single-stud footing: a 1x1 on the base next to a 3x1 of the same colour gets merged into a 4x1
bricks = [
    P(0, 0, 0, 12, 6, 1, 8),
    P(0, 0, 1, 12, 6, 1, 8),
    P(3, 3, 2, 1, 1, 1, 6),
    P(4, 3, 2, 3, 1, 1, 6),
    P(3, 3, 3, 4, 1, 1, 6),
]
grid = np.zeros((4, 6, 12), bool)
for b in bricks:
    grid[b.z : b.z + b.h_plates, b.y : b.y + b.d, b.x : b.x + b.w] = True
m0 = LegoModel(grid=grid, bricks=bricks, base_layers=2)
r0 = stability.report(m0, verbose=False)
assert r0["single_stud_footings"] == 1
new, g2 = tie_repair.repair(tie_repair.RepairTarget(grid, base_layers=2), bricks, verbose=False)
m1 = LegoModel(grid=g2, bricks=new, base_layers=2)
r1 = stability.report(m1, verbose=False)
cov = np.zeros(g2.shape, int)
for b in new:
    cov[b.z : b.z + b.h_plates, b.y : b.y + b.d, b.x : b.x + b.w] += 1
assert (cov == g2.astype(int)).all() and r1["single_stud_footings"] == 0, r1
print("footing fixed:", r0["single_stud_footings"], "->", r1["single_stud_footings"])

# 2. auto margin: tall thin tower needs a wide base; chosen margin is the smallest that works
g = np.zeros((120, 6, 6), bool)
g[:, 2:4, 2:4] = True
base = BaseOptions(plates=2, shape="rect", corner_radius_studs=3, footprint_layers=3, min_tip_angle_degrees=15.0)
import contextlib
import io
from dataclasses import replace

with contextlib.redirect_stdout(io.StringIO()):
    chosen = stand.choose_base_margin(g, base)


def tip(margin):
    with contextlib.redirect_stdout(io.StringIO()):
        g2, _, _, _ = stand.add_base(
            g, np.zeros(g.shape, np.uint8), np.zeros(g.shape, bool), replace(base, margin_studs=margin)
        )
    return stability.stability_check(LegoModel(grid=g2, bricks=[], base_layers=2), 15.0)


assert tip(chosen)["ok"] and not tip(chosen - 1)["ok"], chosen
print("auto margin for a tall tower:", chosen, "studs, tip angle", round(tip(chosen)["tip_angle"], 1))
print("OK")
