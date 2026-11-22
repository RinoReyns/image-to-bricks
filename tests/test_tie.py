import contextlib
import io

import support

support.stub_mesh_libraries()
import numpy as np

from image_to_bricks.analysis import stability
from image_to_bricks.base import stand
from image_to_bricks.mesh import voxelize
from image_to_bricks.options import BaseOptions, BuildOptions, RepairOptions
from image_to_bricks.packing import placement


def synthetic():
    nz, ny, nx = 30, 16, 26
    g = np.zeros((nz, ny, nx), bool)
    c = np.full(g.shape, 9, np.uint8)
    g[0:8, 4:10, 3:9] = True
    g[0:8, 4:10, 15:21] = True
    g[8:20, 3:11, 3:21] = True
    g[20:26, 5:9, 8:16] = True
    g[14:16, 6:8, 21:26] = True
    c[0:6] = 11
    c[8:14] = 5
    c[20:26] = 6
    c[14:16, :, 21:] = 7
    return g, c


def run(recolor, drop=6, ties=True, grow=True):
    g, c = synthetic()
    vis = voxelize.surface_mask(g)
    hol = voxelize.voxels_to_lego_units(g, hollow=True, shell_thickness_studs=2)
    base = BaseOptions(plates=2, shape="rect")
    fill0 = g & ~hol
    g2, c2, v2, _ = stand.add_base(hol, c, vis & hol, base)
    fill = np.zeros(g2.shape, bool)
    fill[2:, 3 : 3 + g.shape[1], 3 : 3 + g.shape[2]] = fill0
    m = placement.optimize_brick_placement(
        placement.PackingCells(g2, c2, visible=v2, base_layers=2, fill=fill if grow else None),
        options=BuildOptions(
            base=base, repair=RepairOptions(tie_repair=ties, tie_max_recoloured_cells=recolor, tie_drop_max_pieces=drop)
        ),
    )
    cov = np.zeros(m.grid.shape, int)
    for b in m.bricks:
        cov[b.z : b.z + b.h_plates, b.y : b.y + b.d, b.x : b.x + b.w] += 1
    assert (cov == m.grid.astype(int)).all(), "coverage mismatch"
    r = stability.report(m, verbose=False)
    return m, r, g2


with contextlib.redirect_stdout(io.StringIO()):
    m0, r0, g0 = run(4, ties=False)
    m1, r1, g1 = run(4)
    m2, r2, g2 = run(0)
    m3, r3, g3 = run(4, drop=0)
    m4, r4, g4 = run(4, grow=False)
print("no ties : pieces", len(m0.bricks), "disconnected pieces", r0["disconnected_pieces"], "weak", r0["weak_joints"])
print(
    "ties r=4: pieces",
    len(m1.bricks),
    "disconnected pieces",
    r1["disconnected_pieces"],
    "weak",
    r1["weak_joints"],
    "cells removed",
    int(g1.sum() - m1.grid.sum()),
)
print(
    "ties r=0: pieces",
    len(m2.bricks),
    "disconnected pieces",
    r2["disconnected_pieces"],
    "weak",
    r2["weak_joints"],
    "cells removed",
    int(g2.sum() - m2.grid.sum()),
)
print("ties nodrop:", "disconnected pieces", r3["disconnected_pieces"], "cells removed", int(g3.sum() - m3.grid.sum()))
print(
    "ties no growth:", "disconnected pieces", r4["disconnected_pieces"], "cells removed", int(g4.sum() - m4.grid.sum())
)
assert r1["disconnected_pieces"] < r0["disconnected_pieces"]
assert r1["disconnected_pieces"] <= r4["disconnected_pieces"]
assert int(m3.grid.sum()) >= int(g3.sum())  # drop=0: nothing removed (growth may add hidden cells)
assert r1["disconnected_pieces"] == 0
print("OK")
