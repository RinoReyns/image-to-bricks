import os
import tempfile

import support

support.stub_mesh_libraries()
from dataclasses import replace

import numpy as np

from image_to_bricks.base import stand
from image_to_bricks.mesh import voxelize
from image_to_bricks.options import BaseOptions, BuildOptions, OutputOptions, RepairOptions
from image_to_bricks.packing import placement

NO_TIES = BuildOptions(repair=RepairOptions(tie_repair=False))
from image_to_bricks.instructions.outputs_handler import save_outputs


def synthetic():
    nz, ny, nx = 30, 16, 26
    g = np.zeros((nz, ny, nx), bool)
    c = np.full(g.shape, 9, np.uint8)
    g[0:8, 4:10, 3:9] = True
    g[0:8, 4:10, 15:21] = True  # two feet/legs
    g[8:20, 3:11, 3:21] = True  # body
    g[20:26, 5:9, 8:16] = True  # head
    g[14:16, 6:8, 21:26] = True  # sword sticking out sideways
    c[0:6] = 11
    c[8:14] = 5
    c[20:26] = 6
    c[14:16, :, 21:] = 7
    c[2, 5, 4] = 6  # speckle
    return g, c


def check(model, grid):
    cover = np.zeros(grid.shape, int)
    for b in model.bricks:
        cover[b.z : b.z + b.h_plates, b.y : b.y + b.d, b.x : b.x + b.w] += 1
    assert (cover == grid.astype(int)).all(), "coverage mismatch"


g, c = synthetic()
solid = g.copy()
vis = voxelize.surface_mask(solid)
hol = voxelize.voxels_to_lego_units(solid, hollow=True, shell_thickness_studs=2)
base_model = placement.optimize_brick_placement(placement.PackingCells(hol, c, visible=vis & hol), options=NO_TIES)
check(base_model, hol)
n0 = len(base_model.bricks)
sig0 = sorted((b.x, b.y, b.z, b.w, b.d, b.h_plates, b.color) for b in base_model.bricks)

for shape in ("rect", "rounded"):
    for n in (1, 2, 3):
        base = BaseOptions(plates=n, shape=shape)
        v = vis & hol
        g2, c2, v2, mask = stand.add_base(hol, c, v, base)
        m = placement.optimize_brick_placement(
            placement.PackingCells(g2, c2, visible=v2, base_layers=n), options=replace(NO_TIES, base=base)
        )
        check(m, g2)
        mm = base.margin_studs
        fig = sorted((b.x - mm, b.y - mm, b.z - n, b.w, b.d, b.h_plates, b.color) for b in m.bricks if b.z >= n)
        assert fig == sig0, "figure pieces changed by base"
        base = [b for b in m.bricks if b.z < n]
        assert all(b.color == 8 and b.h_plates == 1 for b in base)
        # base must fully support the lowest figure layer
        assert (mask >= g2[n]).all()
        # rounded corners really cut
        if shape == "rounded":
            assert mask.sum() < (mask.any(0).sum() and np.ptp(np.nonzero(mask)[0]) + 1) * (
                np.ptp(np.nonzero(mask)[1]) + 1
            )
        print(shape, n, "base pieces", len(base), "figure pieces", len(m.bricks) - len(base), "(no base:", n0, ")")
        last = m
# outputs + pdf
out = tempfile.mkdtemp(prefix="bricks_test_base_")  # never inside the project folder (it is a git repo)
save_outputs(last, "synthetic", out, options=OutputOptions(make_pdf=True))
print(open(os.path.join(out, "synthetic_inventory.csv")).read()[:600])
print(os.listdir(out))
