"""--base-zone: pond / path colours on the visible base; base columns keep one colour over all base layers;
visible base cells are built in their zone colour (pieces never mix zones over visible cells)."""

import contextlib
import io
import sys

import support

support.stub_mesh_libraries()
import numpy as np

from image_to_bricks.base import stand
from image_to_bricks.mesh import voxelize
from image_to_bricks.options import BaseOptions, BuildOptions, RepairOptions
from image_to_bricks.packing import placement

GREEN, BLUE, BROWN = 3, 21, 11
failed = False


def check(name, ok, info=None):
    global failed
    print(("ok   " if ok else "FAIL ") + name + ("" if ok or info is None else f"  {info}"))
    failed |= not ok


def figure():
    g = np.zeros((20, 14, 20), bool)
    g[0:20, 5:9, 6:10] = True  # a 4x4 tower in the middle
    c = np.full(g.shape, 9, np.uint8)
    return g, c


def base_options(zones, seed=1, margin=6, plates=2):
    return BaseOptions(
        plates=plates,
        margin_studs=margin,
        shape="rect",
        colour_id=GREEN,
        zones=zones,
        zone_seed=seed,
        footprint_layers=3,
    )


def base(zones, seed=1, margin=6, plates=2):
    g, c = figure()
    with contextlib.redirect_stdout(io.StringIO()):
        g2, c2, v2, mask = stand.add_base(g, c, voxelize.surface_mask(g), base_options(zones, seed, margin, plates))
    return g2, c2, v2, mask


# no zones: plain
g2, c2, v2, mask = base([])
check("no zones: whole base in --base-color", (c2[:2][:, mask] == GREEN).all())

zones = [(BLUE, "pond", "right", 20), (BROWN, "path", "down", 3)]
g2, c2, v2, mask = base(zones)
top = c2[1]
check("pond and path present", (top == BLUE).any() and (top == BROWN).any())
check("zones only on the base outline", ((top != 0) <= mask).all() and (c2[:2][:, ~mask] == 0).all())
check("every base column has one colour over all base layers", (c2[0] == c2[1]).all())
ys, xs = np.nonzero(mask)
cx = (xs.min() + xs.max()) / 2
bx = np.nonzero(top == BLUE)[1]
check("pond lies on the right half", bx.mean() > cx, (bx.mean(), cx))
area = (top == BLUE).sum() / mask.sum()
check("pond area within 40..130 %% of the request", 0.08 <= area <= 0.26, area)
by = np.nonzero(top == BROWN)[0]
check("path reaches the low-y edge", by.min() <= ys.min() + 1, (by.min(), ys.min()))
check("path width about 3", 2 <= (top[:, :] == BROWN).sum(axis=1).max() <= 6)
g3, c3, _, _ = base(zones)
check("deterministic for a seed", (c3 == c2).all())
g4, c4, _, _ = base(zones, seed=2)
check("another seed gives another shape", not (c4 == c2).all())
check("grid unchanged by zones", (g2 == base([])[0]).all())

# build: visible base cells keep their zone colour, hidden ones are free
g2, c2, v2, mask = base(zones)
with contextlib.redirect_stdout(io.StringIO()):
    mdl = placement.optimize_brick_placement(
        placement.PackingCells(g2, c2, visible=v2, base_layers=2, fill=None),
        options=BuildOptions(base=base_options(zones), repair=RepairOptions(tie_repair=False)),
    )
bad = 0
for b in mdl.bricks:
    if b.z >= 2:
        continue
    cells = v2[b.z, b.y : b.y + b.d, b.x : b.x + b.w]
    want = c2[b.z, b.y : b.y + b.d, b.x : b.x + b.w]
    if cells.any() and not (want[cells] == b.color).all():
        bad += 1
check("pieces over visible base cells have exactly their zone colour", bad == 0, bad)
cov = np.zeros(g2.shape, int)
for b in mdl.bricks:
    cov[b.z : b.z + b.h_plates, b.y : b.y + b.d, b.x : b.x + b.w] += 1
check("exact cover", (cov[g2] == 1).all() and (cov[~g2] == 0).all())

# with a stock: hidden base cells may use any owned colour, visible ones still keep the zone colour
from image_to_bricks.stock import Stock, make_key

have = {
    make_key(h, w, d, col): 30
    for col in (6, 5, GREEN, BLUE)
    for h in (1, 3)
    for (w, d) in [(1, 1), (2, 1), (2, 2), (4, 2), (4, 4), (8, 4)]
}
st = Stock(have)
g2, c2, v2, mask = base(zones)
with contextlib.redirect_stdout(io.StringIO()):
    mdl = placement.optimize_brick_placement(
        placement.PackingCells(g2, c2, visible=v2, base_layers=2, fill=None),
        options=BuildOptions(base=base_options(zones), repair=RepairOptions(tie_repair=False)),
        stock=st,
    )
bad = 0
for b in mdl.bricks:
    if b.z >= 2:
        continue
    cells = v2[b.z, b.y : b.y + b.d, b.x : b.x + b.w]
    want = c2[b.z, b.y : b.y + b.d, b.x : b.x + b.w]
    if cells.any() and not (want[cells] == b.color).all():
        bad += 1
cols = {b.color for b in mdl.bricks if b.z < 2}
check("stock: visible cells keep the zone colour", bad == 0, bad)
check("stock: hidden cells use other owned colours too", len(cols - {GREEN, BLUE, BROWN}) > 0, cols)

print("FAILED" if failed else "all passed")
sys.exit(1 if failed else 0)
