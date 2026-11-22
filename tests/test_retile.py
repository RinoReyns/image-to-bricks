"""--retile: re-tiling all pieces of one layer around a single-stud joint (a 2-wide plate spans both sides).
Synthetic tail root, same shape as the real one: a 2x2 tail hangs under a diagonal step, the pair-wise ties
cannot fix it (the best 2x2 leaves a 1x1 orphan)."""

import sys

import support

support.stub_mesh_libraries()
import numpy as np

from image_to_bricks.model import PlacedBrick as PB
from image_to_bricks.options import RepairOptions
from image_to_bricks.packing import assembly, joint_retile
from image_to_bricks.packing import tie_repair as T
from image_to_bricks.stock import Stock, make_key

NZ, NY, NX = 16, 26, 26


def model():
    b = []
    for z in range(0, 5):  # tail: 2x2 cross-section, two 1x2 plates per layer
        b += [PB(21, 20, z, 1, 2, 1, 5), PB(22, 20, z, 1, 2, 1, 5)]
    b += [PB(19, 20, 5, 3, 1, 1, 5), PB(20, 21, 5, 2, 1, 1, 5), PB(22, 19, 5, 1, 3, 1, 5)]  # step layer
    for z in range(6, 12):  # body above: 4x2 plates, stacked
        b.append(PB(20, 19, z, 2, 2, 1, 5))
    g = np.zeros((NZ, NY, NX), bool)
    for p in b:
        g[p.z : p.z + p.h_plates, p.y : p.y + p.d, p.x : p.x + p.w] = True
    return g, b


def cover(bricks):
    c = np.zeros((NZ, NY, NX), int)
    for p in bricks:
        c[p.z : p.z + p.h_plates, p.y : p.y + p.d, p.x : p.x + p.w] += 1
    return c


def weak(bricks, hang_min=3):
    asm = assembly.Assembly((NZ, NY, NX), bricks)
    return [t for t in assembly.find_issues(asm, hang_min)[2] if t.kind == "weak"]


def run(retile_box=None, **options):
    g, b = model()
    settings = RepairOptions(tie_max_recoloured_cells=0, tie_min_hanging_pieces=3, tie_drop_max_pieces=6, **options)
    return T.repair(T.RepairTarget(g, retile_box=retile_box), b, settings, verbose=False), g


def check(name, ok):
    print(("ok   " if ok else "FAIL ") + name)
    if not ok:
        check.failed = True


check.failed = False

g0, b0 = model()
check("setup: the tail root hangs by one stud", len(weak(b0)) == 1)

# 1. ties alone cannot fix it
(b1, g1), _ = run(retile=False)
check("without --retile the weak joint stays", len(weak(b1)) == 1)

# 2. retile: fixed, at most the one orphan cell removed, nothing else moved
(b2, g2), _ = run(retile=True, retile_drop_max_pieces=1)
removed = g0 & ~g2
check("--retile fixes the joint", len(weak(b2)) == 0)
check("at most 1 cell removed", int(removed.sum()) <= 1)
c = cover(b2)
check("every kept cell covered exactly once", (c[g2] == 1).all() and (c[~g2] == 0).all())
check("grid == covered cells", (g2 == (c > 0)).all())
check("colours unchanged", all(p.color == 5 for p in b2))

# 3. no removal allowed: grid identical, never worse
(b3, g3), _ = run(retile=True, retile_drop_max_pieces=0)
check("--retile-drop 0 removes nothing", (g3 == g0).all())
check("--retile-drop 0 not worse", len(weak(b3)) <= 1)

# 4. box elsewhere: nothing changes
box_out = (0, 3, 0, 5, 0, 5)
(b4, g4), _ = run(retile=True, retile_box=box_out)
check(
    "box away from the joint: pieces unchanged",
    sorted((p.x, p.y, p.z, p.w, p.d) for p in b4) == sorted((p.x, p.y, p.z, p.w, p.d) for p in b1),
)
box_in = (4, 7, 19, 23, 19, 24)
(b5, _), _ = run(retile=True, retile_box=box_in)
check("box around the joint: fixed", len(weak(b5)) == 0)

# 5. sampler / colouring units
cells = {(x, y) for x in range(3) for y in range(2)}
til = joint_retile.sample_tilings(cells, assembly.SIZES, 50)
check("sampler: first tiling is the biggest piece", len(til[0]) == 1 or max(w * d for _, _, w, d in til[0]) >= 4)
check(
    "sampler: all tilings cover the cells exactly once",
    all(
        sorted(c for (x, y, w, d) in t for c in [(xx, yy) for yy in range(y, y + d) for xx in range(x, x + w)])
        == sorted(cells)
        for t in til
    ),
)
col = {c: (1 if c[0] < 2 else 2) for c in cells}
vis = dict.fromkeys(cells, True)
r = joint_retile.colour_tiling([(0, 0, 3, 2)], col, vis, 10)
check("colouring: majority colour, cost = other visible cells", r[0][0][4] == 1 and r[1] == 2)
check("colouring: over budget -> None", joint_retile.colour_tiling([(0, 0, 3, 2)], col, vis, 1) is None)
vh = {c: c[0] < 2 for c in cells}
r = joint_retile.colour_tiling([(0, 0, 3, 2)], col, vh, 0)
check("colouring: hidden cells are free", r is not None and r[1] == 0)

# 6. with a stock the ledger follows the re-tiling
have = {
    make_key(1, w, d, 5): 50
    for (w, d) in [(1, 1), (1, 2), (2, 1), (1, 3), (2, 2), (3, 1), (4, 2), (2, 4), (1, 4), (4, 1)]
}
g, b = model()
st = Stock(have)
st.used.update({})
for p in b:
    st.take_brick(p)
settings = RepairOptions(tie_max_recoloured_cells=0, tie_min_hanging_pieces=3, tie_drop_max_pieces=6)
b6, _ = T.repair(T.RepairTarget(g), b, settings, st, verbose=False)
check("stock: ledger equals the final pieces", st.used == st.recount(b6).used)
check("stock: joint fixed", len(weak(b6)) == 0)

print("FAILED" if check.failed else "all passed")
sys.exit(1 if check.failed else 0)
