"""--stock: owned pieces first, size fallback, colours as hard requirements, base + tie repair respect stock.
Synthetic data, no mesh needed."""

import contextlib
import csv
import io
import os
import tempfile

import support

support.stub_mesh_libraries()
from dataclasses import replace

import numpy as np

from image_to_bricks.analysis import stability
from image_to_bricks.base import stand
from image_to_bricks.mesh import voxelize
from image_to_bricks.options import BaseOptions, BuildOptions, RepairOptions
from image_to_bricks.packing import placement
from image_to_bricks.stock import Stock, make_key

BLACK, ORANGE, YELLOW, GREY = 6, 9, 5, 8


def pack(g, c, stock, base=0, visible=None, ties=False):
    with contextlib.redirect_stdout(io.StringIO()):
        m = placement.optimize_brick_placement(
            placement.PackingCells(g, c, visible=visible, base_layers=base),
            options=BuildOptions(repair=RepairOptions(tie_repair=ties)),
            stock=stock,
        )
    cov = np.zeros(m.grid.shape, int)
    for b in m.bricks:
        cov[b.z : b.z + b.h_plates, b.y : b.y + b.d, b.x : b.x + b.w] += 1
    assert (cov == m.grid.astype(int)).all(), "coverage mismatch"
    return m


def stock_of(*items):
    """items: (h, w, d, colour, qty)"""
    return Stock({make_key(h, w, d, c): q for h, w, d, c, q in items})


def counts(m):
    out = {}
    for b in m.bricks:
        k = make_key(b.h_plates, b.w, b.d, b.color)
        out[k] = out.get(k, 0) + 1
    return out


def block(nz, ny, nx, colour):
    g = np.ones((nz, ny, nx), bool)
    return g, np.full(g.shape, colour, np.uint8)


# 1. a stock with plenty of every standard size changes nothing (same pieces as no stock)
g, c = block(6, 9, 11, BLACK)
c[:, :, 6:] = YELLOW
std = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4)]
rich = stock_of(*[(h, w, d, col, 999) for h in (1, 3) for (w, d) in std for col in (BLACK, YELLOW)])
a = pack(g, c, None)
b = pack(g, c, rich)
sig = lambda m: sorted((x.x, x.y, x.z, x.w, x.d, x.h_plates, x.color) for x in m.bricks)
assert sig(a) == sig(b), "plenty of stock must give the unlimited result"
print("1 plenty of stock == no stock:", len(a.bricks), "pieces")

# 2. colour not owned: still built in the required colour, everything is on the buy list
g, c = block(3, 4, 4, ORANGE)
st = stock_of((3, 2, 4, BLACK, 5), (1, 4, 4, BLACK, 5))
m = pack(g, c, st)
assert all(b.color == ORANGE for b in m.bricks)
fin = st.recount(m.bricks)
rows = [r for r in fin.rows() if r[4] == "Orange"]
assert rows and all(r[9] == "not owned" and r[5] == 0 and r[7] == r[6] for r in rows)
assert fin.used.get(make_key(3, 4, 2, BLACK), 0) == 0
print("2 not owned colour:", sum(r[7] for r in rows), "pieces to buy")

# 3. a size runs out -> smaller owned pieces; owned pieces are used before anything is bought
g, c = block(1, 4, 4, BLACK)
st = stock_of((1, 2, 1, BLACK, 4))  # four 1x2 plates = 8 of 16 cells
m = pack(g, c, st)
cnt = counts(m)
assert cnt.get(make_key(1, 2, 1, BLACK), 0) >= 4, cnt
fin = st.recount(m.bricks)
assert fin.have[make_key(1, 2, 1, BLACK)] <= fin.used[make_key(1, 2, 1, BLACK)], "owned pieces left unused"
print("3 fallback:", {k: v for k, v in cnt.items()})

# 4. bricks: not bought while the colour still has owned plates; bought for a colour with no stock
g, c = block(3, 4, 4, BLACK)
m = pack(g, c, stock_of((1, 4, 4, BLACK, 1), (1, 2, 2, BLACK, 4)))
assert all(b.h_plates == 1 for b in m.bricks), "must use owned plates, not buy bricks"
g, c = block(3, 4, 4, ORANGE)
m = pack(g, c, stock_of((1, 4, 4, BLACK, 1)))
assert all(b.h_plates == 3 for b in m.bricks), "colour with no stock: bricks are fine to buy"
g, c = block(3, 4, 4, BLACK)
m = pack(g, c, stock_of((3, 4, 4, BLACK, 1)))
assert len(m.bricks) == 1 and m.bricks[0].h_plates == 3
print("4 bricks vs plates ok")

# 5. owned big plates (8x4) are used although the unlimited packer never goes above 4x4
g, c = block(1, 8, 16, BLACK)
st = stock_of((1, 8, 4, BLACK, 2))
m = pack(g, c, st)
assert counts(m).get(make_key(1, 8, 4, BLACK)) == 2, counts(m)
print("5 big owned plates used:", counts(m)[make_key(1, 8, 4, BLACK)])


# 6. base + figure + tie repair with a tight stock: coverage exact, ledger == final pieces, report consistent
def synthetic():
    nz, ny, nx = 30, 16, 26
    g = np.zeros((nz, ny, nx), bool)
    c = np.full(g.shape, ORANGE, np.uint8)
    g[0:8, 4:10, 3:9] = True
    g[0:8, 4:10, 15:21] = True
    g[8:20, 3:11, 3:21] = True
    g[20:26, 5:9, 8:16] = True
    g[14:16, 6:8, 21:26] = True
    c[0:6] = 11
    c[8:14] = YELLOW
    c[20:26] = BLACK
    c[14:16, :, 21:] = 7
    return g, c


for label, stock_items in (
    (
        "tight",
        [
            (1, 4, 12, BLACK, 3),
            (1, 2, 4, BLACK, 6),
            (1, 1, 1, BLACK, 5),
            (3, 2, 4, BLACK, 4),
            (1, 4, 4, YELLOW, 3),
            (3, 2, 2, YELLOW, 4),
            (1, 2, 2, YELLOW, 5),
        ],
    ),
    ("empty-ish", [(1, 1, 1, GREY, 1)]),
):
    g, c = synthetic()
    vis = voxelize.surface_mask(g)
    hol = voxelize.voxels_to_lego_units(g, hollow=True, shell_thickness_studs=2)
    base = BaseOptions(plates=2, shape="rect", margin_studs=3, colour_id=BLACK)
    fill0 = g & ~hol
    g2, c2, v2, _ = stand.add_base(hol, c, vis & hol, base)
    fill = np.zeros(g2.shape, bool)
    fill[2:, 3 : 3 + g.shape[1], 3 : 3 + g.shape[2]] = fill0
    st = stock_of(*stock_items)
    with contextlib.redirect_stdout(io.StringIO()):
        m = placement.optimize_brick_placement(
            placement.PackingCells(g2, c2, visible=v2, base_layers=2, fill=fill),
            options=BuildOptions(base=base),
            stock=st,
        )
    cov = np.zeros(m.grid.shape, int)
    for b in m.bricks:
        cov[b.z : b.z + b.h_plates, b.y : b.y + b.d, b.x : b.x + b.w] += 1
    assert (cov == m.grid.astype(int)).all(), "coverage mismatch"
    fin = st.recount(m.bricks)
    assert +fin.used == +st.used, "running ledger differs from final pieces (rollback / remove bookkeeping)"
    r = stability.report(m, verbose=False)
    assert r["disconnected"] == 0 if "disconnected" in r else True
    tot_have, tot_used = sum(st.have.values()), sum(fin.used.values())
    owned_used = sum(min(st.have.get(k, 0), n) for k, n in fin.used.items())
    assert tot_used == len(m.bricks)
    with tempfile.TemporaryDirectory() as d:
        fin.write_report(os.path.join(d, "r.csv"), os.path.join(d, "b.csv"))
        buy = [r_ for r_ in csv.reader(open(os.path.join(d, "b.csv"))) if r_ and r_[0] in ("Plate", "Brick")]
        assert sum(int(r_[5]) for r_ in buy) == fin.need_total() == tot_used - owned_used
    print("   spare:", [(r_[0], r_[1], r_[2], r_[4], r_[8]) for r_ in fin.rows() if r_[8]])
    print(f"6 {label}: {len(m.bricks)} pieces, {owned_used}/{tot_have} owned used, buy {fin.need_total()}")

# 7. stock CSV round trip: an inventory CSV (with section column and TOTAL row) loads as stock
with tempfile.TemporaryDirectory() as d:
    from image_to_bricks.model import export_inventory_csv

    p = os.path.join(d, "inv.csv")
    with contextlib.redirect_stdout(io.StringIO()):
        export_inventory_csv(m, p)
    s2 = Stock.from_csv(p)
    assert sum(s2.have.values()) == len(m.bricks)
print("7 inventory CSV loads as stock")
# 8. hidden base cells (inside the rim, under the figure) may use any owned colour; visible ones keep the base colour
g, c = synthetic()
vis = voxelize.surface_mask(g)
hol = voxelize.voxels_to_lego_units(g, hollow=True, shell_thickness_studs=2)
BLACK_BASE = BaseOptions(plates=2, shape="rect", margin_studs=3, colour_id=BLACK)
g2, c2, v2, _ = stand.add_base(hol, c, vis & hol, BLACK_BASE)
bv = stand.base_visibility(g2, 2)
assert (v2[:2] == bv).all() and bv[0].sum() < g2[0].sum() and bv[1].sum() < g2[1].sum()
rim = g2[0] & ~(
    np.pad(g2[0], 1)[2:, 1:-1] & np.pad(g2[0], 1)[:-2, 1:-1] & np.pad(g2[0], 1)[1:-1, 2:] & np.pad(g2[0], 1)[1:-1, :-2]
)
assert (bv[0] == rim).all(), "bottom layer: only the rim is visible"


def run8(strict):
    st = stock_of((1, 4, 4, YELLOW, 40), (1, 2, 2, GREY, 20))
    options = BuildOptions(
        base=replace(BLACK_BASE, hidden_any_colour=not strict), repair=RepairOptions(tie_repair=False)
    )
    with contextlib.redirect_stdout(io.StringIO()):
        m = placement.optimize_brick_placement(
            placement.PackingCells(g2, c2, visible=v2, base_layers=2), options=options, stock=st
        )
    odd = 0
    for b in m.bricks:
        if b.z < 2:
            sees = bv[b.z, b.y : b.y + b.d, b.x : b.x + b.w].any()
            assert not sees or b.color == BLACK, "a piece with a visible cell must keep the base colour"
            odd += b.color != BLACK
    return m, odd


m8, odd = run8(False)
assert odd > 0, "hidden base cells should use owned pieces of other colours"
m8s, odd_s = run8(True)
assert odd_s == 0
print(
    f"8 hidden base cells: {odd} base pieces in other owned colours (strict: {odd_s}); "
    f"pieces {len(m8.bricks)} vs strict {len(m8s.bricks)}"
)
print("OK")
