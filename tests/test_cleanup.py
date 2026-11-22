"""--cleanup-small: lossless (same cells, same colours), 1x1 / pieces never more, wall not weaker,
stock ledger consistent. Synthetic data."""

import contextlib
import io

import support

support.stub_mesh_libraries()
import numpy as np

from image_to_bricks.mesh import voxelize
from image_to_bricks.options import BuildOptions, RepairOptions
from image_to_bricks.packing import assembly, cleanup, placement
from image_to_bricks.stock import Stock, make_key


def blob(seed):
    rng = np.random.default_rng(seed)
    nz, ny, nx = 12, 18, 22
    g = np.zeros((nz, ny, nx), bool)
    for z in range(nz):
        g[z, 2:16, 2:20] = True
        for _ in range(14):  # ragged holes -> awkward leftover shapes
            y, x = rng.integers(2, 15), rng.integers(2, 19)
            g[z, y : y + rng.integers(1, 3), x : x + rng.integers(1, 3)] = False
    c = np.full(g.shape, 9, np.uint8)
    c[:, :, 11:] = 5
    c[::2][:, 5:9, 4:8] = 6
    return g, c


def run(g, c, clean, stock=None, tol=(0, 0)):
    with contextlib.redirect_stdout(io.StringIO()):
        m = placement.optimize_brick_placement(
            placement.PackingCells(g, c, visible=voxelize.surface_mask(g), base_layers=0),
            options=BuildOptions(
                repair=RepairOptions(cleanup_small=clean, cleanup_extra_seams=tol[0], cleanup_extra_stacks=tol[1])
            ),
            stock=stock,
        )
    cov = np.zeros(g.shape, int)
    col = np.full(g.shape, -1)
    for b in m.bricks:
        cov[b.z : b.z + b.h_plates, b.y : b.y + b.d, b.x : b.x + b.w] += 1
        col[b.z : b.z + b.h_plates, b.y : b.y + b.d, b.x : b.x + b.w] = b.color
    assert (cov == m.grid.astype(int)).all(), "coverage"
    return m, col


def weakness(m):
    asm = assembly.Assembly(m.grid.shape, m.bricks)
    a = s = 0
    for L in range(1, asm.nz):
        q = cleanup._quality(asm, {L}, cleanup.Window(0, asm.ny, 0, asm.nx))
        a += q[0]
        s += q[1]
    return a, s


SEED_COUNT = (
    4 if support.is_exhaustive() else 2
)  # every seed is a different ragged blob; 2 already cover both stock modes
tot_gain = 0
for seed in range(SEED_COUNT):
    g, c = blob(seed)
    for label, st in (
        ("no stock", None),
        ("stock", Stock({make_key(1, 4, 2, 9): 5, make_key(3, 2, 2, 5): 6, make_key(1, 1, 1, 6): 3})),
    ):
        m0, c0 = run(g, c, False, st.fork() if st else None)
        stk = st.fork() if st else None
        m1, c1 = run(g, c, True, stk)
        assert (c0 == c1).all(), "colours must not change"
        n1 = lambda m: sum(b.w == 1 and b.d == 1 for b in m.bricks)
        assert n1(m1) <= n1(m0) and len(m1.bricks) <= len(m0.bricks)
        w0, w1 = weakness(m0), weakness(m1)
        assert w1[0] <= w0[0] and w1[1] <= w0[1], f"wall got weaker {w0} -> {w1}"
        if stk is not None:
            fin = stk.recount(m1.bricks)
            assert +fin.used == +stk.used, "ledger"
            f0 = st.recount(m0.bricks)
            assert fin.owned_used() >= f0.owned_used() and fin.need_total() <= f0.need_total()
        tot_gain += n1(m0) - n1(m1)
        print(
            f"seed {seed} {label:8s}: 1x1 {n1(m0)} -> {n1(m1)}, pieces {len(m0.bricks)} -> {len(m1.bricks)}, "
            f"weakness (seams, stacks) {w0} -> {w1}"
        )
assert tot_gain > 0, "the clean-up should remove some 1x1 on these shapes"
print("OK")
