"""--thicken-thin: 1-wide parts get a plus-shaped cross-section, thick parts are untouched, the layers of a diagonal tail
overlap by 2+ studs. Synthetic grid (a body block + a 1-cell diagonal tail), no mesh needed."""

import support

support.stub_mesh_libraries()

import numpy as np

from image_to_bricks.mesh import shape_edits

g = np.zeros((30, 20, 20), bool)
g[10:30, 2:10, 2:10] = True  # body 8x8
# tail: 1 cell wide, one step (+x,+y) every layer going down from the body
x, y = 8, 8
tail = []
for z in range(9, -1, -1):
    x += 1 if z % 2 == 0 else 0
    y += 1 if z % 2 == 1 else 0
    g[z, y, x] = True
    tail.append((z, y, x))
before = g.copy()


def thin(gr):
    t = 0
    for z in range(gr.shape[0]):
        L = gr[z]
        blk = L[:-1, :-1] & L[1:, :-1] & L[:-1, 1:] & L[1:, 1:]
        cov = np.zeros_like(L)
        for dy in (0, 1):
            for dx in (0, 1):
                cov[dy : 19 + dy, dx : 19 + dx] |= blk
        t += int((L & ~cov).sum())
    return t


def min_overlap(gr, z0, z1):
    return min(int((gr[z] & gr[z + 1]).sum()) for z in range(z0, z1))


o0 = min_overlap(before, 0, 9)
out, n = shape_edits.thicken_thin(g)
assert thin(before) == len(tail)
assert (out[10:30] == before[10:30]).all(), "body layers must not change"
assert (out >= before).all(), "nothing is removed"
o1 = min_overlap(out, 0, 9)
assert o0 <= 1 and o1 >= 2, (o0, o1)
assert n <= 4 * len(tail), n
# a grid with no thin cells is unchanged
b = np.zeros((4, 8, 8), bool)
b[:, 1:5, 1:6] = True
o, k = shape_edits.thicken_thin(b)
assert k == 0 and (o == b).all()
# with a box only thin cells inside it change
ob, nb_ = shape_edits.thicken_thin(g, (slice(0, 5), slice(0, 20), slice(0, 20)))
assert (ob[5:] == before[5:]).all() and nb_ < n and (ob[:5] != before[:5]).any()
# diagonal 2x2 stairs: each layer's block shifts by (1,1) -> the layers touch by one corner stud only
D = np.zeros((12, 14, 14), bool)
D[8:12, 0:8, 0:8] = True  # body on top
for k in range(8):
    z = 7 - k
    D[z, k + 0 : k + 2, k + 0 : k + 2] = True if k < 6 else False  # tail steps down and out
bx = (slice(0, 8), slice(0, 14), slice(0, 14))
ov = lambda gr: [int((gr[z] & gr[z + 1]).sum()) for z in range(0, 7)]
D2, nj = shape_edits.widen_joints(D, bx)
assert min(ov(D)[2:7]) == 1 and min(ov(D2)[2:7]) >= 2 and nj > 0 and (D2 >= D).all(), (ov(D), ov(D2), nj)
D3, nj3 = shape_edits.widen_joints(D, (slice(0, 2), slice(0, 2), slice(0, 2)))
assert nj3 == 0 and (D3 == D).all()
print("widen joints: overlaps", ov(D), "->", ov(D2), f"(+{nj} cells)")
print(f"thicken: thin cells {thin(before)}, tail joints {o0} -> {o1} studs, +{n} cells")
print("OK")
