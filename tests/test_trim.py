"""--trim-colors: each trim feature (hat band tilted over layers, boot cuffs) becomes ONE full ring; stray trim
cells are recoloured; --color-alias renames a colour after classification. Synthetic grid, no mesh needed."""

import support

support.stub_mesh_libraries()

import numpy as np

from image_to_bricks.colours import surface_cleanup

DR, OR, BK = 17, 9, 6
g = np.zeros((40, 20, 30), bool)
g[0:20, 3:8, 3:8] = True  # leg A
g[0:20, 3:8, 12:17] = True  # leg B
g[20:30, 3:8, 3:17] = True  # body joins the legs
g[30:38, 2:10, 2:12] = True  # hat
g[5:8, 12:14, 20:29] = True  # tail, separate part (not touching the legs)
c = np.full(g.shape, OR, np.uint8)
ring = lambda z, ys, xs: [(z, y, x) for y in ys for x in xs if y in (ys[0], ys[-1]) or x in (xs[0], xs[-1])]
for z, y, x in ring(10, range(3, 8), range(3, 8))[:9]:
    c[z, y, x] = DR  # boot A cuff: 9 of 16
for z, y, x in ring(12, range(3, 8), range(12, 17))[:5]:
    c[z, y, x] = DR  # boot B cuff, a different height: 5 of 16
# hat band tilted over layers 32-35: a few cells in each, most at 33
for z, n in ((32, 3), (33, 8), (34, 4), (35, 2)):
    for _, y, x in ring(z, range(2, 10), range(2, 12))[0:n]:
        c[z, y, x] = DR
c[6, 12, 22] = c[6, 13, 24] = DR  # 2 stray cells on the tail: below min_cells
c[20, 5, 9] = DR  # lone cell at the joint: must not trigger
res = surface_cleanup.trim_ring_mask(c, g, [DR], 0.15, 4)
tm, stray = res[DR]
assert tm[10, 3:8, 3:8].sum() == 16 and tm[12, 3:8, 12:17].sum() == 16, "both boot rings must be full"
assert tm[33, 2:10, 2:12].sum() == 32, "hat ring on the layer with most cells"
assert tm.sum() == 16 + 16 + 32 and set(np.nonzero(tm.any(axis=(1, 2)))[0]) == {10, 12, 33}
assert not tm[6].any() and not tm[20].any() and not stray[6].any() and not stray[20].any()
assert stray[32].sum() == 3 and stray[34].sum() == 4 and stray[35].sum() == 2, "hat cells off the ring are stray"
out = surface_cleanup.apply_color_overrides(c, g, {DR: 2})
assert not (out[g] == DR).any() and (out[g] == 2).sum() == (c[g] == DR).sum()
# stray cells take the neighbours' colour, rings are not touched
c2 = c.copy()
c2[g & (c == DR)] = DR
c3 = surface_cleanup.recolour_from_neighbours(c2, g, stray, {DR})
assert not (c3[stray] == DR).any()
print(
    "trim rings:",
    int(tm.sum()),
    "cells in layers",
    sorted(set(np.nonzero(tm.any(axis=(1, 2)))[0].tolist())),
    f"; {int(stray.sum())} stray cells recoloured",
)
print("OK")

# hat with a wide brim: the busiest layer of the band cuts through the brim (outline = brim edge, band not on it);
# the ring must then go to the next layer that shows the crown alone
g2 = np.zeros((20, 24, 24), bool)
g2[0:16, 8:16, 8:16] = True  # crown 8x8
g2[5:7, 2:22, 2:22] = True  # brim 20x20, two plates thick (z 5, 6)
c2 = np.full(g2.shape, OR, np.uint8)
crown_ring = [(y, x) for y in range(8, 16) for x in range(8, 16) if y in (8, 15) or x in (8, 15)]
for y, x in crown_ring:
    c2[6, y, x] = DR  # band cells in the brim layer: 28 (busiest layer)
for y, x in crown_ring[:20]:
    c2[7, y, x] = DR  # ...and 20 on the crown just above the brim
tm2, st2 = surface_cleanup.trim_ring_mask(c2, g2, [DR], 0.15, 4)[DR]
assert tm2[7, 8:16, 8:16].sum() == 28, "ring completed on the crown layer above the brim"
assert tm2.sum() == 28 and not tm2[6].any(), tm2.sum()
assert st2[6].sum() == 28, "band cells in the brim layer are stray, recoloured from neighbours"
print("brim case: ring on the crown layer", int(np.nonzero(tm2.any(axis=(1, 2)))[0][0]), "OK")
