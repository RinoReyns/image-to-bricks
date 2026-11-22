"""Eyes / nose: Dark Pink in the palette, --smooth-iters / --absorb-min-region keep 1-2 cell details."""

import sys

import support

support.stub_mesh_libraries()
import numpy as np

from image_to_bricks.colours import palette_matching, surface_cleanup
from image_to_bricks.consts import COLOR_NAMES, LEGO_RGB, PALETTE

ok = []
DP = [k for k, v in COLOR_NAMES.items() if v == "Dark Pink"]
ok.append(len(DP) == 1 and LEGO_RGB[DP[0]] == (223, 102, 149) and PALETTE[DP[0]][2] == 47)
ok.append(
    palette_matching.nearest_lego_color(np.array([[223, 102, 149]]), [DP[0], 9, 6]) == DP[0]
)  # exact RGB -> Dark Pink
ok.append(
    palette_matching.nearest_lego_color(np.array([[200, 120, 150]]), [DP[0], 9, 6, 1]) == DP[0]
)  # dusty pink -> Dark Pink
ok.append(palette_matching.nearest_lego_color(np.array([[254, 138, 24]]), [DP[0], 9, 6]) == 9)  # orange unaffected

# a flat orange face (one layer thick, 8x8 surface) with a 1-cell nose and a 2-cell eye
OR, BK = 9, 6
g = np.zeros((4, 8, 8), bool)
g[1:3, :, :] = True
c = np.full(g.shape, OR, np.uint8)
c[2, 4, 4] = DP[0]  # nose: 1 cell
c[2, 2, 2] = c[2, 2, 3] = BK  # eye: 2 cells

kept = surface_cleanup.smooth_lego_colors(c, g, iters=0, majority=4)
ok.append(bool((kept == c).all()))  # 0 iterations = unchanged
smoothed = surface_cleanup.smooth_lego_colors(c, g, iters=2, majority=4)
ok.append(smoothed[2, 4, 4] != DP[0])  # default smoothing erases a 1-cell dot
absorbed = surface_cleanup.absorb_small_color_regions_3d(c, g, min_region_size=3)
ok.append(absorbed[2, 4, 4] == OR and absorbed[2, 2, 2] == OR)  # default absorb removes both
kept2 = surface_cleanup.absorb_small_color_regions_3d(c, g, min_region_size=1)
ok.append(bool((kept2 == c).all()))  # 1 = off
kept3 = surface_cleanup.absorb_small_color_regions_3d(c, g, min_region_size=2)
ok.append(kept3[2, 2, 2] == BK and kept3[2, 4, 4] == OR)  # 2 keeps the 2-cell eye, drops the 1-cell dot

# trim clean-up must never leave colour 0: a surface cell whose neighbours are mostly interior (id 0) keeps a real colour
sg = np.zeros((3, 3, 3), bool)
sg[1, 1, 1] = True
sg[1, 1, 2] = True  # surface = 2 cells; the rest is interior
lg = np.zeros(sg.shape, np.uint8)
lg[1, 1, 1] = 17
lg[1, 1, 2] = OR  # stray trim cell next to an orange cell
full = np.ones(sg.shape, bool)
res = surface_cleanup.recolour_from_neighbours(lg, sg, np.isin(lg, [17]) & sg, {17})
ok.append(res[1, 1, 1] == OR)
res2 = surface_cleanup.recolour_from_neighbours(
    lg, full, np.isin(lg, [17]) & full, {17}
)  # even with the whole grid as "valid"
ok.append(res2[1, 1, 1] == OR)

print(ok)
print("all passed" if all(ok) else "FAILED")
sys.exit(0 if all(ok) else 1)
