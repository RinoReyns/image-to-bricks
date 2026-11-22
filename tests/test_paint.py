"""--paint: surface cells in a box get a colour (survives smoothing options), interior cells and empty cells do not."""

import sys

import support

support.stub_mesh_libraries()

import numpy as np

from image_to_bricks.colours import hand_edits
from image_to_bricks.mesh import voxelize
from image_to_bricks.options import Paint

S = slice
ok = []
g = np.zeros((6, 6, 6), bool)
g[1:5, 1:5, 1:5] = True
surface = voxelize.surface_mask(g)
lego = np.full(g.shape, 9, np.uint8)
ok.append(hand_edits.apply_paints(lego, surface) is lego)  # no paints: unchanged
DP, BK = 26, 6
paints = [
    Paint((S(2, 3), S(1, 2), S(2, 4)), BK, False),
    Paint((S(2, 3), S(1, 2), S(3, 4)), DP, False),
]  # front face y=1; 2nd covers the 1st
out = hand_edits.apply_paints(lego, surface, None, paints)
ok += [out[2, 1, 2] == BK, out[2, 1, 3] == DP, int((out != lego).sum()) == 2, bool((lego == 9).all())]
paints = [Paint((S(2, 3), S(2, 3), S(2, 3)), BK, False)]  # an interior cell: not on the surface
ok.append(int((hand_edits.apply_paints(lego, surface, None, paints) != lego).sum()) == 0)
paints = [Paint((S(0, 1), S(0, 1), S(0, 1)), BK, False)]  # empty space
ok.append(int((hand_edits.apply_paints(lego, surface, None, paints) != lego).sum()) == 0)
# --paint-deep: also the hidden wall cell behind the surface; --paint alone leaves it
paints = [Paint((S(2, 3), S(1, 3), S(2, 3)), DP, True)]
deep = hand_edits.apply_paints(lego, surface, g, paints)
ok += [deep[2, 1, 2] == DP, deep[2, 2, 2] == DP, int((deep != lego).sum()) == 2]
paints = [Paint((S(2, 3), S(1, 3), S(2, 3)), DP, False)]
ok.append(int((hand_edits.apply_paints(lego, surface, g, paints) != lego).sum()) == 1)
paints = [Paint((S(0, 1), S(0, 1), S(0, 1)), DP, True)]  # empty space stays empty even when deep
ok.append(int((hand_edits.apply_paints(lego, surface, g, paints) != lego).sum()) == 0)
print(ok)
print("all passed" if all(ok) else "FAILED")
sys.exit(0 if all(ok) else 1)
