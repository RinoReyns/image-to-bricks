import math

import support

support.stub_mesh_libraries()
import numpy as np

from image_to_bricks.analysis import stability as st
from image_to_bricks.model import LegoModel, PlacedBrick


def make(bricks, shape, nb):
    g = np.zeros(shape, bool)
    for b in bricks:
        g[b.z : b.z + b.h_plates, b.y : b.y + b.d, b.x : b.x + b.w] = True
    return LegoModel(grid=g, bricks=bricks, base_layers=nb, base_color=8)


P = PlacedBrick
# 1. symmetric: exact CoM numbers
base = [P(0, 0, 0, 10, 10, 1, 8), P(0, 0, 1, 10, 10, 1, 8)]
tower = [
    P(4, 4, 2, 2, 2, 3, 9),
    P(4, 4, 5, 2, 2, 3, 9),
    P(4, 4, 8, 2, 2, 3, 9),
    P(4, 4, 11, 2, 2, 1, 9),
]  # 10 plates high
m = make(base + tower, (12, 10, 10), 2)
r = st.report(m)
x, y, z = r["stability"]["com_mm"]
assert abs(x - 40) < 1e-6 and abs(y - 40) < 1e-6 and abs(z - 6.4) < 1e-6, (x, y, z)
assert abs(r["stability"]["tip_angle"] - math.degrees(math.atan2(40, 6.4))) < 1e-6
assert r["components"] == 1 and r["weak_joints"] == 0 and r["stability"]["ok"]

# 2. weak joint chain, floating piece, single-stud footing
extra = [
    P(3, 3, 5, 1, 1, 1, 9),
    P(3, 3, 6, 4, 1, 1, 9),  # 1x1 plate on a 4x4 brick? build own body below
    P(0, 0, 12, 1, 1, 1, 9),  # floating, not touching anything
    P(8, 8, 2, 1, 1, 3, 9),
]  # 1x1 brick on the base (single stud)
body = [P(2, 2, 2, 4, 4, 3, 9)]
m = make(base + body + [P(3, 3, 5, 1, 1, 1, 9), P(3, 3, 6, 4, 1, 1, 9)] + [extra[2], extra[3]], (14, 10, 10), 2)
r = st.report(m)
print(r["rows"])
assert r["components"] == 2 and r["disconnected_pieces"] == 1
assert r["weak_joints"] == 3, r["weak_joints"]  # 1x1 brick on base, 1x1 on body, 4x1 on the 1x1
assert r["single_stud_footings"] == 1
hang = sorted(row[5] for row in r["rows"] if row[0] == "weak_joint")
assert hang == [1, 1, 2], hang

# 3. overhang: CoM outside base -> warning, direction, margin hint
base = [P(0, 0, 0, 4, 4, 1, 8), P(0, 0, 1, 4, 4, 1, 8)]
fig = [P(0, 0, 2, 4, 4, 3, 9), P(4, 0, 2, 16, 4, 3, 9)]
m = make(base + fig, (5, 4, 20), 2)
r = st.report(m)
s = r["stability"]
assert not s["ok"] and s["dist_mm"] < 0 and "high x" in s["direction"], s
print("ALL OK")
