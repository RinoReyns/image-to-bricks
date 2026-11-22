"""Pieces with nothing under them are found (PDF/maps mark them) and --prop-box adds a post under a hanging part.
Synthetic, no mesh needed."""

import contextlib
import io
import os
import tempfile

import support

support.stub_mesh_libraries()
import numpy as np

from image_to_bricks.instructions import outputs_handler as oh
from image_to_bricks.mesh import shape_edits
from image_to_bricks.model import LegoModel, PlacedBrick
from image_to_bricks.options import OutputOptions

# a table top hanging from a wall: z 0-5 wall, z 6-8 a ledge one stud out, a cantilevered tip below it is NOT possible,
# so build: wall (x 0-1), ledge from z=3 extends to x 2-4 (nothing under it)
g = np.zeros((6, 4, 6), bool)
g[0:6, 0:4, 0:2] = True
g[3:6, 0:4, 2:5] = True
bricks = [PlacedBrick(0, 0, z, 2, 4, 1, 6) for z in range(6)] + [PlacedBrick(2, 0, z, 3, 4, 1, 6) for z in range(3, 6)]
mdl = LegoModel(grid=g, bricks=bricks, base_layers=0)
fl = oh.unsupported_pieces(mdl)
ledge_low = [b for b in bricks if b.x == 2 and b.z == 3][0]
assert id(ledge_low) in fl and fl[id(ledge_low)] == 4, fl  # held by the next layer's ledge piece? no: z=4 piece above
assert all(id(b) not in fl for b in bricks if b.x == 0), "wall pieces rest on each other"
assert all(id(b) not in fl for b in bricks if b.x == 2 and b.z > 3), "upper ledge pieces rest on the lower ones"
with tempfile.TemporaryDirectory() as d:
    with contextlib.redirect_stdout(io.StringIO()):
        oh.save_outputs(mdl, "t", d, options=OutputOptions(make_pdf=True))
    assert os.path.getsize(os.path.join(d, "t_instructions.pdf")) > 1000
# --prop-box: a diagonal tail hanging from a wall gets a 2x2 post from its lowest cell to the ground
G = np.zeros((14, 12, 12), bool)
G[8:14, 0:6, 0:6] = True  # body, itself standing on nothing (cut: we only test the post)
G[0:8, 0:6, 0:2] = True  # a leg so the body is held
for k in range(5):
    G[7 - k, 6 + k // 2, 5 + k // 2] = True  # tail going down and out
box = (slice(0, 9), slice(5, 12), slice(4, 12))
G2, prop = shape_edits.add_prop(G, box, 2)
low = [z for z in range(14) if G[z, 6:12, 4:12].any()][0]
assert prop.any() and prop[:low].any() and not prop[low:].any(), "post runs below the lowest tail cell only"
assert (G2 & ~G).sum() == prop.sum() and (G2 >= G).all()
assert (prop.sum(axis=0) > 0).sum() <= 4, "a 2x2 footprint"
Gn, pn = shape_edits.add_prop(G, (slice(0, 2), slice(0, 2), slice(0, 2)), 2)
assert not pn.any() and (Gn == G).all(), "nothing hangs in an empty box: unchanged"
print("prop cells:", int(prop.sum()))
print("floating pieces found:", len(fl), "(1 expected)")
print("OK")
