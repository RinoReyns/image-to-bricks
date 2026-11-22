"""A pink 1x2 brick as a nose: one stud in the wall (hidden), one at the surface. --paint-deep colours both wall cells,
the packer then uses the owned brick (also through tie repair + clean-up); without --paint-deep only the surface cell is
pink and the owned brick cannot be used."""

import contextlib
import io
import sys

import support

support.stub_mesh_libraries()
import numpy as np

from image_to_bricks.colours import hand_edits
from image_to_bricks.consts import COLOR_NAMES
from image_to_bricks.mesh import voxelize
from image_to_bricks.options import BuildOptions, PackingOptions, Paint, RepairOptions
from image_to_bricks.packing import placement
from image_to_bricks.stock import Stock, make_key

PINK = [k for k, v in COLOR_NAMES.items() if v == "Bright Pink"][0]
OR = 9
S = slice
g = np.zeros((12, 6, 12), bool)
g[:, 0:6, :] = True  # solid block, front = y 0 (y 1 is not a surface cell)
c = np.full(g.shape, OR, np.uint8)
surface = voxelize.surface_mask(g)
ok = []
for deep in (True, False):
    paints = [Paint((S(3, 6), S(0, 2), S(5, 6)), PINK, deep)]
    with contextlib.redirect_stdout(io.StringIO()):
        cc = hand_edits.apply_paints(c, surface, g, paints)
    for ties in (False, True):
        stock = Stock({make_key(3, 2, 1, PINK): 1})
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                m = placement.optimize_brick_placement(
                    placement.PackingCells(g, cc, visible=surface, base_layers=0),
                    options=BuildOptions(
                        packing=PackingOptions(hidden_any_colour=False),  # as with --strict-hidden
                        repair=RepairOptions(tie_repair=ties, cleanup_small=ties),
                    ),
                    stock=stock,
                )
        finally:
            used = stock.recount(m.bricks)
        bricks = [b for b in m.bricks if b.color == PINK]
        brick_ok = len(bricks) == 1 and (bricks[0].h_plates, sorted((bricks[0].w, bricks[0].d))) == (3, [1, 2])
        ok.append(brick_ok if deep else not brick_ok)
        if deep:
            ok.append(sum(used.used.values()) >= 1 and used.used.get(make_key(3, 2, 1, PINK), 0) == 1)
        print(f"deep={deep} ties={ties}: pink pieces {[(b.h_plates, b.w, b.d) for b in bricks]}")
print(ok)
print("all passed" if all(ok) else "FAILED")
sys.exit(0 if all(ok) else 1)
