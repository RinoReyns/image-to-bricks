"""--accent: a plate in a box becomes the special part (colour change + note + stock), special stock rows are never
packed. Synthetic, no mesh needed."""

import contextlib
import io
import os
import tempfile

import support

support.stub_mesh_libraries()
import numpy as np

from image_to_bricks.instructions import outputs_handler as oh
from image_to_bricks.model import LegoModel, PlacedBrick
from image_to_bricks.options import Accent, OutputOptions
from image_to_bricks.packing import accents
from image_to_bricks.stock import Stock, make_key

CHROME, LBG = 25, 7
# stock file with a special row: not packed, kept aside, matched by the accent
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "s.csv")
    open(p, "w").write(
        "piece_type,width_studs,depth_studs,height_plates,color_name,quantity,shape\n"
        "Plate,2,2,1,Black,3,\n"
        "Plate (special),2,1,1,Chrome Silver,1,plate 1x2 with ball on one stud\n"
        "Plate (special),4,4,1,Black,2,wedge plate 4x4\n"
        "Brick (special),2,1,3,Yellow,2,slope inverted\n"
    )
    with contextlib.redirect_stdout(io.StringIO()):
        st = Stock.from_csv(p)
assert sum(st.have.values()) == 3 and st.have[make_key(1, 2, 2, 6)] == 3, "special rows must not be packable"
assert sum(st.specials.values()) == 5 and st.specials[(1, 2, 1, CHROME, "plate 1x2 with ball on one stud")] == 1

g = np.ones((10, 6, 6), bool)
bricks = [PlacedBrick(0, 0, z, 6, 6, 1, LBG) for z in range(8)]  # wall of big plates
bricks += [PlacedBrick(0, 0, 8, 2, 1, 1, LBG), PlacedBrick(2, 0, 8, 2, 1, 1, LBG), PlacedBrick(4, 0, 8, 2, 1, 1, LBG)]
bricks += [PlacedBrick(0, 1, 9, 2, 1, 1, LBG)]
mdl = LegoModel(grid=g, bricks=bricks, base_layers=0)
ACCENTS = [Accent((slice(0, 10), slice(0, 3), slice(2, 6)), CHROME, 1, 2, "P121 plate with ball", None)]
with contextlib.redirect_stdout(io.StringIO()):
    accents.apply_accents(mdl, ACCENTS, stock=st)
chrome = [b for b in bricks if b.color == CHROME]
# highest 2x1 plate inside the box: layer 8 (x 2 or 4); the layer-9 plate has x=0 -> outside
assert len(chrome) == 1 and chrome[0].z == 8 and chrome[0].x in (2, 4), [(b.x, b.z) for b in chrome]
assert len(mdl.accents) == 1 and "Chrome Silver plate 1x2" in mdl.accents[0][3] and "P121" in mdl.accents[0][3]
assert st.have[make_key(1, 2, 1, CHROME)] == 1, "the special piece in the stock file counts as owned"
fin = st.recount(bricks)
rows = [r for r in fin.rows() if r[4] == "Chrome Silver"]
assert rows and rows[0][5] == 1 and rows[0][6] == 1 and rows[0][7] == 0 and rows[0][9] == "ok", rows
with tempfile.TemporaryDirectory() as d:
    with contextlib.redirect_stdout(io.StringIO()):
        oh.save_outputs(mdl, "t", d, options=OutputOptions(make_pdf=True))
    assert os.path.getsize(os.path.join(d, "t_instructions.pdf")) > 1000
print("accent:", mdl.accents[0][3], "at layer", mdl.accents[0][0] + 1, "| stock row", rows[0])

# regression (step 64): the accent recolour must move the plate in the running ledger too, or every run with
# --accent and --stock printed "the running stock ledger differs from the final pieces"
ledger = Stock(have={make_key(1, 2, 1, LBG): 4}, specials=st.specials)
plates = [PlacedBrick(x, 0, 8, 2, 1, 1, LBG) for x in (0, 2, 4)]
for plate in plates:
    ledger.take(1, plate.w, plate.d, plate.color)  # what the packer recorded
model2 = LegoModel(grid=g, bricks=plates, base_layers=0)
ACCENTS = [Accent((slice(0, 10), slice(0, 3), slice(2, 6)), CHROME, 1, 2, "", LBG)]
with contextlib.redirect_stdout(io.StringIO()):
    accents.apply_accents(model2, ACCENTS, stock=ledger)
assert +ledger.recount(plates).used == +ledger.used, (dict(ledger.used), dict(ledger.recount(plates).used))
print("accent keeps the running ledger == final pieces")

# sturdier accent (step 64): no 1x2 plate of the wanted colour, but two 1x1 plates of it fill a 1x2 footprint ->
# merged into one 1x2 plate (same cells), which becomes the accent; the ledger follows
from image_to_bricks.model import piece_cells

singles = [
    PlacedBrick(3, 1, 8, 1, 1, 1, LBG),
    PlacedBrick(4, 1, 8, 1, 1, 1, LBG),
    PlacedBrick(2, 1, 8, 1, 1, 1, 6),
    PlacedBrick(0, 1, 7, 3, 1, 1, LBG),
]
cells_before = sorted((b.z, c) for b in singles for c in piece_cells(b))
ledger3 = Stock(have={}, specials=st.specials)
for piece in singles:
    ledger3.take_brick(piece)
model3 = LegoModel(grid=g, bricks=list(singles), base_layers=0)
ACCENTS = [Accent((slice(0, 10), slice(0, 3), slice(2, 6)), CHROME, 1, 2, "", LBG)]
with contextlib.redirect_stdout(io.StringIO()) as log:
    accents.apply_accents(model3, ACCENTS, stock=ledger3)
chrome3 = [b for b in model3.bricks if b.color == CHROME]
assert len(chrome3) == 1 and (chrome3[0].x, chrome3[0].y, chrome3[0].z, chrome3[0].w, chrome3[0].d) == (
    3,
    1,
    8,
    2,
    1,
), [(b.x, b.y, b.z, b.w, b.d, b.color) for b in model3.bricks]
assert sorted((b.z, c) for b in model3.bricks for c in piece_cells(b)) == cells_before, "cells must not change"
assert "merged 2 Light Bluish Gray plates" in log.getvalue(), log.getvalue()
assert +ledger3.recount(model3.bricks).used == +ledger3.used
# a footprint whose plate sticks out of it is not merged (the 3x1 at layer 7 is not split)
from image_to_bricks.packing import accent_merge

assert (
    accent_merge.find_mergeable_footprint(
        [PlacedBrick(0, 1, 7, 3, 1, 1, LBG), PlacedBrick(3, 1, 7, 1, 1, 1, LBG)], 1, 2, LBG, lambda z, y, x: True
    )
    is None
)
print("accent merges small plates of the wanted colour into the accent plate")
print("OK")
