"""Layer-map axes count from the base plate's corner (0, 0); --front marks the face side; {front_stud} in an
--accent note; stability.csv has plate_x / plate_y."""

import contextlib
import csv
import io
import os
import sys
import tempfile

import support

support.stub_mesh_libraries()
import numpy as np
from PIL import ImageDraw

from image_to_bricks.analysis import stability
from image_to_bricks.instructions import fast_render as fr
from image_to_bricks.model import LegoModel, front_stud, plate_origin
from image_to_bricks.model import PlacedBrick as PB
from image_to_bricks.options import Accent
from image_to_bricks.packing import accents

failed = False


def check(name, ok, info=None):
    global failed
    print(("ok   " if ok else "FAIL ") + name)
    failed |= not ok


# plate origin: base footprint (bottom layer), figure may overhang it
g = np.zeros((6, 20, 30), bool)
g[0:2, 6:14, 8:28] = True  # base 20 x 8 at x 8, y 6
g[2:6, 4:10, 3:12] = True  # figure overhangs to x 3, y 4
check("origin = base corner", plate_origin(g, 2) == (8, 6))
check("origin without a base = figure footprint", plate_origin(g, 0) == (8, 6) or plate_origin(g[2:], 0) == (3, 4))
check("origin without a base uses the whole figure", plate_origin(g[2:], 0) == (3, 4))

# front stud
b = PB(5, 7, 0, 1, 2, 1, 1)
check("front -y: lower end", front_stud(b, "-y") == ((5, 7), False))
check("front +y: upper end", front_stud(b, "+y") == ((5, 8), False))
check("plate across the front", front_stud(PB(5, 7, 0, 2, 1, 1, 1), "-y")[1] is True)

# layer map tick labels
texts = []
orig = ImageDraw.ImageDraw.text


def spy(self, xy, text, *a, **k):
    texts.append(str(text))
    return orig(self, xy, text, *a, **k)


ImageDraw.ImageDraw.text = spy
brick = PB(8, 6, 0, 2, 2, 1, 1)
fr.layer_map(
    [brick], fr.MapRegion(3, 2, 20, 12, plate_x0=8, plate_y0=6), "t", lambda c: (200, 200, 200), fr.MapMarks(front="-y")
)
ImageDraw.ImageDraw.text = orig
nums = sorted(int(t) for t in texts if t.lstrip("-").isdigit())
check(
    "axis numbers are relative to the plate (x0=3, ox=8 -> -5, 0, 5, 10)",
    {-5, 0, 5, 10} <= set(nums) and 8 not in nums,
    nums,
)
check("FRONT label drawn", "FRONT" in texts)
texts.clear()
ImageDraw.ImageDraw.text = spy
fr.layer_map([brick], fr.MapRegion(3, 2, 20, 12), "t", lambda c: (200, 200, 200))
ImageDraw.ImageDraw.text = orig
check(
    "no front, no origin: unchanged behaviour (grid numbers, no FRONT label)",
    "FRONT" not in texts and 5 in [int(t) for t in texts if t.isdigit()],
)

# accent placeholder
LBG, CHROME = 7, 25
bricks = [PB(0, 0, z, 6, 6, 1, LBG) for z in range(3)] + [PB(2, 1, 3, 1, 2, 1, LBG)]
gg = np.zeros((5, 8, 8), bool)
for p in bricks:
    gg[p.z : p.z + p.h_plates, p.y : p.y + p.d, p.x : p.x + p.w] = True
mdl = LegoModel(grid=gg, bricks=bricks, base_layers=0, front="-y")
ACCENTS = [Accent((slice(0, 5), slice(0, 8), slice(0, 8)), CHROME, 1, 2, "ball on the {front_stud}", LBG)]
with contextlib.redirect_stdout(io.StringIO()):
    accents.apply_accents(mdl, ACCENTS)
txt = mdl.accents[0][3]
check(
    "accent text names the front stud (plate origin 0,0 -> x=2, y=1)", "stud nearest the front (x=2, y=1)" in txt, txt
)

# stability csv columns
st_model = LegoModel(
    grid=g.copy(), bricks=[PB(8, 6, 0, 4, 4, 1, 1), PB(8, 6, 1, 4, 4, 1, 1), PB(3, 4, 2, 1, 1, 1, 1)], base_layers=2
)
st_model.grid[:] = False
for p in st_model.bricks:
    st_model.grid[p.z : p.z + p.h_plates, p.y : p.y + p.d, p.x : p.x + p.w] = True
with tempfile.TemporaryDirectory() as d:
    with contextlib.redirect_stdout(io.StringIO()):
        stability.report(st_model, d, "m")
    rows = list(csv.DictReader(open(os.path.join(d, "m_stability.csv"))))
    hdr = list(rows[0].keys()) if rows else None
    check(
        "stability csv has plate_x / plate_y",
        rows and "plate_x" in hdr and int(rows[0]["plate_x"]) == int(rows[0]["x"]) - 8,
        hdr,
    )

print("FAILED" if failed else "all passed")
sys.exit(1 if failed else 0)
