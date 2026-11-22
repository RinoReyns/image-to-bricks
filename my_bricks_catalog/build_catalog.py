"""Update catalog_stock.csv from the photo records, WITHOUT losing hand-added pieces.

catalog_stock.csv is the ONE catalog (the stock list to use later). Columns:
  piece_type,width_studs,depth_studs,height_plates,color_name,quantity,in_photos,manual,shape,bricklink_part
  quantity  = in_photos + manual      <- the number to use; this is the column to edit by hand
  in_photos = how many of this piece are in catalog_items.csv (generated, do not edit)
  manual    = quantity minus in_photos (generated: pieces that are not in any photo)

How hand edits survive: before rewriting, the script reads the existing catalog_stock.csv and
works out manual = old quantity - old in_photos for every line. New quantity = new photo count +
manual. So you can edit `quantity` (add a line, raise or lower a number) any time; the next run
keeps your difference. Add a new photo to catalog_items.csv and only the photo part grows.
Lines that are only hand-added (in_photos = 0) are kept as they are.

Safety: the old file is copied to catalog_stock.previous.csv before every rewrite.
Irregular pieces (wedge, round, corner, Technic, ...) are in the same file: piece_type ends with
"(special)", width/depth = bounding box in studs, `shape` = short name (part of the line identity,
so a wedge and a round plate of the same colour are separate lines). The packer must ignore every
line whose piece_type contains "special". To add one by hand: new line, e.g.
`Plate (special),4,4,1,Black,3,,,wedge plate 4x4 stepped (10 studs),` (leave in_photos/manual empty).
Close the CSV in Excel before running (a locked file cannot be saved).
"""

import csv
import os
import shutil
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
STOCK = os.path.join(HERE, "catalog_stock.csv")
PREV = os.path.join(HERE, "catalog_stock.previous.csv")
HEADER = [
    "piece_type",
    "width_studs",
    "depth_studs",
    "height_plates",
    "color_name",
    "quantity",
    "in_photos",
    "manual",
    "shape",
    "bricklink_part",
]


type StockKey = tuple[str, int, int, int, str, str]  # (piece_type, width, depth, height, colour name, shape)


def key_of(r: dict[str, str]) -> StockKey:
    """Identity of a catalogue line: piece type, size, colour name and special shape."""
    return (
        r["piece_type"].strip(),
        int(r["width_studs"]),
        int(r["depth_studs"]),
        int(r["height_plates"]),
        r["color_name"].strip(),
        (r.get("shape") or "").strip(),
    )


def num(v: object) -> int | None:
    """The whole number in a CSV cell, or None if the cell holds none."""
    text = str(v).strip()
    return int(text) if text.lstrip("-").isdigit() else None


photo: Counter[StockKey] = Counter()
part_of: dict[StockKey, str] = {}
with open(os.path.join(HERE, "catalog_items.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        photo[key_of(r)] += 1
        part_of.setdefault(key_of(r), r.get("bricklink_part", ""))

old_q: Counter[StockKey] = Counter()
old_p: dict[StockKey, int] = {}
old_part: dict[StockKey, str] = {}
had_photo_col = False
if os.path.exists(STOCK):
    with open(STOCK, encoding="utf-8-sig", newline="") as f:
        text = f.read()
    delim = ";" if text.split("\n", 1)[0].count(";") > text.split("\n", 1)[0].count(",") else ","
    rows = list(csv.DictReader(text.splitlines(), delimiter=delim))
    had_photo_col = bool(rows) and "in_photos" in rows[0]
    for r in rows:
        if not (r.get("piece_type") or "").strip() or num(r.get("quantity", "")) is None:
            continue
        if not (r.get("width_studs") or "").strip():
            continue
        k = key_of(r)
        old_part.setdefault(k, (r.get("bricklink_part") or "").strip())
        old_q[k] += num(r["quantity"]) or 0
        in_photos = num(r.get("in_photos", ""))
        if had_photo_col and in_photos is not None:
            old_p[k] = old_p.get(k, 0) + in_photos

manual: Counter[StockKey] = Counter()
for k, q in old_q.items():
    p = old_p.get(k, photo[k]) if had_photo_col else photo[k]  # no column yet: assume photos unchanged
    manual[k] = q - p

keys = set(photo) | set(old_q)
new = {k: max(0, photo[k] + manual[k]) for k in keys}

if os.path.exists(STOCK):
    shutil.copyfile(STOCK, PREV)
order = sorted(new, key=lambda k: ("special" in k[0], k[4], k[1] * k[2], k[1], k[0], k[5]))
with open(STOCK, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(HEADER)
    for k in order:
        if new[k] == 0 and photo[k] == 0:
            continue
        w.writerow([*k[:5], new[k], photo[k], new[k] - photo[k], k[5], part_of.get(k) or old_part.get(k, "")])
tot_m = sum(new[k] - photo[k] for k in new)
print(f"catalog_stock.csv: {sum(new.values())} pieces = {sum(photo.values())} from photos + {tot_m} added by hand")
