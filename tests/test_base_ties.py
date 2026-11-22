"""--base-sparse-layers (base_ties.py): sparse tie layer under the base plates, ties every N studs of seam + feet."""

import contextlib
import io
import os
import sys

import support

support.stub_mesh_libraries()
import numpy as np

from image_to_bricks.analysis import stability
from image_to_bricks.base import base_packer as bp
from image_to_bricks.base import base_ties as bt
from image_to_bricks.base import stand
from image_to_bricks.mesh import voxelize
from image_to_bricks.options import BaseOptions, BuildOptions, RepairOptions
from image_to_bricks.packing import placement
from image_to_bricks.stock import Stock

GREEN, ORANGE, BLUE, BROWN = 3, 9, 21, 11
failed = False


def check(name, ok, info=None):
    global failed
    print(("ok   " if ok else "FAIL ") + name + ("" if ok or info is None else f"  {info}"))
    failed |= not ok


def build(plates=2, hidden=1, sparse=1, tie_every=4, stock=None, zones=()):
    """Figure (4x4 tower) on a layered base through the pipeline; returns (model, log)."""
    base = BaseOptions(
        plates=plates,
        margin_studs=5,
        shape="rounded",
        corner_radius_studs=3,
        colour_id=GREEN,
        zones=list(zones),
        zone_seed=1,
        footprint_layers=3,
        layered=bp.LayeredBaseOptions(hidden, sparse_layers=sparse, tie_every=tie_every),
    )
    g = np.zeros((12, 14, 20), bool)
    g[:, 5:9, 6:10] = True
    c = np.full(g.shape, ORANGE, np.uint8)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        g2, c2, v2, mask = stand.add_base(g, c, voxelize.surface_mask(g), base)
        model = placement.optimize_brick_placement(
            placement.PackingCells(g2, c2, visible=v2, base_layers=plates),
            options=BuildOptions(base=base, repair=RepairOptions(tie_repair=False)),
            stock=stock,
        )
    return model, mask, out.getvalue()


def layer(model, z):
    return [b for b in model.bricks if b.z == z]


def test_units():
    ids = np.array([[0, 0, 1, 1, 1, 1, 1, 1, 1]])
    mask = np.ones_like(ids, bool)
    seams = bt.seam_edges(ids, mask)
    check("seam_edges: one seam between pieces 0 and 1", list(seams) == [(0, 1)] and len(seams[(0, 1)]) == 1)
    tall = np.array([[0, 1]] * 9)
    reqs = bt.requirements(tall, np.ones_like(tall, bool), 4, with_feet=False)
    check("a 9-stud seam needs ceil(9/4) = 3 ties", len(reqs) == 3, len(reqs))
    feet = bt.corner_cells(np.ones((5, 7), bool))
    check("corner cells of a rectangle are its 4 corners", sorted(feet) == [(0, 0), (0, 6), (4, 0), (4, 6)], feet)


def test_sparse_layer_ties_every_seam_and_has_feet():
    model, mask, log = build()
    check("summary reports the sparse layer", "sparse" in log and "Layered base:" in log, log[-300:])
    check("no untied seam / missing foot WARNING", "could not be placed" not in log, log[-300:])
    top_ids = bp.piece_ids(layer(model, 1), mask.shape)
    reqs = bt.requirements(top_ids, mask, 4, with_feet=True)
    need = dict.fromkeys(np.unique(top_ids[top_ids >= 0]), 1)  # relaxed ties (1 stud) are allowed as fallback
    ties = layer(model, 0)

    def met(r):
        for b in ties:
            held = {}
            for yy in range(b.y, b.y + b.d):
                for xx in range(b.x, b.x + b.w):
                    held[int(top_ids[yy, xx])] = held.get(int(top_ids[yy, xx]), 0) + 1
            if bt._meets(r, (b.x, b.y, b.w, b.d), held, need):
                return True
        return False

    check(
        "every seam chunk and every corner foot is met by a tie (at least 1 stud each side)",
        all(met(r) for r in reqs),
        sum(not met(r) for r in reqs),
    )


def test_sparse_layer_has_holes_and_grid_matches_pieces():
    model, mask, _ = build()
    cov = np.zeros(mask.shape, int)
    for b in layer(model, 0):
        cov[b.y : b.y + b.d, b.x : b.x + b.w] += 1
    check("sparse layer covers fewer cells than the outline", cov.sum() < mask.sum(), (cov.sum(), mask.sum()))
    check("no overlap, nothing outside the outline", cov.max() <= 1 and not (cov[~mask] > 0).any())
    check("grid bottom layer == cells the tie pieces cover", (model.grid[0] == (cov > 0)).all())
    check("top base layer still full", (model.grid[1] == mask).all())


def test_fewer_pieces_than_full_layer():
    sparse = len(layer(build(sparse=1)[0], 0))
    full = len(layer(build(sparse=0)[0], 0))
    check("sparse bottom layer has fewer pieces than a full one", sparse < full, (sparse, full))


def test_one_assembly_and_stands():
    model, _, log = build(zones=[(BLUE, "pond", "right", 20), (BROWN, "path", "down", 3)])
    comp, _, _ = stability.analyse_graph(stability.build_contacts(model))
    check("figure + base are one assembly", len(set(comp)) == 1, len(set(comp)))
    with contextlib.redirect_stdout(io.StringIO()):
        res = stability.stability_check(model, 15.0)
    check("feet keep the tip angle >= 15 deg", res.get("ok"), res.get("tip_angle"))


def test_tie_every_controls_density():
    dense = len(layer(build(tie_every=1)[0], 0))
    loose = len(layer(build(tie_every=8)[0], 0))
    check("--base-tie-every 1 needs at least as many ties as 8", dense >= loose, (dense, loose))


def test_stock_ledger_and_owned_ties():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_base_ties_stock.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "piece_type,width_studs,depth_studs,height_plates,color_name,quantity\n"
            "Plate,8,2,1,Yellow,6\nPlate,2,2,1,Black,6\n"
        )
    with contextlib.redirect_stdout(io.StringIO()):
        stock = Stock.from_csv(path)
    os.remove(path)
    model, _, _ = build(stock=stock)
    recount = stock.recount(model.bricks)
    check("stock ledger == recount of the final pieces", +stock.used == +recount.used)
    owned = sum(1 for b in layer(model, 0) if (max(b.w, b.d), min(b.w, b.d), b.color) in ((8, 2, 5), (2, 2, 6)))
    check("ties use owned plates", owned >= 1, owned)


def test_full_layers_unchanged_by_option_default():
    a = [vars(b) for b in build(plates=3, hidden=2, sparse=0)[0].bricks]
    b = [vars(b) for b in build(plates=3, hidden=2, sparse=0)[0].bricks]
    check("sparse_layers 0 = full layers, deterministic", a == b)
    try:
        bp.pack_layered_base(
            bp.BaseCells(np.ones((3, 4, 4), bool), np.zeros((3, 4, 4), np.uint8), np.full((4, 4), -1), 3),
            None,
            bp.LayeredBaseOptions(1, sparse_layers=2),
            GREEN,
        )
        check("more sparse than hidden layers is rejected", False)
    except ValueError:
        check("more sparse than hidden layers is rejected", True)


if __name__ == "__main__":
    for fn in [v for k, v in list(globals().items()) if k.startswith("test_")]:
        fn()
    sys.exit(1 if failed else 0)
