"""--base-hidden-layers (base_packer.py): layered base of big plates, hidden layers any colour, seams offset."""

import argparse
import contextlib
import io
import os
import sys

import support

support.stub_mesh_libraries()
import numpy as np

from image_to_bricks.base import base_packer as bp
from image_to_bricks.base import stand
from image_to_bricks.mesh import voxelize
from image_to_bricks.options import BaseOptions, BuildOptions, RepairOptions
from image_to_bricks.packing import placement
from image_to_bricks.stock import Stock

GREEN, BLUE, BROWN, BLACK, YELLOW, ORANGE = 3, 21, 11, 6, 5, 9
failed = False


def check(name, ok, info=None):
    global failed
    print(("ok   " if ok else "FAIL ") + name + ("" if ok or info is None else f"  {info}"))
    failed |= not ok


def figure():
    g = np.zeros((12, 14, 20), bool)
    g[:, 5:9, 6:10] = True  # a 4x4 tower in the middle
    return g, np.full(g.shape, ORANGE, np.uint8)


def base_options(plates, hidden, zones=(), max_plate=bp.BASE_MAX_PLATE, shape="rounded"):
    """The base options of these tests."""
    return BaseOptions(
        plates=plates,
        margin_studs=5,
        shape=shape,
        corner_radius_studs=3,
        colour_id=GREEN,
        zones=list(zones),
        zone_seed=1,
        footprint_layers=3,
        layered=bp.LayeredBaseOptions(hidden, max_plate=max_plate) if hidden else None,
    )


def build(plates, hidden, stock=None, **kw):
    """Figure + base through the pipeline (add_base + optimize_brick_placement); returns (model, colours, log)."""
    base = base_options(plates, hidden, **kw)
    g, c = figure()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        g2, c2, v2, _ = stand.add_base(g, c, voxelize.surface_mask(g), base)
        model = placement.optimize_brick_placement(
            placement.PackingCells(g2, c2, visible=v2, base_layers=plates),
            options=BuildOptions(base=base, repair=RepairOptions(tie_repair=False)),
            stock=stock,
        )
    return model, c2, v2, out.getvalue()


def base_pieces(model):
    return [b for b in model.bricks if b.z < model.base_layers]


def coverage(model, plates):
    cov = np.zeros((plates,) + model.grid.shape[1:], int)
    for b in base_pieces(model):
        cov[b.z, b.y : b.y + b.d, b.x : b.x + b.w] += 1
    return cov


def stock_csv(rows):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_base_packer_stock.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("piece_type,width_studs,depth_studs,height_plates,color_name,quantity\n")
        f.writelines(r + "\n" for r in rows)
    with contextlib.redirect_stdout(io.StringIO()):
        stock = Stock.from_csv(path)
    os.remove(path)
    return stock


def test_exact_cover_and_plates_only():
    model, _, _, log = build(3, 2)
    cov = coverage(model, 3)
    mask = model.grid[0]
    check(
        "every base cell covered exactly once, nothing outside",
        (cov[:, mask] == 1).all() and (cov[:, ~mask] == 0).all(),
    )
    check("base is plates only", all(b.h_plates == 1 for b in base_pieces(model)))
    check("summary line printed", "Layered base:" in log, log[-400:])
    check("no loose base pieces", "WARNING" not in log, log[-400:])


def test_fewer_pieces_than_plate_base():
    layered = len(base_pieces(build(3, 2)[0]))
    old = len(base_pieces(build(3, 0)[0]))
    check("layered base needs fewer pieces than the plate packer (no stock)", layered < old, (layered, old))


def test_visible_layer_keeps_zone_colours():
    zones = [(BLUE, "pond", "right", 20), (BROWN, "path", "down", 3)]
    model, colours, visible, _ = build(3, 2, zones=zones)
    top = 2
    ok = True
    for b in base_pieces(model):
        if b.z != top:
            continue
        cells = colours[top, b.y : b.y + b.d, b.x : b.x + b.w][visible[top, b.y : b.y + b.d, b.x : b.x + b.w]]
        ok &= bool((cells == b.color).all())
    check("top-layer pieces have the zone colour of their visible cells", ok)
    colours_top = {b.color for b in base_pieces(model) if b.z == top}
    check("pond and path colours are built", {BLUE, BROWN} <= colours_top, colours_top)


def test_hidden_layers_not_visible():
    _, _, visible, _ = build(3, 2)
    check("hidden layers are not visible (any colour, also the rim)", not visible[:2].any())
    check("top base layer rim stays visible", visible[2].any())


def test_hidden_layers_use_owned_spare_plates_of_any_colour():
    stock = stock_csv(["Plate,8,4,1,Yellow,6", "Plate,8,4,1,Black,6", "Plate,12,6,1,Blue,2"])
    model, _, _, _ = build(3, 2, stock=stock)
    hidden = [b for b in base_pieces(model) if b.z < 2]
    colours = {b.color for b in hidden}
    check("hidden layers use owned colours other than the base colour", colours & {YELLOW, BLACK, BLUE}, colours)
    big = sum(1 for b in hidden if b.w * b.d >= 32)
    check("hidden layers use the big owned plates", big >= 4, big)


def test_stock_ledger_matches_final_pieces():
    stock = stock_csv(["Plate,8,4,1,Yellow,3", "Plate,6,6,1,Green,1", "Plate,4,2,1,Green,5"])
    model, _, _, _ = build(3, 2, stock=stock)
    recount = stock.recount(model.bricks)
    check(
        "stock ledger == recount of the final pieces",
        +stock.used == +recount.used,
        (stock.used - recount.used, recount.used - stock.used),
    )


def test_never_uses_more_owned_than_have_for_unbuyable_sizes():
    stock = stock_csv(["Plate,8,4,1,Yellow,1"])
    model, _, _, _ = build(3, 2, stock=stock)
    n = sum(1 for b in base_pieces(model) if {b.w, b.d} == {8, 4})
    check("an owned size that cannot be bought is used at most as often as owned", n <= 1, n)


def test_max_plate_caps_piece_size():
    stock = stock_csv(["Plate,12,6,1,Black,4"])
    model, _, _, _ = build(3, 2, stock=stock, max_plate=(4, 4))
    biggest = max(max(b.w, b.d) for b in base_pieces(model))
    check("--base-max-plate 4x4: no base plate longer than 4", biggest <= 4, biggest)


def test_seams_offset_between_layers():
    model, _, _, log = build(3, 2, shape="rect")
    ids = {z: bp.piece_ids([b for b in base_pieces(model) if b.z == z], model.grid.shape[1:]) for z in range(3)}
    for upper, lower in ((2, 1), (1, 0)):
        share = bp.bridged_fraction(ids[upper], ids[lower])
        check(f"layer {lower} spans >= 75 % of the joints of layer {upper}", share >= bp.BASE_MIN_BRIDGED, share)


def test_deterministic():
    a = [vars(b) for b in build(3, 2)[0].bricks]
    b = [vars(b) for b in build(3, 2)[0].bricks]
    check("same options give the same base", a == b)


def test_old_base_unchanged_without_option():
    a = [vars(b) for b in build(4, 0)[0].bricks]
    b = [vars(b) for b in build(4, 0)[0].bricks]
    check(
        "without --base-hidden-layers the old packer runs (layered option off)",
        a == b and base_options(4, 0).layered is None,
    )


def test_units():
    check("parse_plate_size orders long side first", bp.parse_plate_size("4x8") == (8, 4))
    for bad in ("8", "0x4", "axb"):
        try:
            bp.parse_plate_size(bad)
            check(f"parse_plate_size rejects {bad!r}", False)
        except ValueError:
            check(f"parse_plate_size rejects {bad!r}", True)
    upper = np.array([[0, 0, 1, 1]])
    check("bridged_fraction: joint inside one lower piece", bp.bridged_fraction(upper, np.array([[5, 6, 6, 7]])) == 1.0)
    check("bridged_fraction: joint on a lower joint", bp.bridged_fraction(upper, np.array([[5, 5, 6, 6]])) == 0.0)
    try:
        bp.pack_layered_base(
            bp.BaseCells(np.ones((2, 4, 4), bool), np.zeros((2, 4, 4), np.uint8), np.full((4, 4), -1), 2),
            None,
            bp.LayeredBaseOptions(2),
            GREEN,
        )
        check("all layers hidden is rejected", False)
    except ValueError:
        check("all layers hidden is rejected", True)


def test_cli_validation():
    from image_to_bricks import cli

    def valid_args() -> argparse.Namespace:
        return argparse.Namespace(
            base=3, base_hidden_layers=2, base_bricks=0, base_max_plate=None, base_sparse_layers=0, base_tie_every=4
        )

    layered = cli._layered_base_options(valid_args())
    check("CLI builds options", layered is not None and layered.hidden_layers == 2)
    for change, name in (
        ({"base": 2}, "too few base plates"),
        ({"base_bricks": 1}, "combined with --base-bricks"),
        ({"base_hidden_layers": 0, "base_max_plate": (4, 4)}, "--base-max-plate alone"),
        ({"base_sparse_layers": 3}, "more sparse than hidden layers"),
        ({"base_hidden_layers": 0, "base_sparse_layers": 1}, "--base-sparse-layers alone"),
        ({"base_tie_every": 0}, "--base-tie-every 0"),
    ):
        args = valid_args()
        for k, v in change.items():
            setattr(args, k, v)
        try:
            cli._layered_base_options(args)
            check(f"CLI rejects {name}", False)
        except SystemExit:
            check(f"CLI rejects {name}", True)


if __name__ == "__main__":
    for fn in [v for k, v in list(globals().items()) if k.startswith("test_")]:
        fn()
    sys.exit(1 if failed else 0)
