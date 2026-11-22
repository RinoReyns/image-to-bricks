"""--stock ownership per layer (owned_layers.py, PDF Own / Buy columns, <name>_owned_layers.csv) and
--base-fill-colors (base_fill.py: base re-tiled with spare owned plates, figure pieces never taken)."""

import contextlib
import io
import os
import sys
import tempfile

import support

support.stub_mesh_libraries()
import numpy as np

from image_to_bricks.base import base_fill
from image_to_bricks.instructions import outputs_handler as oh
from image_to_bricks.instructions import owned_layers
from image_to_bricks.model import LegoModel, PlacedBrick
from image_to_bricks.options import OutputOptions
from image_to_bricks.stock import Stock, make_key

GREEN, BLACK, BLUE, YELLOW = 3, 6, 4, 5
failed = False


def check(name, ok, info=None):
    global failed
    print(("ok   " if ok else "FAIL ") + name + ("" if ok or info is None else f"  {info}"))
    failed |= not ok


def tower(owned_black_1x1: int):
    """4 layers of one Black 1x1 plate each, a stock with `owned_black_1x1` of them."""
    bricks = [PlacedBrick(0, 0, z, 1, 1, 1, BLACK) for z in range(4)]
    return bricks, Stock(have={make_key(1, 1, 1, BLACK): owned_black_1x1})


def test_owned_pieces_are_handed_out_bottom_up():
    bricks, stock = tower(2)
    is_owned = owned_layers.allocate_owned_pieces(bricks, stock)
    check(
        "the two lowest pieces are owned, the rest to buy",
        [is_owned[id(b)] for b in sorted(bricks, key=lambda b: b.z)] == [True, True, False, False],
    )
    layers = owned_layers.ownership_by_layer(bricks, is_owned)
    check("complete layers from the bottom = 2", owned_layers.count_complete_layers_from_bottom(layers) == 2)
    check("layer 3 lists the piece to buy", layers[2].to_buy_by_piece == {make_key(1, 1, 1, BLACK): 1})


def test_a_gap_stops_the_buildable_count():
    bricks = [
        PlacedBrick(0, 0, 0, 1, 1, 1, BLACK),
        PlacedBrick(0, 0, 1, 1, 1, 1, YELLOW),
        PlacedBrick(0, 0, 2, 1, 1, 1, BLACK),
    ]
    stock = Stock(have={make_key(1, 1, 1, BLACK): 5})
    layers = owned_layers.ownership_by_layer(bricks, owned_layers.allocate_owned_pieces(bricks, stock))
    check(
        "layer 2 not owned -> buildable 1, complete anywhere 2",
        owned_layers.count_complete_layers_from_bottom(layers) == 1 and sum(r.is_complete for r in layers) == 2,
    )


def test_parts_table_has_own_and_buy_columns():
    bricks, stock = tower(3)
    is_owned = owned_layers.allocate_owned_pieces(bricks, stock)
    table = oh.parts_table(bricks, is_owned=is_owned)
    check("header Own / Buy", table._cellvalues[0][-2:] == ["Own", "Buy"], table._cellvalues[0])
    check("row: 4 needed, 3 own, 1 buy", table._cellvalues[1][-3:] == ["4", "3", "1"], table._cellvalues[1])
    plain = oh.parts_table(bricks)
    check("without a stock the table is unchanged (5 columns)", len(plain._cellvalues[0]) == 5)


def test_save_outputs_writes_the_layer_report_and_pdf():
    bricks, stock = tower(2)
    model = LegoModel(grid=np.ones((4, 1, 1), bool), bricks=bricks, base_layers=0)
    folder = tempfile.mkdtemp()
    with contextlib.redirect_stdout(io.StringIO()) as log:
        oh.save_outputs(model, "t", folder, stock, OutputOptions(make_pdf=True))
    rows = open(os.path.join(folder, "t_owned_layers.csv")).read().splitlines()
    check("csv: header + 4 layers", len(rows) == 5 and rows[1].startswith("Layer 1,1,1,0,yes,yes"), rows)
    check(
        "console line names the last buildable layer", "you can build up to: Layer 2" in log.getvalue(), log.getvalue()
    )
    check("pdf written", os.path.getsize(os.path.join(folder, "t_instructions.pdf")) > 1000)


def base_with_short_green():
    """Base of 2 layers (4 x 2 studs): hidden layer one Black 4x2 (owned), top layer two Green 2x2 (1 owned).
    Figure: a Blue 2x2 plate on top. Stock: spare Blue 2x2 x1 beyond the figure's one, spare Black 4x2 none."""
    bricks = [
        PlacedBrick(0, 0, 0, 4, 2, 1, BLACK),
        PlacedBrick(0, 0, 1, 2, 2, 1, GREEN),
        PlacedBrick(2, 0, 1, 2, 2, 1, GREEN),
        PlacedBrick(0, 0, 2, 2, 2, 1, BLUE),
    ]
    stock = Stock(have={make_key(1, 4, 2, BLACK): 1, make_key(1, 2, 2, GREEN): 1, make_key(1, 2, 2, BLUE): 2})
    for piece in bricks:
        stock.take_brick(piece)
    return bricks, stock


def no_loose(_bricks) -> int:
    return 0


def test_short_base_piece_takes_a_spare_fill_colour():
    bricks, stock = base_with_short_green()
    options = base_fill.BaseFillOptions([BLUE], hidden_layers=1, buy_sizes=[(2, 2)])
    refills = base_fill.fill_base_from_spare(bricks, 2, stock, options, no_loose)
    top = sorted((b.x, b.color) for b in bricks if b.z == 1)
    check("one Green 2x2 became the spare Blue 2x2", top in ([(0, GREEN), (2, BLUE)], [(0, BLUE), (2, GREEN)]), top)
    check("nothing left to buy", base_fill.count_to_buy(stock, stock.recount(bricks).used) == 0)
    check("ledger == final pieces", +stock.recount(bricks).used == +stock.used)
    check("the figure keeps its Blue plate", any(b.z == 2 and b.color == BLUE for b in bricks))
    check("one layer reported, 1 -> 0 to buy", len(refills) == 1 and refills[0].to_buy_after == 0, refills)


def test_figure_pieces_are_never_taken():
    bricks, stock = base_with_short_green()
    stock.have[make_key(1, 2, 2, BLUE)] = 1  # only the figure's own Blue plate is owned
    options = base_fill.BaseFillOptions([BLUE], hidden_layers=1, buy_sizes=[(2, 2)])
    refills = base_fill.fill_base_from_spare(bricks, 2, stock, options, no_loose)
    check("no spare Blue -> base unchanged", refills == [] and sum(b.color == GREEN for b in bricks) == 2)


def test_colour_not_in_the_fill_list_is_not_used_on_visible_layers():
    bricks, stock = base_with_short_green()
    stock.have[make_key(1, 2, 2, YELLOW)] = 3
    options = base_fill.BaseFillOptions([BLACK], hidden_layers=1, buy_sizes=[(2, 2)])
    base_fill.fill_base_from_spare(bricks, 2, stock, options, no_loose)
    check("spare Yellow is not used on the visible top", not any(b.color == YELLOW for b in bricks))


def test_hidden_layer_takes_any_spare_colour():
    bricks = [PlacedBrick(0, 0, 0, 2, 2, 1, GREEN), PlacedBrick(0, 0, 1, 2, 2, 1, GREEN)]
    stock = Stock(have={make_key(1, 2, 2, GREEN): 1, make_key(1, 2, 2, YELLOW): 1})
    for piece in bricks:
        stock.take_brick(piece)
    options = base_fill.BaseFillOptions([BLACK], hidden_layers=1, buy_sizes=[(2, 2)])
    base_fill.fill_base_from_spare(bricks, 2, stock, options, no_loose)
    check(
        "hidden layer: Green short -> spare Yellow",
        [b.color for b in sorted(bricks, key=lambda b: b.z)] == [YELLOW, GREEN],
        [(b.z, b.color) for b in bricks],
    )


def test_refill_that_makes_loose_pieces_is_rejected():
    bricks, stock = base_with_short_green()
    options = base_fill.BaseFillOptions([BLUE], hidden_layers=1, buy_sizes=[(2, 2)])
    calls = []

    def loose_after_change(trial_bricks) -> int:
        calls.append(1)
        return 0 if len(calls) == 1 else 1  # before: 0, every trial: 1

    refills = base_fill.fill_base_from_spare(bricks, 2, stock, options, loose_after_change)
    check(
        "a re-tiling that adds a loose piece is not kept", refills == [] and sum(b.color == GREEN for b in bricks) == 2
    )


def test_cli_needs_stock():
    source = open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "image_to_bricks", "cli.py"), encoding="utf-8"
    ).read()
    check(
        "--base-fill-colors defined, needs --stock, stock passed to save_outputs",
        '"--base-fill-colors"' in source
        and "--base-fill-colors needs --stock" in source
        and "save_outputs(model, model_name, outdir, stock" in source,
    )


if __name__ == "__main__":
    for test in [value for key, value in list(globals().items()) if key.startswith("test_")]:
        test()
    sys.exit(1 if failed else 0)
