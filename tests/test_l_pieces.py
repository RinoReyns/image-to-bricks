"""--use-l-pieces: owned L pieces (corner plate / brick 2x2, 3 studs) replace small pieces that cover exactly an
L-shaped area. Same cells, colour rules, stability not worse, stock ledger consistent, demand report, skip boxes.
Synthetic data, no mesh needed. Run: python tests/test_l_pieces.py"""

import contextlib
import csv
import io
import os
import shutil
import tempfile

import support

support.stub_mesh_libraries()

import numpy as np

from image_to_bricks.analysis import stability
from image_to_bricks.cli import parse_box
from image_to_bricks.instructions import fast_render as fr
from image_to_bricks.instructions import instruction_parts
from image_to_bricks.instructions import outputs_handler as oh
from image_to_bricks.mesh import voxelize
from image_to_bricks.model import LegoModel, PlacedBrick, export_inventory_csv, piece_cells
from image_to_bricks.options import BuildOptions, OutputOptions, RepairOptions
from image_to_bricks.packing import l_pieces, placement
from image_to_bricks.packing.l_pieces import Box, LOptions
from image_to_bricks.stock import Stock, make_key

BLACK, YELLOW, RED, GREEN, ORANGE = 6, 5, 2, 3, 9  # GREEN = filler colour that is never an owned L colour
L_PLATE, L_BRICK = "corner plate 2x2 (L, 3 studs)", "corner brick 2x2 (L, 3 studs)"
L_CELLS = {(1, 1), (2, 1), (1, 2)}  # the L used by most tests (notch = cell (2, 2))


# --- helpers ------------------------------------------------------------------------------------------------


def stock_with(owned):
    """Stock owning L pieces: {(height, colour): quantity}."""
    specials = {(h, 2, 2, col, L_PLATE if h == 1 else L_BRICK): n for (h, col), n in owned.items()}
    return Stock(specials=specials)


def filler(z, h, skip):
    """The cells of the 4x4 layer that are not in `skip`, as 1x1 GREEN pieces."""
    return [PlacedBrick(x, y, z, 1, 1, h, GREEN) for y in range(4) for x in range(4) if (x, y) not in skip]


def l_group(z, colour=BLACK):
    """A 1x2 + a 1x1 piece covering the L cells in layer z."""
    return [PlacedBrick(1, 1, z, 2, 1, 1, colour), PlacedBrick(1, 2, z, 1, 1, 1, colour)]


def scene(middle_pieces, middle_layers=1):
    """4x4 black floor (z 0), the given pieces from z 1, a 4x4 black lid on top, so everything is connected."""
    grid = np.ones((middle_layers + 2, 4, 4), bool)
    lid = PlacedBrick(0, 0, 1 + middle_layers, 4, 4, 1, BLACK)
    return grid, [PlacedBrick(0, 0, 0, 4, 4, 1, BLACK), lid] + middle_pieces


def plate_scene():
    return scene(l_group(1) + filler(1, 1, L_CELLS))


def swap(bricks, grid, stock, visible=None, **option_args):
    with contextlib.redirect_stdout(io.StringIO()):
        return l_pieces.use_l_pieces(bricks, grid, visible, stock, LOptions(**option_args))


def coverage(bricks, shape):
    """(times each cell is covered, colour of each cell) from the pieces' real footprints."""
    cover, colour = np.zeros(shape, int), np.full(shape, -1)
    for b in bricks:
        for x, y in piece_cells(b):
            cover[b.z : b.z + b.h_plates, y, x] += 1
            colour[b.z : b.z + b.h_plates, y, x] = b.color
    return cover, colour


def only_l(bricks):
    return [b for b in bricks if b.notch >= 0][0]


# --- swaps --------------------------------------------------------------------------------------------------


def test_plate_l_replaces_small_pieces_without_changing_cells_or_colours():
    grid, bricks = plate_scene()
    stock = stock_with({(1, BLACK): 2}).recount(bricks)
    cover0, colour0 = coverage(bricks, grid.shape)
    out, n = swap(bricks, grid, stock)
    cover1, colour1 = coverage(out, grid.shape)
    piece = only_l(out)
    assert n == 1 and len(out) == len(bricks) - 1
    assert (cover0 == cover1).all() and (cover1 == 1).all() and (colour0 == colour1).all()
    assert (piece.w, piece.d, piece.h_plates, piece.color, piece.notch) == (2, 2, 1, BLACK, 3)
    assert set(piece_cells(piece)) == L_CELLS


def test_ledger_matches_recount_of_final_pieces():
    grid, bricks = plate_scene()
    stock = stock_with({(1, BLACK): 2}).recount(bricks)
    out, _ = swap(bricks, grid, stock)
    final = Stock(specials=stock.specials).recount(out)
    assert final.l_used == stock.l_used and +final.used == +stock.used
    assert stock.l_used[(1, BLACK)] == 1 and stock.l_free(1, BLACK) == 1


def test_two_visible_colours_in_one_l_give_no_swap():
    grid, bricks = scene(
        [PlacedBrick(1, 1, 1, 2, 1, 1, BLACK), PlacedBrick(1, 2, 1, 1, 1, 1, RED)] + filler(1, 1, L_CELLS)
    )
    out, n = swap(bricks, grid, stock_with({(1, BLACK): 2, (1, RED): 2}))
    assert n == 0 and len(out) == len(bricks)


def test_visible_colour_that_is_not_owned_gives_no_swap():
    grid, bricks = plate_scene()
    _, n = swap(bricks, grid, stock_with({(1, YELLOW): 2}))
    assert n == 0


def test_hidden_cells_take_any_owned_colour():
    grid, bricks = plate_scene()
    hidden = np.ones(grid.shape, bool)
    hidden[:, [y for _, y in L_CELLS], [x for x, _ in L_CELLS]] = False
    out, n = swap(bricks, grid, stock_with({(1, YELLOW): 1}), visible=hidden)
    assert n == 1 and only_l(out).color == YELLOW


def test_piece_sticking_out_of_the_l_gives_no_swap():
    grid, bricks = scene([PlacedBrick(1, 1, 1, 2, 2, 1, BLACK)] + filler(1, 1, {(1, 1), (2, 1), (1, 2), (2, 2)}))
    _, n = swap(bricks, grid, stock_with({(1, BLACK): 2}))
    assert n == 0


def test_brick_l_replaces_six_plates_over_three_layers():
    middle = []
    for z in (1, 2, 3):
        middle += l_group(z) + filler(z, 1, L_CELLS)
    grid, bricks = scene(middle, middle_layers=3)
    stock = stock_with({(3, BLACK): 1}).recount(bricks)
    cover0, _ = coverage(bricks, grid.shape)
    out, n = swap(bricks, grid, stock)
    piece = only_l(out)
    assert n == 1 and (piece.h_plates, piece.z) == (3, 1) and len(out) == len(bricks) - 5
    assert (coverage(out, grid.shape)[0] == cover0).all()


def test_nothing_happens_without_owned_l_pieces():
    grid, bricks = plate_scene()
    assert swap(bricks, grid, Stock())[1] == 0
    assert swap(bricks, grid, None)[1] == 0


def test_never_more_l_pieces_than_owned():
    first = [PlacedBrick(0, 0, 1, 2, 1, 1, BLACK), PlacedBrick(0, 1, 1, 1, 1, 1, BLACK)]
    second = [PlacedBrick(2, 2, 1, 2, 1, 1, BLACK), PlacedBrick(3, 3, 1, 1, 1, 1, BLACK)]
    taken = {(0, 0), (1, 0), (0, 1), (2, 2), (3, 2), (3, 3)}
    grid, bricks = scene(first + second + filler(1, 1, taken))
    stock = stock_with({(1, BLACK): 1}).recount(bricks)
    _, n = swap(bricks, grid, stock)
    assert n == 1 and stock.l_used[(1, BLACK)] == 1


# --- contacts, outputs ---------------------------------------------------------------------------------------


def _model_with_l():
    grid, bricks = plate_scene()
    out, _ = swap(bricks, grid, stock_with({(1, BLACK): 1}).recount(bricks))
    return LegoModel(grid=grid, bricks=out)


def test_l_piece_touches_the_layers_above_and_below_with_its_three_studs():
    model = _model_with_l()
    adj = stability.build_contacts(model)
    index = next(i for i, b in enumerate(model.bricks) if b.notch >= 0)
    assert sum(adj[index].values()) == 6  # 3 studs down + 3 studs up
    assert stability.report(model, None, "t", verbose=False)["disconnected_pieces"] == 0


def test_inventory_csv_lists_the_l_piece_as_special_line():
    handle, path = tempfile.mkstemp(suffix=".csv")
    os.close(handle)
    with contextlib.redirect_stdout(io.StringIO()):
        export_inventory_csv(_model_with_l(), path)
    rows = list(csv.reader(open(path)))
    os.remove(path)
    assert any("special" in r[1] and r[7].startswith("corner 2x2") for r in rows[1:] if len(r) > 7)


def test_outputs_draw_the_l_piece_and_keep_the_notch_cell_of_its_neighbour():
    model = _model_with_l()
    piece = only_l(model.bricks)
    colours = oh.model_colors(model)
    notch_x, notch_y = piece.x + (piece.notch & 1), piece.y + (piece.notch >> 1)
    assert (colours[piece.z, notch_y, notch_x] == instruction_parts.colour_rgb(GREEN)).all()
    folder = tempfile.mkdtemp()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            oh.save_outputs(model, "t", folder, options=OutputOptions(make_pdf=True))
        assert os.path.exists(os.path.join(folder, "t_instructions.pdf"))
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    table = oh.parts_table([piece])
    assert any("Corner" in str(cell) for row in table._cellvalues for cell in row)


def test_l_outline_is_three_quarters_of_the_box_for_every_notch():
    for notch in range(4):
        pts = fr._l_outline(0, 0, 100, 100, notch)
        area = abs(sum(pts[i][0] * pts[(i + 1) % 6][1] - pts[(i + 1) % 6][0] * pts[i][1] for i in range(6))) / 2
        assert len(pts) == 6 and abs(area - 7500) < 1e-6


# --- demand report, skip boxes, CLI helpers --------------------------------------------------------------------


def test_demand_counts_places_for_colours_that_are_not_owned():
    grid, bricks = plate_scene()
    places, saved = l_pieces.demand(bricks, grid, None)
    assert places[(1, BLACK)] >= 1 and saved >= 1


def test_demand_lists_hidden_only_places_under_none():
    grid, bricks = plate_scene()
    hidden = np.ones(grid.shape, bool)
    hidden[1] = False
    places, _ = l_pieces.demand(bricks, grid, hidden)
    assert places[(1, None)] >= 1


def test_demand_table_is_printed_even_when_no_l_piece_is_owned():
    grid, bricks = plate_scene()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        l_pieces.use_l_pieces(bricks, grid, None, stock_with({(1, YELLOW): 1}))
    text = buffer.getvalue()
    assert "L demand" in text and "Black" in text and "L pieces: 1 owned" in text


def test_skip_box_keeps_pieces_out_of_the_swap_and_the_demand():
    grid, bricks = plate_scene()
    box = Box(1, 2, 1, 3, 1, 3)
    out, n = swap(bricks, grid, stock_with({(1, BLACK): 2}), skip=(box,))
    assert n == 0 and out == bricks
    assert l_pieces.demand(bricks, grid, None, (box,))[0].get((1, BLACK), 0) == 0
    far = Box(5, 6, 0, 1, 0, 1)
    assert swap(bricks, grid, stock_with({(1, BLACK): 2}), skip=(far,))[1] >= 1


def test_box_touches_only_overlapping_pieces():
    box = Box(1, 2, 1, 3, 1, 3)
    assert box.touches(PlacedBrick(2, 2, 1, 2, 2, 1, BLACK))
    assert not box.touches(PlacedBrick(3, 1, 1, 1, 1, 1, BLACK))  # next to it in x
    assert not box.touches(PlacedBrick(1, 1, 2, 1, 1, 1, BLACK))  # one layer above


def test_parse_box_and_its_error_message():
    assert parse_box("75:76,4:6,11:12", "--x") == (slice(75, 76), slice(4, 6), slice(11, 12))
    for bad in ("1:2,3:4", "a:b,1:2,3:4", "1:2:3,4:5,6:7"):
        try:
            parse_box(bad, "--l-skip-box")
        except SystemExit as error:
            assert "--l-skip-box" in str(error)
        else:
            raise AssertionError(f"{bad!r} should be rejected")


def test_figure_box_is_shifted_by_base_layers_and_margin():
    box = placement._figure_box_to_grid((slice(75, 76), slice(4, 6), slice(11, 12)), 4, 4)
    assert box == Box(79, 80, 8, 10, 15, 16)


# --- whole pipeline ------------------------------------------------------------------------------------------


def _blob(seed):
    rng = np.random.default_rng(seed)
    grid = np.zeros((12, 18, 22), bool)
    for z in range(12):
        grid[z, 2:16, 2:20] = True
        for _ in range(14):  # ragged holes -> awkward leftover shapes
            y, x = rng.integers(2, 15), rng.integers(2, 19)
            grid[z, y : y + rng.integers(1, 3), x : x + rng.integers(1, 3)] = False
    colours = np.full(grid.shape, ORANGE, np.uint8)
    colours[:, :, 11:] = YELLOW
    colours[::2][:, 5:9, 4:8] = BLACK
    return grid, colours


def _pipeline(grid, colours, use_l, stock):
    options = BuildOptions(repair=RepairOptions(use_l_pieces=use_l))
    with contextlib.redirect_stdout(io.StringIO()):
        return placement.optimize_brick_placement(
            placement.PackingCells(grid, colours, visible=voxelize.surface_mask(grid), base_layers=0),
            options=options,
            stock=stock,
        )


def test_whole_pipeline_keeps_cells_visible_colours_and_ledger():
    placed_total = 0
    for seed in range(4):
        grid, colours = _blob(seed)
        owned = Stock(
            {make_key(1, 4, 2, ORANGE): 5},
            specials={
                (1, 2, 2, ORANGE, L_PLATE): 6,
                (1, 2, 2, YELLOW, L_PLATE): 6,
                (3, 2, 2, ORANGE, L_BRICK): 4,
                (3, 2, 2, BLACK, L_BRICK): 4,
            },
        )
        without = _pipeline(grid, colours, False, owned.fork())
        ledger = owned.fork()
        with_l = _pipeline(grid, colours, True, ledger)
        cover0, colour0 = coverage(without.bricks, grid.shape)
        cover1, colour1 = coverage(with_l.bricks, grid.shape)
        visible = voxelize.surface_mask(grid) & grid
        assert (cover1 == with_l.grid.astype(int)).all() and (cover0 == cover1).all()
        assert (colour0[visible] == colour1[visible]).all()
        final = owned.recount(with_l.bricks)
        assert final.l_used == ledger.l_used and +final.used == +ledger.used
        assert all(n <= owned.l_have[k] for k, n in final.l_used.items())
        assert len(with_l.bricks) <= len(without.bricks)
        assert (
            stability.report(with_l, None, "t", verbose=False)["disconnected_pieces"]
            <= stability.report(without, None, "t", verbose=False)["disconnected_pieces"]
        )
        placed_total += sum(b.notch >= 0 for b in with_l.bricks)
    assert placed_total > 0


def run_all():
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"ok   {name}")
    print(f"all {len(tests)} passed")


if __name__ == "__main__":
    run_all()
