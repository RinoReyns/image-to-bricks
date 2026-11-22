"""--open-gaps (gap_spurs.py), --fill-pinholes (pinholes.py), --despeckle and --neutral-enclaves (color_cleanup.py)."""

import contextlib
import io
import sys

import support

support.stub_mesh_libraries()
import numpy as np

from image_to_bricks.colours import color_cleanup as cc
from image_to_bricks.colours import gap_spurs, palette_matching, pinholes, surface_cleanup
from image_to_bricks.consts import LEGO_RGB
from image_to_bricks.mesh import shape_edits
from image_to_bricks.options import ColourOptions, Paint, ShapeOptions

ORANGE, DARK_ORANGE, BLACK, LBG, RED, WHITE, BLO = 9, 19, 6, 7, 2, 1, 16
failed = False


def check(name, ok, info=None):
    global failed
    print(("ok   " if ok else "FAIL ") + name + ("" if ok or info is None else f"  {info}"))
    failed |= not ok


def palette_lab():
    return {c: palette_matching.rgb_to_lab(np.array([rgb]))[0] for c, rgb in LEGO_RGB.items()}


# ---- pinholes -------------------------------------------------------------------------------------------
def diagonal_sheet():
    """A 1-cell-thick sheet stepping diagonally (x = y + 2) for 12 layers, standing on a 10 x 10 slab."""
    g = np.zeros((14, 12, 14), bool)
    g[0:2, 1:11, 1:13] = True
    for z in range(2, 14):
        for y in range(1, 10):
            g[z, y, y + 2] = True
    return g


def test_pinhole_in_a_diagonal_sheet_is_closed_in_the_sheet():
    g = diagonal_sheet()
    broken = g.copy()
    broken[7, 5, 7] = False  # one cell lost: see-through from the side
    fixed, added = pinholes.fill_pinholes(broken, 3)
    check("one pinhole -> one cell added", added == 1, added)
    check("the lost sheet cell is restored", fixed[7, 5, 7])
    check("nothing else changes", (fixed == g).all())


def test_real_gap_stays_open():
    g = np.zeros((12, 6, 10), bool)
    g[:, 1:5, 1:3] = True  # two legs ...
    g[:, 1:5, 7:9] = True
    g[8:, 1:5, 1:9] = True  # ... joined at the hips ...
    g[:2, 1:5, 1:9] = True  # ... and standing on a base: an enclosed 6 x 4 gap between them
    fixed, added = pinholes.fill_pinholes(g, 3)
    check("a gap bigger than the limit stays open", added == 0 and (fixed == g).all(), added)
    _, added = pinholes.fill_pinholes(g, 40)
    check("a bigger limit would fill it", added > 0)


def test_no_holes_no_change():
    g = np.zeros((5, 5, 5), bool)
    g[1:4, 1:4, 1:4] = True
    fixed, added = pinholes.fill_pinholes(g, 3)
    check("solid block unchanged", added == 0 and (fixed == g).all())


# ---- gap spurs ------------------------------------------------------------------------------------------
def two_boots():
    """Two 3 x 3 boots (x 1-3 and x 5-7) 6 layers high, joined above (z 6-7), a 1-cell gap at x 4."""
    g = np.zeros((8, 6, 9), bool)
    g[:, 1:4, 1:4] = True
    g[:, 1:4, 5:8] = True
    g[6:, 1:4, 1:8] = True
    return g


def test_heel_spur_between_boots_is_removed():
    g = two_boots()
    g[2, 4, 4] = True  # sticks out of nothing ... make it hang off the left boot:
    g[2, 4, 3] = True  # left boot heel cell, then the spur at x 4 touching x 5 at a corner
    spurs = gap_spurs.gap_spurs(g)
    check("the spur in the gap is found", (2, 4, 4) in spurs, spurs)
    out, removed = gap_spurs.open_gaps(g)
    check("only that cell is removed", removed == [(2, 4, 4)] and not out[2, 4, 4] and out[2, 4, 3], removed)


def test_details_are_kept():
    g = two_boots()
    g[0:3, 4, 4] = True  # a 3-layer post in the gap: cells have neighbours above / below
    check("a column with cells above / below is kept", gap_spurs.gap_spurs(g) == [])
    g = two_boots()
    g[3, 2, 8] = True  # a 1-cell bump on the outside: links nothing
    check("a bump that links nothing is kept", gap_spurs.gap_spurs(g) == [])


def test_open_gaps_then_pinholes_keeps_a_sheet_closed():
    g = diagonal_sheet()
    g[7, 5, 7] = False
    g[7, 5, 8] = True  # the sheet cell sits one step off: a spur-like cell
    opened, _ = gap_spurs.open_gaps(g)
    closed, _ = pinholes.fill_pinholes(opened, 3)
    check(
        "after both steps the sheet has no see-through hole",
        not (pinholes._silhouette_holes(closed, 2, 3)).any() and not (pinholes._silhouette_holes(closed, 1, 3)).any(),
    )


def test_pipeline_step_runs_on_the_coloured_grid():
    g = diagonal_sheet()
    g[7, 5, 7] = False  # a pinhole in the sheet
    g[2, 11, 12] = True  # a spur: sticks out of the slab edge? (no: slab is y 1-10)
    lego = np.where(g, BLACK, 0).astype(np.uint8)
    lego[2:, :, :][g[2:]] = ORANGE  # the sheet is orange, the slab black
    with contextlib.redirect_stdout(io.StringIO()) as out:
        grid, colours = shape_edits.fix_gaps_and_pinholes(
            g, lego, ShapeOptions(open_gaps=True, fill_pinholes_max_cells=3)
        )
    check("pinhole filled with the colour of the sheet around it", grid[7, 5, 7] and colours[7, 5, 7] == ORANGE)
    check("removed / empty cells carry no colour", (colours[~grid] == 0).all())
    check("both steps print a line", "Open gaps:" in out.getvalue() and "Fill pinholes:" in out.getvalue())
    same_grid, same_colours = shape_edits.fix_gaps_and_pinholes(g, lego, ShapeOptions())
    check("off by default: grid and colours unchanged", same_grid is g and same_colours is lego)


# ---- despeckle ------------------------------------------------------------------------------------------
def wall(colour=ORANGE):
    """A flat wall: 6 layers of one 1 x 10 row each, all cells surface."""
    g = np.zeros((6, 3, 10), bool)
    g[:, 1, :] = True
    lego = np.where(g, colour, 0).astype(np.uint8)
    return g, lego


def test_lone_cell_takes_neighbour_colour():
    g, lego = wall()
    lego[3, 1, 4] = DARK_ORANGE
    out, n = cc.despeckle(lego, g, np.zeros(g.shape, bool), 2)
    check("lone cell recoloured", out[3, 1, 4] == ORANGE and n == 1, (out[3, 1, 4], n))


def test_patch_size_limit_and_shared_colour():
    g, lego = wall()
    lego[2, 1, 3:6] = BLACK  # 3 cells: bigger than the limit 2
    lego[4, 1, 0] = DARK_ORANGE  # edge cell with one neighbour only
    out, _ = cc.despeckle(lego, g, np.zeros(g.shape, bool), 2)
    check("patch of 3 stays with limit 2", (out[2, 1, 3:6] == BLACK).all())
    check("cell with a single neighbour stays", out[4, 1, 0] == DARK_ORANGE)
    out, _ = cc.despeckle(lego, g, np.zeros(g.shape, bool), 3)
    check("patch of 3 goes with limit 3", (out[2, 1, 3:6] == ORANGE).all())


def test_locked_cells_stay():
    g, lego = wall()
    lego[3, 1, 4] = WHITE
    locked = np.zeros(g.shape, bool)
    locked[3, 1, 4] = True
    out, n = cc.despeckle(lego, g, locked, 2)
    check("locked lone cell (a painted eye) stays", out[3, 1, 4] == WHITE and n == 0)


def test_only_surface_cells_change():
    g, lego = wall()
    lego[3, 1, 4] = DARK_ORANGE
    surface = g.copy()
    surface[3, 1, 4] = False
    out, n = cc.despeckle(lego, surface, np.zeros(g.shape, bool), 2)
    check("hidden cells are not touched", out[3, 1, 4] == DARK_ORANGE and n == 0)


# ---- neutral enclaves -----------------------------------------------------------------------------------
def test_grey_highlight_in_orange_fur():
    g, lego = wall()
    lego[2:5, 1, 3:7] = LBG  # 12 grey cells inside orange
    lego[0, 1, :] = BLO
    out, n = cc.neutral_enclaves(lego, g, np.zeros(g.shape, bool), palette_lab())
    check("grey region inside colour takes the colour", n == 12 and not (out == LBG).any(), n)


def test_grey_next_to_black_stays():
    g, lego = wall(BLACK)
    lego[:, 1, 6:] = ORANGE
    lego[2:4, 1, 4:6] = LBG  # a grey guard against a black cape and an orange hand
    out, n = cc.neutral_enclaves(lego, g, np.zeros(g.shape, bool), palette_lab())
    check("grey with a mostly black border stays", n == 0 and (out[2:4, 1, 4:6] == LBG).all(), n)


def test_black_is_never_an_enclave():
    g, lego = wall()
    lego[3, 1, 4:6] = BLACK  # pupils: black inside orange
    out, n = cc.neutral_enclaves(lego, g, np.zeros(g.shape, bool), palette_lab())
    check("black inside colour stays (details such as pupils)", n == 0 and (out[3, 1, 4:6] == BLACK).all())


def test_locked_enclave_stays():
    g, lego = wall()
    lego[2:4, 1, 3:5] = LBG
    locked = np.zeros(g.shape, bool)
    locked[2, 1, 3] = True
    out, n = cc.neutral_enclaves(lego, g, locked, palette_lab())
    check("enclave with a locked cell (accent box) stays", n == 0)


# ---- pipeline glue --------------------------------------------------------------------------------------
def test_pipeline_cleanup_keeps_trim_and_paints():
    g, lego = wall()
    lego[0, 1, :] = RED  # a ring
    lego[3, 1, 2] = RED  # trim spread over layers (a tilted cuff): a lone red cell
    lego[4, 1, 7] = DARK_ORANGE  # lone cell
    lego[2, 1, 7] = WHITE  # painted eye (locked by its paint box)
    lego[5, 1, 3:6] = LBG  # grey inside orange
    colour = ColourOptions(
        paints=[Paint((slice(2, 3), slice(1, 2), slice(7, 8)), WHITE, False)],
        despeckle_max_cells=2,
        neutral_enclaves=True,
    )
    with contextlib.redirect_stdout(io.StringIO()) as out:
        res = surface_cleanup.apply_automatic_colour_cleanup(lego, g, [RED], colour)
    check("every trim-colour cell kept (ring and lone trim)", res[3, 1, 2] == RED and (res[0, 1, :] == RED).all())
    check("lone cell gone", res[4, 1, 7] == ORANGE)
    check("grey enclave gone", not (res == LBG).any())
    check("painted cell kept", res[2, 1, 7] == WHITE)
    check("one console line per rule", out.getvalue().count("Colour clean-up:") == 2, out.getvalue())
    with contextlib.redirect_stdout(io.StringIO()):
        same = surface_cleanup.apply_automatic_colour_cleanup(lego, g, [RED], ColourOptions())
    check("off by default: nothing changes", (same == lego).all())


def test_cli_rejects_negative():
    import argparse

    from image_to_bricks import cli

    try:
        cli._non_negative_int("-1")
        check("negative rejected", False)
    except argparse.ArgumentTypeError:
        check("negative rejected", True)
    check("zero accepted", cli._non_negative_int("0") == 0)


if __name__ == "__main__":
    for fn in [v for k, v in list(globals().items()) if k.startswith("test_")]:
        fn()
    sys.exit(1 if failed else 0)
