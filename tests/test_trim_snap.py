"""--trim-snap (trim_snap.py): ring-less trim snaps to the nearest trim ring layer; trim off the rings is recoloured."""

import contextlib
import io
import sys

import support

support.stub_mesh_libraries()
import numpy as np

from image_to_bricks.colours import surface_cleanup, trim_snap
from image_to_bricks.mesh import voxelize
from image_to_bricks.options import ColourOptions

BLACK, DARK_RED, ORANGE = 6, 17, 9
failed = False


def check(name, ok, info=None):
    global failed
    print(("ok   " if ok else "FAIL ") + name + ("" if ok or info is None else f"  {info}"))
    failed |= not ok


def legs():
    """Two 4 x 4 boots (x 1-4 and x 7-10) up to z 9, merged into one body above, and a 2 x 2 tail at y 8-9.
    Left cuff: a full red ring at z 6. Right cuff: 3 red cells higher up (z 12, over the right boot)."""
    g = np.zeros((16, 11, 12), bool)
    g[:10, 1:5, 1:5] = True
    g[:10, 1:5, 7:11] = True
    g[10:, 1:5, 1:11] = True
    g[3:8, 8:10, 8:10] = True  # tail beside the right boot
    lego = np.where(g, BLACK, 0).astype(np.uint8)
    lego[3:8, 8:10, 8:10] = ORANGE
    ring = trim_snap.part_outline(g[6, :, :] & (np.arange(12) < 6)[None, :])
    lego[6][ring] = DARK_RED
    lego[12, 1, 8:11] = DARK_RED  # the right cuff, scattered and higher
    lego[7, 1, 2] = DARK_RED  # a leftover just above the left ring
    return g, lego


def test_outline_is_the_part_border():
    part = np.zeros((5, 5), bool)
    part[1:4, 1:4] = True
    check("outline of a 3 x 3 part = 8 border cells", int(trim_snap.part_outline(part).sum()) == 8)


def test_ringless_trim_snaps_to_the_ring_layer_of_its_own_boot():
    g, lego = legs()
    trim = surface_cleanup.trim_ring_mask(lego, g, [DARK_RED], 0.15, 4)
    rings, _ = trim[DARK_RED]
    check("before: one ring (left boot, z 6)", sorted(set(np.nonzero(rings)[0])) == [6] and not rings[6, :, 7:].any())
    snapped, added = trim_snap.snap_all(lego, g, voxelize.surface_mask(g), trim, trim_snap.SnapOptions(12))
    rings2, strays = snapped[DARK_RED]
    right = trim_snap.part_outline(g[6] & (np.arange(12) >= 7)[None, :] & (np.arange(11) < 5)[:, None])
    check("one ring added", added == 1, added)
    check("right boot gets its ring in the same layer", (rings2[6][right]).all())
    check("rings stay in one layer", sorted(set(np.nonzero(rings2)[0])) == [6])
    check("the tail next to the boot gets no ring", not rings2[6, 8:10, 8:10].any())
    check("every trim cell off the rings is a stray", strays[12, 1, 8:11].all() and strays[7, 1, 2])


def test_ring_covers_the_visible_top_of_the_part():
    g = np.zeros((6, 7, 7), bool)
    g[:3, 1:6, 1:6] = True  # a 5 x 5 boot, 3 layers
    g[3:, 2:5, 2:5] = True  # a narrower leg on top: the boot's top rim is visible
    surface = voxelize.surface_mask(g)
    rings = np.zeros(g.shape, bool)
    rings[2] = trim_snap.part_outline(g[2])
    band = trim_snap.visible_bands(g, surface, rings)
    check(
        "band = outline + visible top cells of that part", band[2][surface[2]].all() and (band[2] == surface[2]).all()
    )
    check("band stays in the ring layer", not band[[0, 1, 3, 4, 5]].any())


def test_trim_beside_a_part_does_not_ring_it():
    g, lego = legs()
    g[11:14, 8:10, 1:3] = True  # an arm sticking out over empty space
    lego[11:14, 8:10, 1:3] = ORANGE
    lego[12, 8, 1:3] = DARK_RED  # red on the arm: nothing under it at the ring layer
    trim = surface_cleanup.trim_ring_mask(lego, g, [DARK_RED], 0.15, 4)
    _, added = trim_snap.snap_rings(lego, g, DARK_RED, trim[DARK_RED][0], trim_snap.SnapOptions(12))
    g0, lego0 = legs()
    trim0 = surface_cleanup.trim_ring_mask(lego0, g0, [DARK_RED], 0.15, 4)
    _, added0 = trim_snap.snap_rings(lego0, g0, DARK_RED, trim0[DARK_RED][0], trim_snap.SnapOptions(12))
    check("trim with no part under it adds no ring (same as without the arm)", added == added0 == 1, (added, added0))
    under = trim_snap.part_under(g[6], np.array([[8, 1], [8, 2]]))
    check("part_under: nothing below -> empty", not under.any())
    under = trim_snap.part_under(g[6], np.array([[2, 8], [9, 9]]))
    check("part_under: the part under most cells", under[2, 8] and under[1, 7])


def test_clear_strays_after_smoothing():
    g, lego = legs()
    trim = surface_cleanup.trim_ring_mask(lego, g, [DARK_RED], 0.15, 4)
    rings = trim[DARK_RED][0]
    spread = lego.copy()
    spread[rings] = DARK_RED
    spread[5, 1, 2] = DARK_RED  # smoothing put red just below the ring
    out, n = trim_snap.clear_strays(spread, voxelize.surface_mask(g), trim)
    check("red spread off the ring is cleared", out[5, 1, 2] != DARK_RED and n >= 1, n)
    check("the ring itself stays", (out[rings & voxelize.surface_mask(g)] == DARK_RED).all())


def test_too_far_does_not_snap():
    g, lego = legs()
    trim = surface_cleanup.trim_ring_mask(lego, g, [DARK_RED], 0.15, 4)
    _, added = trim_snap.snap_all(lego, g, voxelize.surface_mask(g), trim, trim_snap.SnapOptions(3))
    check("trim 6 layers away does not snap with N = 3", added == 0)


def test_no_ring_no_snap():
    g, lego = legs()
    lego[6][lego[6] == DARK_RED] = BLACK  # no ring anywhere
    rings, added = trim_snap.snap_rings(lego, g, DARK_RED, np.zeros(g.shape, bool), trim_snap.SnapOptions(12))
    check("without any ring layer nothing snaps", added == 0 and not rings.any())


def test_pipeline_step():
    g, lego = legs()
    trim = surface_cleanup.trim_ring_mask(lego, g, [DARK_RED], 0.15, 4)
    with contextlib.redirect_stdout(io.StringIO()) as out:
        res, trim2 = surface_cleanup.snap_trim_to_rings(
            lego, g, voxelize.surface_mask(g), trim, ColourOptions(trim_snap_layers=12)
        )
    for tid, (m, _) in trim2.items():
        res[m] = tid  # as assign_voxel_colors does with the rings
    surf = voxelize.surface_mask(g)
    red_layers = sorted(set(np.nonzero((res == DARK_RED) & surf)[0]))
    check("after the step red is only on the ring layer", red_layers == [6], red_layers)
    check("both boots red at z 6", (res[6, 1, 1:5] == DARK_RED).all() and (res[6, 1, 7:11] == DARK_RED).all())
    check("strays took a non-trim colour", res[12, 1, 9] == BLACK and res[7, 1, 2] == BLACK)
    check("console line printed", "Trim snap: 1 ring(s) added" in out.getvalue(), out.getvalue())
    check("returned strays are empty", not any(s.any() for _, s in trim2.values()))


def legs_with_rising_tail():
    """legs() plus a 3 x 3 tail (9 cells, a boot is 16) standing behind the right boot (one stud of air between)
    from z 3 up, joined to the body above z 10, with red stripe cells on the tail at z 9 straight above its cross-
    section at the ring layer (a real figure's tail got a ring on its PDF layer 18 this way)."""
    g, lego = legs()
    g[3:8, 8:10, 8:10] = False
    lego[3:8, 8:10, 8:10] = 0
    g[3:13, 6:9, 8:11] = True
    g[10:13, 5, 8:11] = True  # tail joins the body
    lego[g & (lego == 0)] = ORANGE
    lego[9, 8, 8:11] = DARK_RED
    lego[12, 1, 8:11] = BLACK  # no right-cuff red: only the tail stripe can snap
    return g, lego


def test_tail_with_red_stripes_gets_no_ring():
    g, lego = legs_with_rising_tail()
    trim = surface_cleanup.trim_ring_mask(lego, g, [DARK_RED], 0.15, 4)
    rings, added = trim_snap.snap_rings(lego, g, DARK_RED, trim[DARK_RED][0], trim_snap.SnapOptions(12))
    check(
        "red stripe on a smaller tail adds no ring (9 vs 16 cells)", added == 0 and not rings[6, 6:9, 8:11].any(), added
    )
    _, added_off = trim_snap.snap_rings(lego, g, DARK_RED, trim[DARK_RED][0], trim_snap.SnapOptions(12, 0.0))
    check("with the size check off (0) the tail would be ringed (the step-63 bug)", added_off == 1, added_off)


def test_second_boot_still_snaps_next_to_a_tail():
    g, lego = legs_with_rising_tail()
    lego[12, 1, 8:11] = DARK_RED  # the right cuff is back
    trim = surface_cleanup.trim_ring_mask(lego, g, [DARK_RED], 0.15, 4)
    rings, added = trim_snap.snap_rings(lego, g, DARK_RED, trim[DARK_RED][0], trim_snap.SnapOptions(12))
    check("only the right boot gets a ring", added == 1 and rings[6, 1, 8] and not rings[6, 6:9, 8:11].any(), added)


def test_trim_on_a_separate_body_does_not_ring_the_part_below():
    g = np.zeros((14, 6, 12), bool)
    g[:8, 1:5, 1:5] = True  # left boot (ring at z 6)
    g[:8, 1:5, 7:11] = True  # right boot, nothing above it
    g[10:13, 1:5, 7:11] = True  # a separate block hanging over it (air at z 8-9)
    lego = np.where(g, BLACK, 0).astype(np.uint8)
    lego[6][trim_snap.part_outline(g[6] & (np.arange(12) < 6)[None, :])] = DARK_RED
    lego[11, 1, 8:11] = DARK_RED
    trim = surface_cleanup.trim_ring_mask(lego, g, [DARK_RED], 0.15, 4)
    _, added = trim_snap.snap_rings(lego, g, DARK_RED, trim[DARK_RED][0], trim_snap.SnapOptions(12))
    check("red on a separate body over a boot adds no ring", added == 0, added)
    part = g[6] & (np.arange(12) >= 7)[None, :]
    cells = np.argwhere(lego == DARK_RED)
    cells = cells[cells[:, 0] == 11]
    check("is_same_body: False across air", not trim_snap.is_same_body(g, part, 6, cells))
    g[8:10, 1:5, 7:11] = True
    check("is_same_body: True once joined", trim_snap.is_same_body(g, part, 6, cells))


def test_similar_size_rule():
    check(
        "is_similar_size: 16 vs 19 ok, 10 vs 19 not, 0 = off",
        trim_snap.is_similar_size(16, [19], 0.65)
        and not trim_snap.is_similar_size(10, [19], 0.65)
        and trim_snap.is_similar_size(10, [19], 0.0)
        and not trim_snap.is_similar_size(40, [19], 0.65),
    )


def test_cli_option():
    from image_to_bricks import cli

    check(
        "--trim-snap-similarity accepts 0..1, rejects 1.5",
        cli._ratio_from_zero_to_one("0.5") == 0.5 and _raises(cli._ratio_from_zero_to_one, "1.5"),
    )


def _raises(function, argument):
    try:
        function(argument)
    except Exception:
        return True
    return False


if __name__ == "__main__":
    for fn in [v for k, v in list(globals().items()) if k.startswith("test_")]:
        fn()
    sys.exit(1 if failed else 0)
