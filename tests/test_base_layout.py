"""--base-layout (base_layout.py): a hand-built base is loaded, checked (seams, corners), placed where the figure
tips least and appears exactly in the final model, also after tie repair; special pieces count as owned."""

import contextlib
import io
import os
import sys
import tempfile

import support

support.stub_mesh_libraries()
import numpy as np

from image_to_bricks.base import base_layout
from image_to_bricks.consts import COLOR_NAMES
from image_to_bricks.mesh import voxelize
from image_to_bricks.options import BaseOptions, BuildOptions, RepairOptions
from image_to_bricks.packing import placement
from image_to_bricks.stock import Stock, make_key

GREEN, BLACK, ORANGE, YELLOW = 3, 6, 9, 5
NAME_TO_ID = {name.lower(): colour for colour, name in COLOR_NAMES.items()}
failed = False

# 8 x 6 base: top layer two Green 4x6 plates (seam at x 3|4); bottom layer a Black 4x6 frame part on the left,
# a Black 2x6 on the right edge and a Black 4x2 across the seam (x 2-5, y 2-3); window at x 6-7 is not needed.
TIED_LAYOUT = """layer,x,y,width,depth,color,shape
1,0,0,2,6,Black,
1,2,2,4,2,Black,
1,6,0,2,6,Black,technic plate 2x6 test
1,2,0,4,1,Black,
1,2,5,4,1,Black,
2,0,0,4,6,Green,
2,4,0,4,6,Green,
"""


def check(name, ok, info=None):
    global failed
    print(("ok   " if ok else "FAIL ") + name + ("" if ok or info is None else f"  {info}"))
    failed |= not ok


def write_layout(text: str) -> str:
    path = os.path.join(tempfile.mkdtemp(), "layout.csv")
    open(path, "w").write(text)
    return path


def colour_id(name: str) -> int:
    return NAME_TO_ID[name.strip().lower()]


def test_load_and_validate():
    pieces = base_layout.load_layout(write_layout(TIED_LAYOUT), colour_id)
    check(
        "7 plates, 2 layers, 8 x 6",
        len(pieces) == 7 and base_layout.layer_count(pieces) == 2 and base_layout.layout_size(pieces) == (8, 6),
    )
    check("shape note kept", pieces[2].shape == "technic plate 2x6 test")
    overlapping = TIED_LAYOUT + "1,0,0,2,2,Black,\n"
    try:
        base_layout.load_layout(write_layout(overlapping), colour_id)
        check("overlap rejected", False)
    except ValueError as error:
        check("overlap rejected with layer and cell", "layer 1 at x 0, y 0" in str(error), str(error))
    try:
        base_layout.load_layout(write_layout("layer,x,y,width,depth,color\n1,a,0,2,2,Black\n"), colour_id)
        check("bad number rejected", False)
    except ValueError as error:
        check("bad number rejected with the row", "row 2" in str(error), str(error))


def test_seam_check():
    pieces = base_layout.load_layout(write_layout(TIED_LAYOUT), colour_id)
    check("tied layout: no problems", base_layout.untied_seams(pieces) == [], base_layout.untied_seams(pieces))
    untied = (
        TIED_LAYOUT.replace("1,2,2,4,2,Black,\n", "")
        .replace("1,2,0,4,1,Black,\n", "")
        .replace("1,2,5,4,1,Black,\n", "")
    )  # nothing crosses the seam x 3|4 any more
    problems = base_layout.untied_seams(base_layout.load_layout(write_layout(untied), colour_id))
    check("missing tie reported", any("layer 2: seam" in p for p in problems), problems)
    no_corner = TIED_LAYOUT.replace("1,0,0,2,6,Black,\n", "1,0,1,2,5,Black,\n")
    problems = base_layout.untied_seams(base_layout.load_layout(write_layout(no_corner), colour_id))
    check("missing corner plate reported", any("corner x 0, y 0" in p for p in problems), problems)


def figure_grid():
    """Model grid after add_base: 2 base layers + a 2 x 2 tower (x 7-8, y 5-6) on a 16 x 12 grid."""
    grid = np.zeros((10, 12, 16), bool)
    grid[2:, 5:7, 7:9] = True
    colours = np.where(grid, ORANGE, 0).astype(np.uint8)
    return grid, colours


def test_placement_keeps_feet_on_top_plates_and_is_centred():
    pieces = base_layout.load_layout(write_layout(TIED_LAYOUT), colour_id)
    grid, _ = figure_grid()
    chosen = base_layout.choose_placement(grid, 2, pieces, lambda trial: 10.0)
    check("equal tip angles: centred under the tower", (chosen.grid_x, chosen.grid_y) == (4, 3), chosen)
    favour_left = base_layout.choose_placement(grid, 2, pieces, lambda trial: -float(np.nonzero(trial[0])[1].min()))
    check("larger tip angle wins over centring", favour_left.grid_x == 1, favour_left)
    small = np.zeros((10, 5, 5), bool)
    small[2:, 2, 2] = True
    check("does not fit -> None", base_layout.choose_placement(small, 2, pieces, lambda trial: 1.0) is None)


def build_options(base: BaseOptions) -> BuildOptions:
    return BuildOptions(base=base, repair=RepairOptions(tie_repair=True, cleanup_small=True))


def base_options(layout_path: str):
    """The base options with the layout."""
    return BaseOptions(
        plates=2,
        margin_studs=0,
        shape="rect",
        corner_radius_studs=3,
        colour_id=GREEN,
        footprint_layers=3,
        layout=base_layout.load_layout(layout_path, colour_id),
        layout_path=layout_path,
    )


def test_pipeline_uses_the_layout_exactly():
    path = write_layout(TIED_LAYOUT)
    stock = Stock(
        have={make_key(1, 4, 6, GREEN): 2, make_key(1, 2, 6, BLACK): 1},
        specials={(1, 6, 2, BLACK, "technic plate 2x6 test"): 1},
    )
    base = base_options(path)
    grid, colours = figure_grid()
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        model = placement.optimize_brick_placement(
            placement.PackingCells(grid, colours, visible=voxelize.surface_mask(grid), base_layers=2),
            options=build_options(base),
            stock=stock,
        )
    base = sorted((b.z, b.x, b.y, b.w, b.d, b.color) for b in model.bricks if b.z < 2)
    corner_x, corner_y = 4, 3
    expected = sorted(
        (p.layer - 1, corner_x + p.x, corner_y + p.y, p.width, p.depth, p.colour)
        for p in base_layout.load_layout(path, colour_id)
    )
    check("base pieces == layout (after tie repair and clean-up)", base == expected, (base, expected))
    covered = np.zeros(model.grid[:2].shape, bool)
    for b in model.bricks:
        if b.z < 2:
            covered[b.z, b.y : b.y + b.d, b.x : b.x + b.w] = True
    check("grid base layers == layout cells (windows empty)", (model.grid[:2] == covered).all())
    check("console line", "Base layout:" in log.getvalue() and "every seam tied" in log.getvalue(), log.getvalue())
    check("technic special counted as owned", stock.have[make_key(1, 6, 2, BLACK)] == 2)
    check("ledger == final pieces", +stock.recount(model.bricks).used == +stock.used)


def test_wrong_layer_count_stops():
    base = base_options(write_layout(TIED_LAYOUT))
    grid, colours = figure_grid()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            placement.optimize_brick_placement(
                placement.PackingCells(grid[1:], colours[1:], visible=voxelize.surface_mask(grid[1:]), base_layers=1),
                options=build_options(base),
            )
        check("--base 1 with a 2-layer layout stops", False)
    except SystemExit as error:
        check("--base 1 with a 2-layer layout stops with a hint", "use --base 2" in str(error), str(error))


def test_cli_option():
    source = open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "image_to_bricks", "cli.py"), encoding="utf-8"
    ).read()
    check("--base-layout in the CLI", '"--base-layout"' in source and "base_layout.load_layout" in source)
    check("Sand Green in the palette", "sand green" in NAME_TO_ID)


if __name__ == "__main__":
    for test in [value for key, value in list(globals().items()) if key.startswith("test_")]:
        test()
    sys.exit(1 if failed else 0)
