"""Layer maps show the layer BELOW (fast_render.draw_layer_below, --no-layer-below): its pieces are hatched in
their own colour where the new layer leaves them uncovered; covered studs, continuing bricks and the model are
unchanged; without the option the map is drawn as before."""

import contextlib
import io
import os
import re
import sys
import tempfile

import support

support.stub_mesh_libraries()
import numpy as np
from support import load_image

from image_to_bricks.instructions import fast_render as fr
from image_to_bricks.instructions import outputs_handler as oh
from image_to_bricks.model import LegoModel, PlacedBrick

BLACK, ORANGE, YELLOW = (20, 20, 20), (230, 120, 20), (255, 240, 120)
COLOUR_BY_ID = {1: BLACK, 2: ORANGE, 3: YELLOW}
CELL_PX = 34
failed = False


def check(name, ok, info=None):
    global failed
    print(("ok   " if ok else "FAIL ") + name + ("" if ok or info is None else f"  {info}"))
    failed |= not ok


def colour_of(colour_id):
    return COLOUR_BY_ID[colour_id]


def share_of_cell_in_colour(img, frame, stud, rgb):
    """Fraction of a stud cell's inner pixels (2 px border skipped) that have exactly this colour."""
    left, top, right, bottom = frame.cell_box(*stud)
    pixels = np.asarray(img)[top + 4 : bottom - 4, left + 4 : right - 4].reshape(-1, 3)
    return float((pixels == np.array(rgb)).all(axis=1).mean())


def two_layer_pieces():
    """Layer 0: orange 2x1 at (0, 0)-(1, 0) + black 1x1 at (3, 0). Layer 1: black 1x1 at (0, 0) (covers half of the
    orange plate) + black 1x1 at (0, 2) (overhang, nothing under it)."""
    below = [PlacedBrick(0, 0, 0, 2, 1, 1, 2), PlacedBrick(3, 0, 0, 1, 1, 1, 1)]
    current = [PlacedBrick(0, 0, 1, 1, 1, 1, 1), PlacedBrick(0, 2, 1, 1, 1, 1, 1)]
    return below, current


def draw(below, current, depth=3, width=4):
    image = fr.layer_map(
        current,
        fr.MapRegion(0, 0, width, depth, cell=CELL_PX),
        "Layer 2",
        colour_of,
        fr.MapMarks(new_z=1, below_bricks=below),
    )
    return image, fr.layer_map_frame(depth, width, CELL_PX, 0, 0)


def test_uncovered_studs_of_the_layer_below_are_hatched():
    below, current = two_layer_pieces()
    image, frame = draw(below, current)
    orange_share = share_of_cell_in_colour(image, frame, (1, 0), ORANGE)
    black_share = share_of_cell_in_colour(image, frame, (3, 0), fr.ghost_colour(BLACK))
    white_share = share_of_cell_in_colour(image, frame, (1, 0), (255, 255, 255))
    check(
        "uncovered stud of the orange plate below is hatched orange (not filled)",
        0.1 < orange_share < 0.5 and white_share > 0.4,
        (orange_share, white_share),
    )
    check("uncovered black 1x1 below is hatched black", 0.1 < black_share < 0.5, black_share)


def cell_pixels(img, frame, stud):
    """Pixels of one stud cell (its whole box)."""
    left, top, right, bottom = frame.cell_box(*stud)
    return np.asarray(img)[top:bottom, left:right]


def test_covered_studs_and_empty_studs_show_no_ghost():
    # Compared with the same map drawn WITHOUT the layer below, so the check does not depend on the font: a 1x1 gets
    # a "1x1" label with Arial (Windows) but none with DejaVu Sans (Linux), where the text does not fit the cell.
    below, current = two_layer_pieces()
    image, frame = draw(below, current)
    without_layer_below = fr.layer_map(
        current, fr.MapRegion(0, 0, 4, 3, cell=CELL_PX), "Layer 2", colour_of, fr.MapMarks(new_z=1)
    )
    check(
        "stud covered by the new piece shows the new piece, no hatching",
        np.array_equal(cell_pixels(image, frame, (0, 0)), cell_pixels(without_layer_below, frame, (0, 0)))
        and share_of_cell_in_colour(image, frame, (0, 0), ORANGE) == 0,
    )
    check(
        "overhanging new piece has nothing hatched under it",
        np.array_equal(cell_pixels(image, frame, (0, 2)), cell_pixels(without_layer_below, frame, (0, 2))),
    )
    check(
        "empty stud with nothing below stays white",
        share_of_cell_in_colour(image, frame, (2, 2), (255, 255, 255)) > 0.95,
    )


def test_continuing_brick_keeps_its_faded_style():
    brick = PlacedBrick(2, 1, 0, 1, 1, 3, 2)  # 3-plate brick started on layer 0
    image, frame = draw([brick], [brick])
    faded = tuple(int(c + (255 - c) * fr.CONTINUING_FADE) for c in ORANGE)
    check("continuing brick drawn faded, not hatched", share_of_cell_in_colour(image, frame, (2, 1), faded) > 0.9)


def test_light_colours_are_darkened_so_they_show():
    check(
        "white / yellow ghost colours darkened",
        fr.ghost_colour((255, 255, 255)) != (255, 255, 255)
        and fr.ghost_colour(YELLOW) != YELLOW
        and fr.ghost_colour(BLACK) == BLACK,
    )


def test_l_piece_below_hatches_only_its_three_studs():
    l_piece = PlacedBrick(0, 0, 0, 2, 2, 1, 2, notch=3)  # notch at (1, 1)
    image, frame = draw([l_piece], [])
    check(
        "L below: its 3 studs hatched",
        all(share_of_cell_in_colour(image, frame, stud, ORANGE) > 0.1 for stud in ((0, 0), (1, 0), (0, 1))),
    )
    check("L below: the notch stud stays white", share_of_cell_in_colour(image, frame, (1, 1), (255, 255, 255)) > 0.95)


def test_without_layer_below_the_map_is_drawn_as_before():
    below, current = two_layer_pieces()
    old_style = fr.layer_map(
        current, fr.MapRegion(0, 0, 4, 3, cell=CELL_PX), "Layer 2", colour_of, fr.MapMarks(new_z=1)
    )
    explicit_none = fr.layer_map(
        current, fr.MapRegion(0, 0, 4, 3, cell=CELL_PX), "Layer 2", colour_of, fr.MapMarks(new_z=1, below_bricks=None)
    )
    with_ghost, frame = draw(below, current)
    check(
        "below_bricks=None == default call (pixel identical)",
        np.array_equal(np.asarray(old_style), np.asarray(explicit_none)),
    )
    check("no hatching without the option", share_of_cell_in_colour(old_style, frame, (1, 0), (255, 255, 255)) > 0.95)
    check(
        "legend row only with the option",
        with_ghost.height == old_style.height + fr.legend_height(4 * CELL_PX, CELL_PX),
    )


def small_model():
    """Base-less 2-layer model: layer 0 = black 2x2 at x 4..5; layer 1 = orange 1x1 at x 1 (far left)."""
    grid = np.zeros((2, 6, 8), bool)
    grid[0, 2:4, 4:6] = True
    grid[1, 2, 1] = True
    bricks = [PlacedBrick(4, 2, 0, 2, 2, 1, 1), PlacedBrick(1, 2, 1, 1, 1, 1, 2)]
    return LegoModel(grid=grid, bricks=bricks)


def test_map_crop_includes_the_layer_below():
    model = small_model()
    check(
        "crop with the layer below spans both layers",
        oh.map_region(model, 1, show_layer_below=True, margin_studs=0) == (1, 6, 2, 4),
        oh.map_region(model, 1, show_layer_below=True, margin_studs=0),
    )
    check(
        "crop without it: own footprint only",
        oh.map_region(model, 1, show_layer_below=False, margin_studs=0) == (1, 2, 2, 3),
    )
    check(
        "first layer: nothing below to add",
        oh.map_region(model, 0, show_layer_below=True, margin_studs=0) == (4, 6, 2, 4),
    )


def test_build_layers_write_ghost_maps_and_leave_the_model_alone():
    model = small_model()
    before = [(b.x, b.y, b.z, b.w, b.d, b.h_plates, b.color) for b in model.bricks]
    with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
        on = oh.generate_build_layers(model, os.path.join(folder, "on"), show_layer_below=True)
        off = oh.generate_build_layers(model, os.path.join(folder, "off"), show_layer_below=False)
        sizes_on = [load_image(path).size for path in on]
        sizes_off = [load_image(path).size for path in off]
    after = [(b.x, b.y, b.z, b.w, b.d, b.h_plates, b.color) for b in model.bricks]
    check("one map per layer either way", len(on) == len(off) == 2)
    check(
        "first layer: same crop, only the legend rows added",
        sizes_on[0][0] == sizes_off[0][0] and sizes_on[0][1] > sizes_off[0][1],
    )
    check(
        "layer 2 map wider with the layer below (crop) and has the legend row",
        sizes_on[1][0] > sizes_off[1][0] and sizes_on[1][1] > sizes_off[1][1],
        (sizes_on, sizes_off),
    )
    check("model pieces unchanged", before == after)


def count_pdf_pages(path):
    """Number of pages, read from the PDF objects (no PDF reader needed)."""
    return len(re.findall(rb"/Type\s*/Page[^s]", open(path, "rb").read()))


def tall_model():
    """2 layers, 30 studs deep, one palette colour per row: a long parts table and a tall map that is sized to fill
    the rest of the page, so any line the page does not count pushes the map over."""
    grid = np.zeros((2, 30, 6), bool)
    grid[:, :, 1:3] = True
    bricks = [PlacedBrick(1, y, z, 2, 1, 1, 1 + y % 22) for z in range(2) for y in range(30)]
    return LegoModel(grid=grid, bricks=bricks)


def test_layer_notes_are_counted_so_the_map_stays_on_its_page():
    model = tall_model()
    is_owned = {id(brick): False for brick in model.bricks}  # adds a "From your stock" line per layer
    with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
        images = oh.generate_build_layers(model, os.path.join(folder, "layers"), show_layer_below=True)
        plain = os.path.join(folder, "plain.pdf")
        noted = os.path.join(folder, "noted.pdf")
        oh.generate_instruction_pdf(model, images, plain, oh.InstructionPdf("t"))
        oh.generate_instruction_pdf(model, images, noted, oh.InstructionPdf("t", is_owned=is_owned))
        pages_plain, pages_noted = count_pdf_pages(plain), count_pdf_pages(noted)
    check(
        "ownership line does not push the layer map to an extra page",
        pages_noted == pages_plain,
        (pages_plain, pages_noted),
    )


def test_cli_option_reaches_the_layer_maps():
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    cli = open(os.path.join(folder, "image_to_bricks", "cli.py"), encoding="utf-8").read()
    outputs = open(
        os.path.join(folder, "image_to_bricks", "instructions", "outputs_handler.py"), encoding="utf-8"
    ).read()
    check(
        "--no-layer-below in the CLI and passed on",
        '"--no-layer-below"' in cli
        and "show_layer_below=not args.no_layer_below" in cli
        and "save_outputs(model, model_name, outdir, stock, options.output)" in cli,
    )
    check(
        "save_outputs passes it to the layer maps",
        "generate_build_layers(model, layers_dir, show_layer_below=show_layer_below, plan=plan)" in outputs,
    )


if __name__ == "__main__":
    for test in [value for key, value in list(globals().items()) if key.startswith("test_")]:
        test()
    sys.exit(1 if failed else 0)
