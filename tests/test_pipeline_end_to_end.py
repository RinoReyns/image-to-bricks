"""End to end: model_to_bricks.py on a synthetic two-colour figure GLB (glTF Y-up, like the image-to-3D models),
run as a real CLI process. Checks every output file, the build itself (exact cover, one assembly, head on top,
colours), inventory / stock report / buy list against the pieces, determinism and the size option.
Needs the real trimesh. Run: python tests/test_pipeline_end_to_end.py   (about 1 minute)"""

import csv
import functools
import os
import re

import numpy as np
import support
import trimesh

from image_to_bricks.analysis import stability
from image_to_bricks.consts import COLOR_NAMES
from image_to_bricks.instructions.outputs_handler import load_model

COLOUR_ID_BY_NAME = {name: colour_id for colour_id, name in COLOR_NAMES.items()}
BODY_RGB = (30, 60, 200)
HEAD_RGB = (200, 30, 30)
MODEL_NAME = "figure"
TARGET_STUDS = 8
BASE_PLATES = 2
BASE_COLOUR = "Green"
ALLOWED_COLOURS = "Blue,Red"
STOCK_ROWS = [
    ["Plate", 2, 1, 1, "Blue", 20],
    ["Plate", 4, 2, 1, "Blue", 5],
    ["Brick", 2, 2, 3, "Blue", 6],
    ["Plate", 2, 2, 1, "Red", 4],
    ["Plate", 8, 2, 1, "Green", 2],
]
OUTPUT_FILE_SUFFIXES = (
    "instructions.pdf",
    "inventory.csv",
    "stability.csv",
    "stock_report.csv",
    "buy_list.csv",
    "owned_layers.csv",
    "model.pkl",
    "view_front.png",
    "view_back.png",
    "view_left.png",
    "view_right.png",
    "dbg1_raw_rgb.png",
    "dbg4_final_ids.png",
)


def coloured_box(extents_mm: tuple, centre_mm: tuple, rgb: tuple) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=extents_mm)
    box.apply_translation(centre_mm)
    box.visual.vertex_colors = rgb + (255,)
    return box


def write_figure_glb(folder: str) -> str:
    """Blue body with an arm and a red head, stored Y-up (glTF convention) with the face toward +Z."""
    body = coloured_box((40.0, 56.0, 32.0), (0.0, 28.0, 0.0), BODY_RGB)
    arm = coloured_box((10.0, 34.0, 12.0), (25.0, 34.0, 0.0), BODY_RGB)
    head = coloured_box((44.0, 28.0, 36.0), (0.0, 70.0, 0.0), HEAD_RGB)
    glb_path = os.path.join(folder, f"{MODEL_NAME}.glb")
    trimesh.util.concatenate([body, arm, head]).export(glb_path)
    return glb_path


def write_stock_csv(folder: str) -> str:
    path = os.path.join(folder, "stock.csv")
    with open(path, "w", newline="", encoding="utf-8") as stock_file:
        writer = csv.writer(stock_file)
        writer.writerow(["piece_type", "width_studs", "depth_studs", "height_plates", "color_name", "quantity"])
        writer.writerows(STOCK_ROWS)
    return path


def cli_arguments(glb_path: str, stock_path: str, outdir: str, target_studs: int = TARGET_STUDS) -> list:
    """A typical command: the general options of the documented builds, nothing model-specific."""
    return [
        glb_path,
        "--pdf",
        "--rotate-x",
        "90",
        "--target-studs",
        str(target_studs),
        "--voxel-supersample",
        "3",
        "--hollow",
        "--shell",
        "2",
        "--strict-hidden",
        "--colors",
        ALLOWED_COLOURS,
        "--base",
        str(BASE_PLATES),
        "--base-shape",
        "rounded",
        "--base-color",
        BASE_COLOUR,
        "--front",
        "down",
        "--tie-recolor",
        "10",
        "--cleanup-small",
        "--stock",
        stock_path,
        "--use-l-pieces",
        "--open-gaps",
        "--fill-pinholes",
        "3",
        "--despeckle",
        "2",
        "--anchor-floating",
        "2",
        "--merge-plates",
        "--prefer-bricks",
        "--save-model",
        "--outdir",
        outdir,
    ]


@functools.cache
def build_once(run_label: str, target_studs: int = TARGET_STUDS) -> str:
    """Run the CLI once per label (cached: several tests check the same run); returns the output folder."""
    folder = support.make_temp_dir(f"e2e_{run_label}")
    outdir = os.path.join(folder, "out")
    process = support.run_cli(cli_arguments(write_figure_glb(folder), write_stock_csv(folder), outdir, target_studs))
    support.assert_cli_succeeded(process)
    return outdir


def output_path(outdir: str, suffix: str) -> str:
    return os.path.join(outdir, f"{MODEL_NAME}_{suffix}")


def load_built_model(outdir: str):
    return load_model(output_path(outdir, "model.pkl"))


def read_inventory(outdir: str) -> tuple:
    """(rows as dicts, TOTAL PIECES number) from the inventory CSV."""
    with open(output_path(outdir, "inventory.csv"), newline="") as inventory_file:
        rows = list(csv.DictReader(inventory_file))
    piece_rows = [row for row in rows if row["piece_type"]]
    total = int(next(row["quantity"] for row in rows if row["section"] == "TOTAL PIECES"))
    return piece_rows, total


def read_csv_total(path: str, column: str) -> int:
    with open(path, newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    return int(next(row[column] for row in rows if row["piece_type"] == "TOTAL PIECES"))


def test_every_output_file_is_written():
    outdir = build_once("main")
    for suffix in OUTPUT_FILE_SUFFIXES:
        assert os.path.getsize(output_path(outdir, suffix)) > 0, suffix
    layer_maps = [name for name in os.listdir(os.path.join(outdir, "layers")) if name.endswith(".png")]
    assert layer_maps, "no layer maps"
    with open(os.path.join(outdir, "run_log.txt"), encoding="utf-8") as log_file:
        assert log_file.readline().startswith("python "), "the log starts with the command"


def test_pdf_has_a_page_for_every_layer_where_pieces_start():
    outdir = build_once("main")
    with open(output_path(outdir, "instructions.pdf"), "rb") as pdf_file:
        pdf_bytes = pdf_file.read()
    assert pdf_bytes.startswith(b"%PDF")
    page_count = pdf_bytes.count(b"/Type /Page") - pdf_bytes.count(b"/Type /Pages")
    layers_with_new_pieces = {piece.z for piece in load_built_model(outdir).bricks}
    assert page_count > len(layers_with_new_pieces), (page_count, len(layers_with_new_pieces))


def test_pieces_cover_the_grid_exactly():
    support.assert_pieces_cover_grid_exactly(load_built_model(build_once("main")))


def test_build_is_one_connected_assembly_without_weak_joints():
    model = load_built_model(build_once("main"))
    with support.quiet_stdout():
        report = stability.report(model, verbose=False)
    assert report["components"] == 1, report["components"]
    assert report["disconnected_pieces"] == 0


def test_base_is_the_lowest_layers_in_the_base_colour():
    model = load_built_model(build_once("main"))
    assert model.base_layers == BASE_PLATES
    base_pieces = [piece for piece in model.bricks if piece.z < BASE_PLATES]
    assert base_pieces and all(piece.h_plates == 1 for piece in base_pieces)


def test_saved_model_records_the_automatic_base_margin() -> None:
    outdir = build_once("main")
    model = load_built_model(outdir)
    with open(os.path.join(outdir, "run_log.txt"), encoding="utf-8") as log_file:
        chosen = re.search(r"Base margin \(auto\): (\d+) studs", log_file.read())
    assert chosen, "no 'Base margin (auto)' line in run_log.txt"
    assert model.base_margin == int(chosen.group(1)) > 0


def test_head_is_on_top_and_body_below_after_rotation():
    model = load_built_model(build_once("main"))
    figure_pieces = [piece for piece in model.bricks if piece.z >= model.base_layers]
    top_layer = max(piece.z + piece.h_plates for piece in figure_pieces) - 1
    lowest_figure_layer = model.base_layers
    colours_on_top = {
        COLOR_NAMES[piece.color] for piece in figure_pieces if piece.z <= top_layer < piece.z + piece.h_plates
    }
    colours_at_bottom = {COLOR_NAMES[piece.color] for piece in figure_pieces if piece.z == lowest_figure_layer}
    assert colours_on_top == {"Red"}, colours_on_top
    assert colours_at_bottom == {"Blue"}, colours_at_bottom


def test_figure_uses_only_the_allowed_colours():
    model = load_built_model(build_once("main"))
    allowed = {COLOUR_ID_BY_NAME[name] for name in ALLOWED_COLOURS.split(",")}
    figure_colours = {piece.color for piece in model.bricks if piece.z >= model.base_layers}
    assert figure_colours <= allowed, {COLOR_NAMES[colour_id] for colour_id in figure_colours}


def test_figure_width_matches_the_target_studs():
    model = load_built_model(build_once("main"))
    figure_cells = model.grid[model.base_layers :]
    filled_y, filled_x = np.nonzero(figure_cells.any(axis=0))
    widest_side_studs = max(filled_x.max() - filled_x.min() + 1, filled_y.max() - filled_y.min() + 1)
    assert abs(int(widest_side_studs) - TARGET_STUDS) <= 1, widest_side_studs


def test_inventory_counts_every_piece_once():
    outdir = build_once("main")
    model = load_built_model(outdir)
    inventory_rows, inventory_total = read_inventory(outdir)
    assert inventory_total == len(model.bricks) == sum(int(row["quantity"]) for row in inventory_rows)
    pieces_by_colour = {}
    for piece in model.bricks:
        pieces_by_colour[COLOR_NAMES[piece.color]] = pieces_by_colour.get(COLOR_NAMES[piece.color], 0) + 1
    inventory_by_colour = {}
    for row in inventory_rows:
        inventory_by_colour[row["color_name"]] = inventory_by_colour.get(row["color_name"], 0) + int(row["quantity"])
    assert inventory_by_colour == pieces_by_colour


def test_stock_report_and_buy_list_agree_with_the_pieces():
    outdir = build_once("main")
    model = load_built_model(outdir)
    used_total = read_csv_total(output_path(outdir, "stock_report.csv"), "used")
    still_need_total = read_csv_total(output_path(outdir, "stock_report.csv"), "still_need")
    buy_total = read_csv_total(output_path(outdir, "buy_list.csv"), "quantity")
    assert used_total == len(model.bricks)
    assert buy_total == still_need_total
    owned_pieces = sum(row[5] for row in STOCK_ROWS)
    assert still_need_total >= len(model.bricks) - owned_pieces


def test_same_command_gives_the_same_build():
    first, second = build_once("main"), build_once("repeat")
    assert support.fingerprint_model(load_built_model(first)) == support.fingerprint_model(load_built_model(second))
    assert support.read_text_lines(output_path(first, "inventory.csv")) == support.read_text_lines(
        output_path(second, "inventory.csv")
    )


def test_larger_target_gives_a_larger_build():
    small, large = load_built_model(build_once("main")), load_built_model(build_once("large", TARGET_STUDS + 4))
    assert large.grid.sum() > small.grid.sum() and len(large.bricks) > len(small.bricks)


def test_run_without_stock_skips_the_stock_files():
    folder = support.make_temp_dir("e2e_nostock")
    outdir = os.path.join(folder, "out")
    arguments = [
        write_figure_glb(folder),
        "--rotate-x",
        "90",
        "--target-studs",
        "6",
        "--colors",
        ALLOWED_COLOURS,
        "--outdir",
        outdir,
    ]
    support.assert_cli_succeeded(support.run_cli(arguments))
    assert os.path.exists(output_path(outdir, "inventory.csv"))
    assert not os.path.exists(output_path(outdir, "stock_report.csv"))
    assert not os.path.exists(output_path(outdir, "instructions.pdf")), "no PDF without --pdf"


def test_debug_option_writes_the_placement_check():
    folder = support.make_temp_dir("e2e_debug")
    outdir = os.path.join(folder, "out")
    arguments = [
        write_figure_glb(folder),
        "--rotate-x",
        "90",
        "--target-studs",
        "6",
        "--colors",
        ALLOWED_COLOURS,
        "--no-views",
        "--debug",
        "--outdir",
        outdir,
    ]
    process = support.run_cli(arguments)
    support.assert_cli_succeeded(process)
    for suffix in ("voxelized.png", "placed_bricks.png", "missing_from_bricks.png", "extra_from_bricks.png"):
        assert os.path.getsize(output_path(outdir, suffix)) > 0, suffix
    assert "missing cells: 0" in process.stdout and "overlapping cells: 0" in process.stdout
    assert "WARNING: Brick placement differs" not in process.stdout


def test_without_debug_no_debug_images_are_written():
    outdir = build_once("main")
    assert not os.path.exists(output_path(outdir, "voxelized.png"))
    assert not os.path.exists(output_path(outdir, "placed_bricks.png"))


if __name__ == "__main__":
    support.run_tests(globals())
