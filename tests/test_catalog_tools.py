"""my_bricks_catalog tools on synthetic files in temp folders (the real catalogue is never touched):
build_catalog.py (photo counts + hand edits, backup, specials) and check_cost.py (pricing rules, profiles, factor,
fallback guess, folder input, cost CSV). Run: python tests/test_catalog_tools.py"""

import csv
import importlib.util
import os
import shutil
import subprocess
import sys

import support

CATALOG_DIR = os.path.join(support.PROJECT_DIR, "my_bricks_catalog")
BUILD_CATALOG_SCRIPT = os.path.join(CATALOG_DIR, "build_catalog.py")
CHECK_COST_SCRIPT = os.path.join(CATALOG_DIR, "check_cost.py")
ITEM_HEADER = [
    "item_id",
    "source_photo",
    "piece_type",
    "width_studs",
    "depth_studs",
    "height_plates",
    "color_name",
    "bricklink_part",
    "shape",
]
STOCK_HEADER = [
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
INVENTORY_HEADER = [
    "section",
    "piece_type",
    "width_studs",
    "depth_studs",
    "height_plates",
    "color_name",
    "quantity",
    "shape",
]
PRICE_HEADER = ["piece_type", "width_studs", "depth_studs", "color_name", "price_lego_shop", "price_compatible", "note"]


def write_csv(path: str, header: list, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)


def read_csv_rows(path: str) -> list:
    with open(path, newline="", encoding="utf-8-sig") as csv_file:
        return list(csv.DictReader(csv_file))


def load_check_cost_module():
    """check_cost.py imported as a module (its folder is not a package)."""
    specification = importlib.util.spec_from_file_location("check_cost", CHECK_COST_SCRIPT)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def run_build_catalog(item_rows: list, old_stock_rows: list = None) -> str:
    """Copy build_catalog.py into a new temp folder with synthetic CSVs, run it there; returns that folder."""
    folder = support.make_temp_dir("catalog")
    shutil.copy(BUILD_CATALOG_SCRIPT, folder)
    write_csv(os.path.join(folder, "catalog_items.csv"), ITEM_HEADER, item_rows)
    if old_stock_rows is not None:
        write_csv(os.path.join(folder, "catalog_stock.csv"), STOCK_HEADER, old_stock_rows)
    process = subprocess.run(
        [sys.executable, os.path.join(folder, "build_catalog.py")], capture_output=True, text=True, cwd=folder
    )
    assert process.returncode == 0, process.stderr
    return folder


def stock_quantity_by_piece(folder: str) -> dict:
    return {
        (row["piece_type"], row["width_studs"], row["depth_studs"], row["color_name"], row["shape"]): (
            int(row["quantity"]),
            int(row["in_photos"]),
            int(row["manual"]),
        )
        for row in read_csv_rows(os.path.join(folder, "catalog_stock.csv"))
    }


def item(item_id: str, piece_type: str, width: int, depth: int, colour: str, shape: str = "") -> list:
    height = 3 if piece_type.startswith("Brick") else 1
    return [item_id, "1.jpg", piece_type, width, depth, height, colour, "", shape]


def test_build_catalog_counts_photographed_pieces():
    folder = run_build_catalog(
        [item("1-P01", "Plate", 2, 1, "Red"), item("1-P02", "Plate", 1, 2, "Red"), item("1-P03", "Brick", 2, 2, "Blue")]
    )
    quantities = stock_quantity_by_piece(folder)
    assert quantities[("Plate", "2", "1", "Red", "")] == (1, 1, 0)
    assert quantities[("Plate", "1", "2", "Red", "")] == (1, 1, 0), "orientation is kept as photographed"
    assert quantities[("Brick", "2", "2", "Blue", "")] == (1, 1, 0)


def test_build_catalog_keeps_hand_edits_and_backs_up_the_old_stock():
    old_stock = [["Plate", 2, 1, 1, "Red", 5, 1, 4, "", ""], ["Plate", 4, 4, 1, "Green", 2, 0, 2, "", ""]]
    folder = run_build_catalog([item("1-P01", "Plate", 2, 1, "Red"), item("2-P02", "Plate", 2, 1, "Red")], old_stock)
    quantities = stock_quantity_by_piece(folder)
    assert quantities[("Plate", "2", "1", "Red", "")] == (6, 2, 4), "a new photo adds to the hand-edited count"
    assert quantities[("Plate", "4", "4", "Green", "")] == (2, 0, 2), "hand-only lines are kept"
    backup_rows = read_csv_rows(os.path.join(folder, "catalog_stock.previous.csv"))
    assert backup_rows[0]["quantity"] == "5"


def test_build_catalog_keeps_special_shapes_on_their_own_lines():
    folder = run_build_catalog(
        [
            item("1-P01", "Plate (special)", 2, 2, "Red", "corner 2x2 (L, 3 studs)"),
            item("1-P02", "Plate (special)", 2, 2, "Red", "round plate 2x2"),
        ]
    )
    quantities = stock_quantity_by_piece(folder)
    assert len(quantities) == 2


def make_cost_case(inventory_rows: list, stock_rows: list, price_rows: list) -> tuple:
    """Temp build folder with an inventory + stock + price list; returns (build folder, stock path, prices path)."""
    folder = support.make_temp_dir("cost")
    write_csv(os.path.join(folder, "figure_inventory.csv"), INVENTORY_HEADER, inventory_rows)
    stock_path, prices_path = os.path.join(folder, "stock.csv"), os.path.join(folder, "prices.csv")
    write_csv(stock_path, STOCK_HEADER, stock_rows)
    write_csv(prices_path, PRICE_HEADER, price_rows)
    return folder, stock_path, prices_path


def run_check_cost(folder: str, stock_path: str, prices_path: str, *options: str) -> tuple:
    """Run check_cost.py; returns (stdout, cost rows keyed by (piece_type, w, d, colour))."""
    process = subprocess.run(
        [sys.executable, CHECK_COST_SCRIPT, folder, "--stock", stock_path, "--prices", prices_path, *options],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    cost_rows = [row for row in read_csv_rows(os.path.join(folder, "figure_cost.csv")) if row["piece_type"]]
    by_piece = {
        (row["piece_type"], row["width_studs"], row["depth_studs"], row["color_name"]): row
        for row in cost_rows
        if row["piece_type"] != "TOTAL"
    }
    total = next(row for row in cost_rows if row["piece_type"] == "TOTAL")
    return process.stdout, by_piece, total


def test_check_cost_prices_only_the_shortfall_and_prefers_the_colour_price():
    folder, stock_path, prices_path = make_cost_case(
        [["figure", "Plate", 2, 1, 1, "Red", 5, ""], ["figure", "Plate", 2, 1, 1, "Blue", 2, ""]],
        [["Plate", 2, 1, 1, "Red", 3, 3, 0, "", ""]],
        [["Plate", 2, 1, "", "0.50", "", ""], ["Plate", 2, 1, "Blue", "1.00", "", "blue is dearer"]],
    )
    _, by_piece, total = run_check_cost(folder, stock_path, prices_path)
    assert by_piece[("Plate", "2", "1", "Red")]["need"] == "2"
    assert by_piece[("Plate", "2", "1", "Red")]["cost_pln"] == "1.0"
    assert by_piece[("Plate", "2", "1", "Blue")]["cost_pln"] == "2.0"
    assert total["need"] == "4" and total["cost_pln"] == "3.0"


def test_check_cost_compatible_profile_falls_back_to_the_shop_price_with_a_note():
    folder, stock_path, prices_path = make_cost_case(
        [["figure", "Plate", 2, 2, 1, "Red", 1, ""]], [], [["Plate", 2, 2, "", "0.80", "", ""]]
    )
    stdout, by_piece, _ = run_check_cost(folder, stock_path, prices_path, "--profile", "compatible")
    assert by_piece[("Plate", "2", "2", "Red")]["unit_pln"] == "0.8"
    assert "no compatible price" in stdout


def test_check_cost_factor_multiplies_every_price():
    folder, stock_path, prices_path = make_cost_case(
        [["figure", "Plate", 2, 2, 1, "Red", 2, ""]], [], [["Plate", 2, 2, "", "1.00", "", ""]]
    )
    _, _, total = run_check_cost(folder, stock_path, prices_path, "--factor", "0.5")
    assert total["cost_pln"] == "1.0"


def test_check_cost_guesses_an_unpriced_size_per_stud_and_says_so():
    folder, stock_path, prices_path = make_cost_case([["figure", "Brick", 4, 2, 3, "Red", 1, ""]], [], [])
    stdout, by_piece, _ = run_check_cost(folder, stock_path, prices_path, "--fallback-per-stud", "0.1", "0.5")
    assert by_piece[("Brick", "4", "2", "Red")]["unit_pln"] == "4.0"
    assert by_piece[("Brick", "4", "2", "Red")]["price_source"] == "UNPRICED GUESS"
    assert "UNPRICED GUESS" in stdout


def test_check_cost_reports_a_folder_without_inventory():
    empty_folder = support.make_temp_dir("cost_empty")
    process = subprocess.run([sys.executable, CHECK_COST_SCRIPT, empty_folder], capture_output=True, text=True)
    assert process.returncode != 0 and "No *_inventory.csv" in process.stderr


def test_check_cost_piece_names_and_size_normalisation():
    check_cost = load_check_cost_module()
    assert check_cost.size_key("Plate ", "1", "4") == ("Plate", 4, 1)
    assert check_cost.piece_name({"piece_type": "Brick"}) == "Brick"
    assert (
        check_cost.piece_name({"piece_type": "Plate (special)", "shape": "corner 2x2 (L, 3 studs)"})
        == "Corner plate (L)"
    )
    assert check_cost.piece_name({"piece_type": "Plate (special)", "shape": "round 2x2"}) is None


if __name__ == "__main__":
    support.run_tests(globals())
