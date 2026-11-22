"""check_cost.py prices the L-shaped corner pieces too (owned ones are subtracted, others are priced from
price_list.csv). Synthetic files in a temp folder. Run: python tests/test_check_cost_l.py"""

import csv
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "my_bricks_catalog", "check_cost.py")
L_SHAPE = "corner 2x2 (L, 3 studs)"


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def run_cost(inventory_rows, stock_rows):
    """Run check_cost.py on synthetic files; returns (stdout, cost rows as dicts)."""
    folder = tempfile.mkdtemp()
    write_csv(
        os.path.join(folder, "t_inventory.csv"),
        ["section", "piece_type", "width_studs", "depth_studs", "height_plates", "color_name", "quantity", "shape"],
        inventory_rows,
    )
    write_csv(
        os.path.join(folder, "stock.csv"),
        [
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
        ],
        stock_rows,
    )
    write_csv(
        os.path.join(folder, "prices.csv"),
        ["piece_type", "width_studs", "depth_studs", "color_name", "price_lego_shop", "price_compatible", "note"],
        [
            ["Plate", 2, 1, "", "0.40", "", "t"],
            ["Corner plate (L)", 2, 2, "", "0.14", "", "t"],
            ["Corner brick (L)", 2, 2, "", "0.15", "", "t"],
        ],
    )
    out = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            folder,
            "--stock",
            os.path.join(folder, "stock.csv"),
            "--prices",
            os.path.join(folder, "prices.csv"),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows = list(csv.DictReader(open(os.path.join(folder, "t_cost.csv"), encoding="utf-8")))
    return out, [r for r in rows if r["piece_type"] not in ("", "TOTAL")]


def test_l_pieces_are_priced_and_owned_ones_subtracted():
    inventory = [
        ["figure", "Plate", 2, 1, 1, "Black", 3, ""],
        ["figure", "Plate (special)", 2, 2, 1, "Black", 10, L_SHAPE],
        ["figure", "Brick (special)", 2, 2, 3, "Orange", 4, L_SHAPE],
        ["figure", "Plate (special)", 2, 1, 1, "Black", 5, "plate 1x2 with 1 center stud"],
    ]
    stock = [
        ["Plate (special)", 2, 2, 1, "Black", 4, 4, 0, "corner plate 2x2 (L, 3 studs)", "2420"],
        ["Plate (special)", 2, 1, 1, "Black", 9, 9, 0, "plate 1x2 with 1 center stud", "3794"],
    ]
    out, rows = run_cost(inventory, stock)
    by_name = {r["piece_type"]: r for r in rows}
    assert by_name["Corner plate (L)"]["need"] == "6" and by_name["Corner plate (L)"]["cost_pln"] == "0.84"
    assert by_name["Corner brick (L)"]["need"] == "4" and by_name["Corner brick (L)"]["cost_pln"] == "0.6"
    assert by_name["Plate"]["need"] == "3"
    assert len(rows) == 3, "other special pieces must stay out of the cost"
    assert "TOTAL TO BUY: 2.64 PLN" in out


if __name__ == "__main__":
    test_l_pieces_are_priced_and_owned_ones_subtracted()
    print("ok   test_l_pieces_are_priced_and_owned_ones_subtracted\nall 1 passed")
